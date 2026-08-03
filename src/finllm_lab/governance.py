from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import timezone
from typing import Any, Callable, Mapping

from .contracts import AuditEvent, GateDecision, ToolCall
from .core import canonical_hash


@dataclass(frozen=True)
class ToolSchema:
    """Declared contract for one tool.

    Required arguments are declared per tool rather than inferred from the tool
    name. Matching on a name substring (``tool.endswith("market_snapshot")``)
    both over-fires on unrelated tools and silently misses a renamed one.
    """

    name: str
    required_arguments: frozenset[str] = frozenset()
    numeric_arguments: frozenset[str] = frozenset()
    material: bool = False
    description: str = ""


class PolicyError(ValueError):
    """Raised when a gate is asked to evaluate a call it cannot reason about."""


class PolicyGate:
    """Deterministic authorization gate. Every path returns a decision.

    Ordering matters and is deliberate: completeness and well-formedness are
    checked before authority and limits, so a malformed material call is
    rejected rather than escalated. No input may raise out of ``decide`` — a
    gate that throws on hostile input has failed open in any caller with a
    broad ``except``.
    """

    def __init__(
        self,
        permissions: Mapping[str, set[str]],
        material_tools: set[str] | None = None,
        max_notional: float = 100_000.0,
        schemas: Mapping[str, ToolSchema] | None = None,
    ):
        self.permissions = {actor: set(tools) for actor, tools in permissions.items()}
        self.material_tools = set(material_tools or set())
        self.max_notional = float(max_notional)
        self.schemas = dict(schemas or {})

    def decide(self, call: ToolCall) -> GateDecision:
        try:
            return self._decide(call)
        except Exception as exc:
            return GateDecision("reject", f"gate could not evaluate the call: {exc}")

    def _decide(self, call: ToolCall) -> GateDecision:
        args = dict(call.arguments)
        schema = self.schemas.get(call.tool)

        # 1. Completeness and well-formedness, before authority.
        if schema is not None:
            missing = sorted(schema.required_arguments - set(args))
            if missing:
                return GateDecision(
                    "reject", f"{call.tool} requires argument(s): {', '.join(missing)}"
                )
            for name in sorted(schema.numeric_arguments & set(args)):
                try:
                    float(args[name])
                except (TypeError, ValueError):
                    return GateDecision(
                        "reject", f"argument {name!r} must be numeric, got {args[name]!r}"
                    )
        if "notional" in args:
            try:
                notional = float(args["notional"])
            except (TypeError, ValueError):
                return GateDecision(
                    "reject", f"argument 'notional' must be numeric, got {args['notional']!r}"
                )
            if notional < 0:
                return GateDecision("reject", "notional must be non-negative")
        else:
            notional = None

        # 2. Authority.
        allowed = self.permissions.get(call.actor, set())
        if call.tool not in allowed:
            return GateDecision("reject", f"{call.actor} is not authorized for {call.tool}")

        # 3. Delegated limits.
        material = (
            call.tool in self.material_tools
            or call.side_effect == "material"
            or (schema is not None and schema.material)
        )
        if material:
            return GateDecision("escalate", "material action requires named human approval", args)
        if notional is not None and notional > self.max_notional:
            return GateDecision("escalate", "notional exceeds delegated limit", args)
        return GateDecision("allow", "authorized and within deterministic limits", args)


@dataclass
class ExecutionRecord:
    result: Any
    argument_fingerprint: str


class GuardedExecutor:
    """Execute gated tool calls with caller-supplied idempotency keys.

    Idempotency is keyed on ``ToolCall.request_id`` — the caller's declaration
    that two attempts are the *same* operation — not on an argument hash. An
    argument hash is a memoizer: it silently collapses two genuinely distinct
    requests that happen to share arguments. The argument hash is retained as a
    conflict detector: replaying a key with different arguments is rejected.

    The audit chain is anchored by ``genesis_hash`` and exposes ``head_hash``.
    A bare hash chain detects a mutated link but not a truncated tail or a fully
    recomputed chain; publishing the head to an append-only external store is
    what makes it tamper-evident. ``verify_chain`` reports both conditions.
    """

    def __init__(
        self,
        registry: Mapping[str, Callable[..., Any]],
        gate: PolicyGate,
        genesis: str = "finllm-lab-audit-v1",
    ):
        self.registry = dict(registry)
        self.gate = gate
        self.genesis_hash = canonical_hash({"genesis": genesis})
        self.cache: dict[str, ExecutionRecord] = {}
        self.audit: list[AuditEvent] = []

    @property
    def head_hash(self) -> str:
        """Current chain head. Publish this externally to detect truncation."""
        if not self.audit:
            return self.genesis_hash
        return canonical_hash(asdict(self.audit[-1]))

    def execute(self, call: ToolCall) -> tuple[GateDecision, Any | None]:
        argument_fingerprint = canonical_hash(
            {"actor": call.actor, "tool": call.tool, "arguments": call.arguments}
        )
        decision = self.gate.decide(call)
        result = None
        replayed = False

        if decision.status == "allow":
            cached = self.cache.get(call.request_id)
            if cached is not None:
                if cached.argument_fingerprint != argument_fingerprint:
                    decision = GateDecision(
                        "reject",
                        f"idempotency key {call.request_id!r} was already used with "
                        "different arguments",
                    )
                else:
                    result = cached.result
                    replayed = True
            elif call.tool not in self.registry:
                decision = GateDecision("reject", "tool is not registered")
            else:
                result = self.registry[call.tool](**decision.sanitized_arguments)
                self.cache[call.request_id] = ExecutionRecord(result, argument_fingerprint)

        payload = {
            "request_id": call.request_id,
            "argument_fingerprint": argument_fingerprint,
            "call": asdict(call),
            "decision": asdict(decision),
            "executed": decision.status == "allow" and not replayed,
            "replayed": replayed,
            "result_hash": canonical_hash(result) if decision.status == "allow" else None,
        }
        parent_hash = self.head_hash
        self.audit.append(
            AuditEvent(
                event_id=str(
                    uuid.uuid5(uuid.NAMESPACE_URL, argument_fingerprint + str(len(self.audit)))
                ),
                event_type="tool_execution",
                timestamp=call.requested_at.astimezone(timezone.utc),
                payload=payload,
                parent_hash=parent_hash,
            )
        )
        return decision, result

    def verify_chain(self, expected_head: str | None = None) -> dict[str, Any]:
        """Verify link integrity, the genesis anchor, and (optionally) the head.

        ``links_intact`` alone is not tamper evidence: an adversary with write
        access can recompute every link. Only ``head_matches`` against an
        externally held value detects truncation or wholesale rewriting.
        """
        links_intact = True
        expected_parent = self.genesis_hash
        for event in self.audit:
            if event.parent_hash != expected_parent:
                links_intact = False
                break
            expected_parent = canonical_hash(asdict(event))
        return {
            "events": len(self.audit),
            "anchored_to_genesis": bool(
                not self.audit or self.audit[0].parent_hash == self.genesis_hash
            ),
            "links_intact": links_intact,
            "head_hash": self.head_hash,
            "head_matches": None if expected_head is None else self.head_hash == expected_head,
            "note": (
                "links_intact detects a mutated event; only an externally held "
                "head hash detects truncation or a fully recomputed chain"
            ),
        }
