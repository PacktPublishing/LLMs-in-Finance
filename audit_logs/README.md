# Audit logs

The notebooks write only deterministic teaching examples. Production audit logs
should be immutable, access-controlled, privacy-reviewed, and retained according
to institutional policy.

Material runs should record:

- model, prompt, corpus, tool, and policy versions;
- the decision time and every evidence availability time;
- canonical request hashes and idempotency keys;
- human approvals and overrides;
- the output hash and release decision.

Local experimental logs belong in `audit_logs/local/`, which is ignored by Git.

The trusted inputs and exclusion boundary for the v1.0.1 integrity rebuild are
recorded in [`REBUILD_PROVENANCE.md`](REBUILD_PROVENANCE.md).
