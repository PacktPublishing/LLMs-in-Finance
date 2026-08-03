from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np

from .contracts import Claim, Document, Evidence, EvidenceBundle
from .temporal import filter_available
from .text import is_numeric_token, normalize_numeric_token, tokenize_financial

# Abbreviations that end in a period but do not end a sentence.
_ABBREVIATIONS = (
    "u.s",
    "u.k",
    "e.u",
    "inc",
    "corp",
    "co",
    "ltd",
    "llc",
    "plc",
    "vs",
    "approx",
    "no",
    "fig",
    "est",
    "q1",
    "q2",
    "q3",
    "q4",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sept",
    "sep",
    "oct",
    "nov",
    "dec",
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries without breaking abbreviations or decimals.

    A naive ``(?<=[.!?])\\s+`` split cuts ``$1.5 million vs. $1.2 million`` in
    two and orphans a figure into the next chunk, which then fails numerical
    claim support for a reason that has nothing to do with the model.
    """
    pieces = _SENTENCE_BOUNDARY.split(text.strip())
    sentences: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if sentences:
            previous = sentences[-1].rstrip()
            last_word = re.split(r"[\s(]", previous)[-1].rstrip(".").lower()
            ends_in_abbreviation = last_word in _ABBREVIATIONS
            ends_in_initial = bool(re.search(r"\b[A-Za-z]\.$", previous))
            if ends_in_abbreviation or ends_in_initial:
                sentences[-1] = previous + " " + piece
                continue
        sentences.append(piece)
    return sentences


def chunk_financial_document(
    document: Document, max_tokens: int = 90, overlap_sentences: int = 1
) -> list[Document]:
    """Chunk a document on sentence boundaries with a small sentence overlap.

    Overlap keeps a figure and the sentence that qualifies it in the same chunk
    often enough to matter. Sentences longer than ``max_tokens`` are split on
    token boundaries rather than silently overflowing the budget.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be non-negative")

    sentences: list[str] = []
    for sentence in split_sentences(document.text):
        tokens = tokenize_financial(sentence)
        if len(tokens) <= max_tokens:
            sentences.append(sentence)
            continue
        words = sentence.split()
        step = max(1, max_tokens // 2)
        for start in range(0, len(words), step):
            fragment = " ".join(words[start : start + step])
            if fragment:
                sentences.append(fragment)

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        sentence_size = len(tokenize_financial(sentence))
        if current and size + sentence_size > max_tokens:
            chunks.append(" ".join(current))
            carry = current[-overlap_sentences:] if overlap_sentences else []
            current = list(carry)
            size = sum(len(tokenize_financial(item)) for item in current)
        current.append(sentence)
        size += sentence_size
    if current:
        chunks.append(" ".join(current))

    return [
        Document(
            document_id=f"{document.document_id}::c{index:02d}",
            text=text,
            source=document.source,
            available_at=document.available_at,
            issuer=document.issuer,
            section=document.section,
            metadata={**dict(document.metadata), "parent_id": document.document_id},
        )
        for index, text in enumerate(chunks)
    ]


class BM25Index:
    """Small, inspectable BM25 index with point-in-time corpus statistics.

    Filtering candidates by availability is not sufficient for a point-in-time
    retriever: inverse document frequency and average document length are corpus
    statistics, so an index built over the full corpus scores today's query with
    tomorrow's vocabulary. Pass ``decision_time`` at construction, or to
    ``search``, and the index is rebuilt over the admissible subset before any
    statistic is computed.
    """

    def __init__(
        self,
        documents: Sequence[Document],
        k1: float = 1.5,
        b: float = 0.75,
        decision_time: datetime | None = None,
    ):
        self._all_documents = tuple(documents)
        self.k1 = k1
        self.b = b
        self.decision_time = decision_time
        self.documents = (
            filter_available(self._all_documents, decision_time)
            if decision_time is not None
            else self._all_documents
        )
        self.tokens = [tokenize_financial(d.text) for d in self.documents]
        self.lengths = np.array([len(tokens) for tokens in self.tokens], dtype=float)
        self.avg_length = float(self.lengths.mean()) if len(self.lengths) else 0.0
        self.term_counts = [Counter(tokens) for tokens in self.tokens]
        self.doc_frequency: Counter = Counter()
        for counts in self.term_counts:
            self.doc_frequency.update(counts.keys())
        self._as_of_cache: dict[datetime, "BM25Index"] = {}

    def as_of(self, decision_time: datetime) -> "BM25Index":
        """Return an index whose statistics use only documents available then."""
        if decision_time == self.decision_time:
            return self
        if decision_time not in self._as_of_cache:
            self._as_of_cache[decision_time] = BM25Index(
                self._all_documents, self.k1, self.b, decision_time=decision_time
            )
        return self._as_of_cache[decision_time]

    def _idf(self, term: str) -> float:
        n = len(self.documents)
        df = self.doc_frequency[term]
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> np.ndarray:
        if not self.documents:
            return np.array([], dtype=float)
        scores = np.zeros(len(self.documents), dtype=float)
        for term in tokenize_financial(query):
            idf = self._idf(term)
            for i, counts in enumerate(self.term_counts):
                tf = counts[term]
                if tf == 0:
                    continue
                norm = tf + self.k1 * (
                    1 - self.b + self.b * self.lengths[i] / max(self.avg_length, 1.0)
                )
                scores[i] += idf * tf * (self.k1 + 1) / norm
        return scores

    def search(
        self,
        query: str,
        k: int = 5,
        decision_time: datetime | None = None,
        issuer: str | None = None,
        section: str | None = None,
        period: str | None = None,
    ) -> list[Evidence]:
        if decision_time is not None:
            return self.as_of(decision_time).search(
                query, k=k, issuer=issuer, section=section, period=period
            )
        candidates = list(range(len(self.documents)))
        if issuer:
            candidates = [
                i for i in candidates if self.documents[i].issuer.casefold() == issuer.casefold()
            ]
        if section:
            candidates = [
                i
                for i in candidates
                if self.documents[i].section.casefold() == section.casefold()
            ]
        if period:
            candidates = [
                i
                for i in candidates
                if str(self.documents[i].metadata.get("period", "")).casefold()
                == period.casefold()
            ]
        raw = self.scores(query)
        ranked = sorted(candidates, key=lambda i: (-raw[i], self.documents[i].document_id))[:k]
        max_score = max((raw[i] for i in ranked), default=0.0)
        return [
            Evidence(
                evidence_id=f"ev::{self.documents[i].document_id}",
                document_id=self.documents[i].document_id,
                passage=self.documents[i].text,
                available_at=self.documents[i].available_at,
                score=float(raw[i] / max_score) if max_score > 0 else 0.0,
                raw_score=float(raw[i]),
                source=self.documents[i].source,
            )
            for i in ranked
        ]


def lexical_support(
    claim: Claim, evidence: Sequence[Evidence], threshold: float = 0.42
) -> bool:
    """Lexical proxy for claim support: token overlap plus numeric agreement.

    This is a deliberately shallow proxy, not entailment. It cannot detect a
    claim that copies the right words in the wrong relation, and it will reject
    a correct paraphrase whose wording diverges far enough. Numeric tokens are
    normalized so that ``$1.5 million`` and ``1.5 million`` agree; the remaining
    false negatives are inherent to the method and are shown in Lab 04.
    """
    claim_tokens = {
        t for t in tokenize_financial(claim.text) if len(t) > 2 or is_numeric_token(t)
    }
    if not claim_tokens:
        return False
    evidence_tokens: set[str] = set()
    for item in evidence:
        evidence_tokens.update(tokenize_financial(item.passage))
    evidence_numbers = {
        normalize_numeric_token(t) for t in evidence_tokens if is_numeric_token(t)
    }
    overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens)
    claim_numbers = {
        normalize_numeric_token(t) for t in claim_tokens if is_numeric_token(t)
    }
    numerical_ok = not claim_numbers or claim_numbers.issubset(evidence_numbers)
    if claim_numbers:
        non_numeric = {t for t in claim_tokens if not is_numeric_token(t)}
        overlap = (
            len(non_numeric & evidence_tokens) / len(non_numeric) if non_numeric else 1.0
        )
    return overlap >= threshold and numerical_ok


def faithfulness_score(bundles: Iterable[EvidenceBundle]) -> float:
    material = [bundle for bundle in bundles if bundle.claim.material]
    if not material:
        return 1.0
    return float(
        np.mean([lexical_support(bundle.claim, bundle.evidence) for bundle in material])
    )
