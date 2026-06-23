from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import re

from logger import logger


_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def _chunk_kind(chunk: Dict[str, Any]) -> str:
    meta = chunk.get("metadata", {}) or {}
    raw_kind = (
        meta.get("node_type")
        or meta.get("chunk_type")
        or chunk.get("node_type")
        or chunk.get("chunk_type")
        or ""
    )
    kind = str(raw_kind).lower()
    if kind in {"child", "leaf", "raw", "chunk"}:
        return "raw"
    if kind in {"parent", "section", "chapter", "paragraph"}:
        return "parent"
    if kind in {"summary", "raptor_summary", "abstract"}:
        return "summary"
    return "raw" if not kind else kind


def _chunk_text(chunk: Dict[str, Any]) -> str:
    return (chunk.get("context_text") or chunk.get("text") or "").strip()


@dataclass
class ContextPackConfig:
    default_max_tokens: int = 1500
    simple_max_tokens: int = 1100
    synthesis_max_tokens: int = 1650
    global_max_tokens: int = 1800
    max_chunks: int = 7
    max_per_document: int = 3
    max_per_parent: int = 2
    raw_bias: float = 1.0
    parent_bias: float = 0.72
    summary_bias: float = 0.55


class AdaptiveContextPacker:
    """
    Deterministically trims retrieval results to the best evidence budget.

    Design goals:
    - Prefer raw/fact chunks for direct lookup questions.
    - Let parent/section chunks participate when synthesis is needed.
    - Keep the final prompt small enough for local Ollama / Groq budgets.
    - Preserve source diversity so one document cannot crowd out all evidence.
    """

    def __init__(self, config: Optional[ContextPackConfig] = None):
        self.config = config or ContextPackConfig()

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4.0))

    def _query_profile(self, query: str) -> Dict[str, Any]:
        q = (query or "").strip()
        ql = q.lower()
        tokens = _tokenize(q)
        token_set = set(tokens)
        return {
            "query": q,
            "query_lower": ql,
            "tokens": token_set,
            "token_count": len(tokens),
            "is_fact_lookup": any(ql.startswith(prefix) for prefix in ("what is", "who is", "when is", "where is", "define", "list")),
            "is_synthesis": any(
                phrase in ql
                for phrase in ("compare", "difference", "compare", "across", "between", "summarize", "summary", "overall", "how do", "why does")
            ),
            "has_negation": bool(re.search(r"\b(not|n't|never|no|none|without|except|excluding|isn't|doesn't|don't|didn't|cannot|can't|won't|wouldn't|shouldn't)\b", ql)),
            "has_quotes": ('"' in q) or ("'" in q),
        }

    def _target_tokens(self, answer_plan: Optional[Dict[str, Any]], mode: Optional[str], chunks: Optional[Sequence[Dict[str, Any]]] = None) -> int:
        budget = (answer_plan or {}).get("budget", {}) if answer_plan else {}
        from_plan = budget.get("max_tokens_for_context")
        if isinstance(from_plan, int) and from_plan > 0:
            base_budget = max(400, min(from_plan, 2200))
        else:
            mode_l = (mode or "").lower()
            if mode_l == "doc_rag":
                base_budget = self.config.default_max_tokens
            elif mode_l == "web_rag":
                base_budget = 1200
            elif mode_l == "direct_web":
                base_budget = 900
            else:
                base_budget = self.config.default_max_tokens

        # FIX (#9): the token budget used to be a single flat number per
        # complexity tier, regardless of how dense the underlying document's
        # chunks are. A document like Newton's Principia or the Attention
        # paper needs more surrounding notation/context per "fact" to answer
        # faithfully than a sparse, repetitive document does -- a flat
        # ceiling can truncate exactly the content needed for a correct
        # answer in the dense case, while wasting budget on filler in the
        # sparse case. We use the `doc_shape` tag the chunker (chunking.py)
        # already attaches to every chunk's metadata as a cheap proxy for
        # density, since it's already computed at ingestion time -- no extra
        # work needed here beyond reading it back off the retrieved chunks.
        if chunks:
            shapes = {
                (c.get("metadata", {}) or {}).get("doc_shape") or c.get("doc_shape")
                for c in chunks
            }
            shapes.discard(None)
            if shapes & {"academic_dense", "numbered_spec_academic"}:
                # Equation/notation-dense content: widen the budget so a
                # formula and the sentence that explains it are less likely
                # to be split across the truncation boundary.
                base_budget = int(base_budget * 1.35)
            elif shapes & {"legal_clause"}:
                # Clause-dense legal text packs a lot of meaning per
                # character (cross-references, exceptions, enumerated
                # conditions) -- a modest widening reduces the chance of
                # cutting a clause's qualifying sub-clause.
                base_budget = int(base_budget * 1.15)

        return max(400, min(base_budget, 2800))

    def _score_chunk(self, query_profile: Dict[str, Any], chunk: Dict[str, Any], mode: Optional[str]) -> Tuple[float, List[str]]:
        kind = _chunk_kind(chunk)
        text = _chunk_text(chunk)
        text_lower = text.lower()
        tokens = _tokenize(text)
        token_set = set(tokens)
        query_tokens = query_profile["tokens"]

        base = 0.0
        for key in ("rerank_score", "fused_score", "score"):
            value = chunk.get(key)
            if isinstance(value, (int, float)):
                base = max(base, float(value))

        overlap = len(query_tokens & token_set)
        overlap_ratio = overlap / max(1.0, math.sqrt(len(query_tokens) or 1))
        phrase_boost = 0.0
        exact_query_match = query_profile["query_lower"] and query_profile["query_lower"] in text_lower
        if exact_query_match:
            phrase_boost += 1.5

        if query_profile["is_fact_lookup"]:
            if kind == "raw":
                base += 0.35
            elif kind == "parent":
                base -= 0.10
            elif kind == "summary":
                base -= 0.25
        elif query_profile["is_synthesis"]:
            if kind == "parent":
                base += 0.32
            elif kind == "summary":
                base += 0.20
            elif kind == "raw":
                base += 0.08
        else:
            if kind == "parent":
                base += 0.12
            elif kind == "summary":
                base += 0.05

        if mode == "global":
            if kind == "summary":
                base += 0.30
            elif kind == "parent":
                base += 0.16
        elif mode == "simple":
            if kind == "raw":
                base += 0.20
            elif kind == "parent":
                base -= 0.12

        if query_profile["has_negation"]:
            base += 0.08 if kind == "raw" else 0.02

        if query_profile["has_quotes"]:
            if any(qt in text_lower for qt in query_tokens if len(qt) > 2):
                phrase_boost += 0.2

        base += min(1.0, overlap_ratio) * 0.45
        base += phrase_boost

        # Prefer concise raw evidence for fact questions, broader context for synthesis.
        length = max(len(tokens), 1)
        if query_profile["is_fact_lookup"] and kind == "raw":
            base += max(0.0, 0.18 - (length / 2500.0))
        if query_profile["is_synthesis"] and kind in {"parent", "summary"}:
            base += min(0.22, length / 3000.0)

        reasons = [f"kind={kind}", f"base={base:.3f}", f"overlap={overlap}"]
        return base, reasons

    def _normalize_context_text(self, chunk: Dict[str, Any], prefer_parent: bool) -> str:
        kind = _chunk_kind(chunk)
        if kind in {"summary", "parent"}:
            return _chunk_text(chunk)
        if prefer_parent:
            parent_text = chunk.get("metadata", {}).get("parent_text") or chunk.get("parent_text")
            if parent_text:
                return str(parent_text)
        return _chunk_text(chunk)

    def pack(
        self,
        query: str,
        chunks: Sequence[Dict[str, Any]],
        answer_plan: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return a compact, ordered subset of chunks that fits the context budget.
        """
        if not chunks:
            return []

        query_profile = self._query_profile(query)
        effective_mode = (mode or (answer_plan or {}).get("mode") or "").lower()
        token_budget = max_tokens or self._target_tokens(answer_plan, effective_mode, chunks=chunks)
        hard_chunk_cap = self.config.max_chunks

        preferred_parent = (
            query_profile["is_synthesis"]
            or effective_mode in {"global", "multi_hop", "doc_rag"}
            or (answer_plan or {}).get("retrieval_complexity") in {"multi_hop", "global"}
        )

        unique: Dict[str, Dict[str, Any]] = {}
        ordered_candidates: List[Dict[str, Any]] = []
        for chunk in chunks:
            cid = str(chunk.get("chunk_id") or "")
            if not cid or cid in unique:
                continue
            normalized = dict(chunk)
            normalized.setdefault("metadata", chunk.get("metadata", {}) or {})
            normalized["context_text"] = self._normalize_context_text(normalized, prefer_parent=preferred_parent)
            unique[cid] = normalized
            ordered_candidates.append(normalized)

        scored: List[Tuple[float, Dict[str, Any], List[str]]] = []
        for chunk in ordered_candidates:
            score, reasons = self._score_chunk(query_profile, chunk, effective_mode)
            chunk["_pack_score"] = score
            chunk["_pack_reasons"] = reasons
            scored.append((score, chunk, reasons))

        scored.sort(key=lambda item: (
            -item[0],
            _chunk_kind(item[1]) != "raw",
            len(_tokenize(item[1].get("context_text", "")))
        ))

        selected: List[Dict[str, Any]] = []
        doc_counts: Dict[str, int] = {}
        parent_counts: Dict[str, int] = {}
        used_tokens = 0

        # Always keep at least one raw chunk when available.
        raw_seed = next((c for _, c, _ in scored if _chunk_kind(c) == "raw"), None)
        if raw_seed is not None:
            seed_tokens = self._estimate_tokens(raw_seed.get("context_text", ""))
            if seed_tokens <= token_budget:
                selected.append(raw_seed)
                doc_counts[str(raw_seed.get("metadata", {}).get("document_id", raw_seed.get("document_id", "unknown")))] = 1
                parent_counts[str(raw_seed.get("metadata", {}).get("parent_id", raw_seed.get("parent_id", "unknown")))] = 1
                used_tokens += seed_tokens

        for score, chunk, reasons in scored:
            if chunk in selected:
                continue

            kind = _chunk_kind(chunk)
            doc_id = str(chunk.get("metadata", {}).get("document_id", chunk.get("document_id", "unknown")))
            parent_id = str(chunk.get("metadata", {}).get("parent_id", chunk.get("parent_id", "unknown")))

            if doc_counts.get(doc_id, 0) >= self.config.max_per_document:
                continue
            if parent_counts.get(parent_id, 0) >= self.config.max_per_parent:
                continue

            chunk_tokens = self._estimate_tokens(chunk.get("context_text", ""))
            if not selected and chunk_tokens > token_budget:
                continue
            if used_tokens + chunk_tokens > token_budget and len(selected) >= 1:
                continue

            if len(selected) >= hard_chunk_cap:
                break

            selected.append(chunk)
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
            parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
            used_tokens += chunk_tokens

        # Backfill if the budget was too tight or diversity caps were restrictive.
        if len(selected) < min(hard_chunk_cap, len(scored)):
            for score, chunk, reasons in scored:
                if chunk in selected:
                    continue
                doc_id = str(chunk.get("metadata", {}).get("document_id", chunk.get("document_id", "unknown")))
                parent_id = str(chunk.get("metadata", {}).get("parent_id", chunk.get("parent_id", "unknown")))
                if doc_counts.get(doc_id, 0) >= self.config.max_per_document:
                    continue
                if parent_counts.get(parent_id, 0) >= self.config.max_per_parent:
                    continue
                chunk_tokens = self._estimate_tokens(chunk.get("context_text", ""))
                if used_tokens + chunk_tokens > token_budget and selected:
                    continue
                selected.append(chunk)
                doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
                parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
                used_tokens += chunk_tokens
                if len(selected) >= hard_chunk_cap or used_tokens >= token_budget:
                    break

        for idx, chunk in enumerate(selected, 1):
            chunk["packed_rank"] = idx
            chunk["packing_score"] = float(chunk.get("_pack_score", 0.0))
            chunk["packing_kind"] = _chunk_kind(chunk)
            chunk["packing_tokens"] = self._estimate_tokens(chunk.get("context_text", ""))
            chunk["packing_reasons"] = chunk.get("_pack_reasons", [])
            # Keep the original text visible to downstream code; context_text is the
            # packed evidence surface the generator will actually see.
            if not chunk.get("context_text"):
                chunk["context_text"] = _chunk_text(chunk)

        logger.info(
            "Packed %d/%d chunks for query=%r (mode=%s, tokens~%d)",
            len(selected),
            len(chunks),
            query[:120],
            effective_mode or "n/a",
            used_tokens,
        )
        return selected


__all__ = ["AdaptiveContextPacker", "ContextPackConfig"]
