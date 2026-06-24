# from typing import List, Dict, Any
# from vector_store import VectorStore
# from bm25_store import BM25Store
# from router import QueryRouter
# from reranker import Reranker
# from logger import logger

# # Minimum RRF fused score to keep a chunk. With k=120 the max possible score
# # per list is 1/(120+1) ≈ 0.0083. A chunk that ranks in the top half of BOTH
# # lists scores ~0.0083 + 0.0040 = 0.0123. Setting the floor just below that
# # (0.009) drops chunks that only appear in one list at a mediocre rank — the
# # main source of context-precision noise.
# RRF_SCORE_THRESHOLD = 0.009

# # RRF k parameter. The default of 60 over-weights BM25's term-frequency
# # advantage for common tokens (e.g. "length", "array") because low-k RRF
# # gives a steep bonus to rank-1. Raising k to 120 flattens the curve so a
# # rank-1 BM25 hit for a common term doesn't overwhelm a rank-2 dense hit
# # for the semantically correct chunk.
# RRF_K = 120


# class HybridRetriever:
#     def __init__(self, vector_store: VectorStore, bm25_store: BM25Store):
#         self.vector_store = vector_store
#         self.bm25_store = bm25_store
#         self.router = QueryRouter()
#         self.reranker = Reranker()

#     def _reciprocal_rank_fusion(
#         self,
#         results_lists: List[List[Dict[str, Any]]],
#         k: int = RRF_K,
#         score_threshold: float = RRF_SCORE_THRESHOLD,
#     ) -> List[Dict[str, Any]]:
#         """Fuses multiple ranked lists using RRF.

#         k controls how steeply rank-1 is rewarded; higher k = flatter curve
#         (less BM25 dominance for common-token queries). score_threshold drops
#         chunks whose fused score falls below the floor before they reach the
#         reranker, improving context precision without hurting recall on
#         queries where the relevant chunk ranks highly in at least one list.
#         """
#         fused_scores = {}
#         doc_map = {}

#         for results in results_lists:
#             for rank, doc in enumerate(results):
#                 doc_id = doc["chunk_id"]
#                 if doc_id not in fused_scores:
#                     fused_scores[doc_id] = 0.0
#                     doc_map[doc_id] = doc
#                 fused_scores[doc_id] += 1.0 / (k + rank + 1)

#         # Sort by fused score descending
#         reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

#         final_results = []
#         for doc_id, score in reranked:
#             if score < score_threshold:
#                 # Everything after this is also below threshold (sorted desc)
#                 logger.debug(
#                     f"RRF threshold filter: dropping {len(reranked) - len(final_results) - 1} "
#                     f"low-score chunks (score={score:.5f} < threshold={score_threshold})"
#                 )
#                 break
#             doc = doc_map[doc_id].copy()
#             doc["score"] = score
#             final_results.append(doc)

#         return final_results

#     def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#         route = self.router.route(query)
#         logger.info(f"Retrieval route selected: {route}")

#         candidates = []

#         if route == "bm25":
#             candidates = self.bm25_store.search(query, top_k=top_k * 2)
#         elif route == "dense":
#             candidates = self.vector_store.search(query, top_k=top_k * 2)
#         else:  # hybrid
#             bm25_results = self.bm25_store.search(query, top_k=top_k * 2)
#             dense_results = self.vector_store.search(query, top_k=top_k * 2)
#             # RRF with tuned k + threshold filter (fixes q9 BM25 over-weighting)
#             candidates = self._reciprocal_rank_fusion([bm25_results, dense_results])

#         # Optional Reranking
#         final_results = self.reranker.rerank(query, candidates, top_k=top_k)

#         # Parent resolution: Instead of returning the child chunk's text,
#         # we provide the parent_text to the generator to give broader context,
#         # but keep the child metadata for precise citations.
#         for doc in final_results:
#             if "parent_text" in doc.get("metadata", {}):
#                 doc["context_text"] = doc["metadata"]["parent_text"]
#             else:
#                 doc["context_text"] = doc["text"]

#         return final_results


"""
advanced_hybrid_retriever.py

Drop-in upgrade for the current HybridRetriever.

What this does:
- Query-aware retrieval budgeting
- Parallel BM25 + dense retrieval
- Query-variant expansion for better lexical/semantic coverage
- Weighted Reciprocal Rank Fusion (RRF)
- Always-on cross-encoder reranking on a small shortlist
- Parent-context expansion for generation
- Greedy diversity filtering to avoid duplicate chunks from the same parent/doc
- Retrieval trace storage for debugging/telemetry

Usage:
    from advanced_hybrid_retriever import HybridRetriever
    retriever = HybridRetriever(vector_store, bm25_store)
"""

from __future__ import annotations

import re
import time
import math
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from logger import logger
from vector_store import VectorStore
from bm25_store import BM25Store
from router import QueryRouter

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - optional dependency guard
    CrossEncoder = None

# Module-level singleton so the cross-encoder weights are loaded exactly once
# per process, regardless of how many HybridRetriever instances are created
# (the eval loop creates a fresh instance per document).
_GLOBAL_CROSS_ENCODER: Optional["CrossEncoder"] = None  # type: ignore[type-arg]


def _get_cross_encoder(model_name: str) -> "CrossEncoder":
    global _GLOBAL_CROSS_ENCODER
    if _GLOBAL_CROSS_ENCODER is None:
        if CrossEncoder is None:
            raise RuntimeError("sentence_transformers.CrossEncoder is not available")
        logger.info("Loading cross-encoder reranker...")
        _GLOBAL_CROSS_ENCODER = CrossEncoder(model_name)
        logger.info("Cross-encoder reranker loaded.")
    return _GLOBAL_CROSS_ENCODER


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class RetrievalConfig:
    # First-stage budgets
    final_top_k: int = 5
    fused_candidate_cap: int = 30
    rerank_pool_size: int = 16

    # Route-dependent retrieval budgets
    bm25_primary_multiplier: int = 4
    dense_primary_multiplier: int = 4
    bm25_secondary_multiplier: int = 2
    dense_secondary_multiplier: int = 2

    # Fusion / filtering
    rrf_k: int = 120
    rrf_threshold: float = 0.009

    # Diversity controls
    max_per_parent: int = 2
    max_per_document: int = 3

    # Reranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_batch_size: int = 16

    # Variant generation
    max_query_variants: int = 4

    # RAPTOR / summary-node controls
    summary_fusion_weight: float = 0.35
    summary_reserve_k: int = 3


@dataclass
class QueryFeatures:
    has_numbers: bool = False
    has_quotes: bool = False
    has_capitalized_tokens: bool = False
    is_comparative: bool = False
    is_semantic: bool = False
    has_code_tokens: bool = False
    has_negation: bool = False
    word_count: int = 0


@dataclass
class RetrievalTrace:
    query: str
    route: str
    features: Dict[str, Any] = field(default_factory=dict)
    variants: List[str] = field(default_factory=list)
    stage_timings_ms: Dict[str, float] = field(default_factory=dict)
    candidate_counts: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    # Actual candidate lists for debugging/analysis. Previously these keys
    # didn't exist on this dataclass at all, so every consumer that called
    # raw_trace.get("embedding_candidates", []) (used by HybridRetrieverWrapper,
    # RoutedHybridRetrieverWrapper, RaptorRetrieverWrapper in rag_eval_v3.py)
    # silently got an empty list back for every query, even though retrieval
    # itself worked correctly -- this made it impossible to inspect what was
    # actually retrieved for any architecture except naive_dense, and zeroed
    # out any downstream metric that depends on retrieved candidate metadata.
    embedding_candidates: List[Dict[str, Any]] = field(default_factory=list)
    bm25_candidates: List[Dict[str, Any]] = field(default_factory=list)
    after_rrf: List[Dict[str, Any]] = field(default_factory=list)
    after_rerank: List[Dict[str, Any]] = field(default_factory=list)
    submitted_to_llm: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _summarize_candidates_for_trace(docs: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    """Compact, JSON-safe summary of retrieved candidates for trace/debugging
    purposes. Keeps just enough to inspect what was actually retrieved
    (chunk id, page, score, a text preview) without bloating the trace with
    full chunk bodies."""
    out: List[Dict[str, Any]] = []
    for doc in docs[:limit]:
        meta = doc.get("metadata", {}) or {}
        text = doc.get("context_text") or doc.get("text", "") or ""
        out.append({
            "chunk_id": doc.get("chunk_id", ""),
            "page": meta.get("page_start", meta.get("page", 0)),
            "score": float(doc.get("score", doc.get("fused_score", doc.get("rerank_score", 0.0))) or 0.0),
            "node_type": meta.get("node_type", doc.get("node_type", "raw")),
            "text_preview": text[:200],
        })
    return out


# FIX (#1): these used to be defined here AND independently in
# query_understanding.py as a byte-for-byte duplicate, with nothing keeping
# the two in sync. Now imported from a single shared module -- see
# domain_patterns.py for the full rationale and history (the rfc_q1
# regression this guard exists to prevent).
from domain_patterns import (
    LEGAL_OR_POLARITY_RE as _LEGAL_OR_POLARITY_RE,
    LEGAL_AMBIGUOUS_RE as _LEGAL_AMBIGUOUS_RE,
    TECHNICAL_DOMAIN_RE as _TECHNICAL_DOMAIN_RE,
    is_legal_or_polarity as _is_legal_or_polarity,
    NEGATION_RE as _NEGATION_RE,
    has_negation_signal as _has_negation_signal,
)


# ---------------------------------------------------------------------
# Main retriever
# ---------------------------------------------------------------------

# class HybridRetriever:
#     """
#     Drop-in replacement for the current HybridRetriever.
#     Preserves:
#         - __init__(vector_store, bm25_store)
#         - retrieve(query, top_k=5) -> List[Dict[str, Any]]
#         - route_for(query) -> str
#     """

#     def __init__(
#         self,
#         vector_store: VectorStore,
#         bm25_store: BM25Store,
#         config: Optional[RetrievalConfig] = None,
#     ):
#         self.vector_store = vector_store
#         self.bm25_store = bm25_store
#         self.router = QueryRouter()
#         self.config = config or RetrievalConfig()

#         self._cross_encoder = None
#         self.last_trace: Optional[RetrievalTrace] = None

class HybridRetriever:
    def __init__(self, vector_store, bm25_store, config=None):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.router = QueryRouter()
        self.config = config or RetrievalConfig()

        class _RerankerCompat:
            def __init__(self, outer):
                self._outer = outer

            def _load_model(self):
                self._outer._load_reranker()

            def rerank(self, query, candidates, top_k=5):
                return self._outer._rerank(query, candidates, top_k)

        self.reranker = _RerankerCompat(self)
        self._cross_encoder = None
        self.last_trace = None

    # -----------------------------
    # Query understanding
    # -----------------------------
    def route_for(self, query: str) -> str:
        return self.router.route(query)

    def _should_force_standard_pipeline(self, query: str, complexity: Optional[str]) -> bool:
        """Conservative override for short but clause-heavy legal questions."""
        if complexity != "simple":
            return False
        q = (query or "").strip()
        if not q:
            return False

        ql = q.lower()
        # FIX (#1): broadened from explicit-negation-words-only to also catch
        # implicit negative-polarity verb patterns ("lacks the authority to",
        # "is barred from", "has no power to") -- see domain_patterns.py.
        if _has_negation_signal(ql):
            return True
        if _is_legal_or_polarity(ql):
            return True

        # Extra guard for short constitutional-style questions that use ordinary words
        # but still require exact clause matching.
        legal_hints = (
            "what right", "which branch", "what branch", "what does article",
            "what senate vote", "vote threshold", "inferior federal courts",
            "prohibit", "apportion", "succession", "treaty", "ratification",
        )
        return any(hint in ql for hint in legal_hints)

    def _analyze_query(self, query: str) -> QueryFeatures:
        q = query or ""
        ql = q.lower()

        features = QueryFeatures(
            has_numbers=bool(re.search(r"\d", q)),
            has_quotes=('"' in q) or ("'" in q),
            has_capitalized_tokens=bool(re.search(r"\b[A-Z][a-zA-Z0-9_]{1,}\b", q)),
            is_comparative=any(k in ql for k in [" compare ", " comparison", " difference", " vs ", " versus ", " better ", " which one"]),
            is_semantic=any(k in ql for k in ["what is", "why", "how", "explain", "describe", "summarize", "meaning of", "definition of"]),
            has_code_tokens=bool(re.search(r"[\(\)\[\]{}<>:=_/\\.-]", q)) or bool(re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\(\)", q)),
            # FIX (#1): was a fourth inline duplicate of the explicit-negation
            # regex, and explicit-only (missed "lacks the authority to" /
            # "is barred from" style implicit negative polarity). Now uses
            # the shared, broader signal -- see domain_patterns.py.
            has_negation=_has_negation_signal(ql),
            word_count=len(q.split()),
        )
        return features

    def _generate_query_variants(self, query: str, features: QueryFeatures) -> List[str]:
        q = _normalize_ws(query)
        variants = [q]

        # Lowercased / normalized variant helps lexical search when punctuation is noisy.
        stripped = re.sub(r"[^\w\s]+", " ", q)
        stripped = _normalize_ws(stripped)
        if stripped and stripped.lower() != q.lower():
            variants.append(stripped)

        # Hyphen and slash normalized variant for technical queries.
        tech_norm = re.sub(r"[-/]", " ", stripped if stripped else q)
        tech_norm = _normalize_ws(tech_norm)
        if tech_norm and tech_norm.lower() not in {v.lower() for v in variants}:
            variants.append(tech_norm)

        # Negation queries benefit from an affirmative counterpart so the retriever
        # can still surface the underlying clause, then let the generator answer
        # with polarity awareness. Only strips explicit negation WORDS
        # (NEGATION_RE) -- implicit polarity phrases like "lacks the
        # authority to" can't be cleanly word-stripped into an affirmative
        # form the same way, so features.has_negation (which also covers
        # those) is intentionally broader than what this specific step acts on.
        if features.has_negation:
            neg_free = _NEGATION_RE.sub(" ", q)
            neg_free = _normalize_ws(neg_free)
            if neg_free and neg_free.lower() not in {v.lower() for v in variants}:
                variants.append(neg_free)

        # Legal/clause queries benefit from an extremely compact clause-focused variant.
        if _is_legal_or_polarity(q.lower()):
            clause_focus = q
            clause_focus = re.sub(r"(?i)\b(according to|what does|what is|what are|how does|which branch does|which branch|what right does|what senate vote threshold does|what vote threshold does)\b", "", clause_focus)
            clause_focus = _normalize_ws(clause_focus)
            if clause_focus and clause_focus.lower() not in {v.lower() for v in variants}:
                variants.append(clause_focus)

        # Comparison rewrite: "X vs Y" -> "difference between X and Y"
        if features.is_comparative:
            comp = q
            comp = re.sub(r"\bvs\b", "versus", comp, flags=re.IGNORECASE)
            comp = comp.replace(" versus ", " and ")
            comp = comp.replace(" compare ", " comparison ")
            variants.append(_normalize_ws(comp))

            diff = re.sub(r"\b(compare|comparison|difference between|difference)\b", "difference between", q, flags=re.IGNORECASE)
            variants.append(_normalize_ws(diff))

        return _dedupe_preserve_order(variants)[: self.config.max_query_variants]

    # -----------------------------
    # First-stage retrieval
    # -----------------------------
    def _route_budgets(self, route: str, top_k: int) -> Tuple[int, int]:
        if route == "bm25":
            return max(top_k * self.config.bm25_primary_multiplier, 12), max(top_k * self.config.dense_secondary_multiplier, 6)
        if route == "dense":
            return max(top_k * self.config.dense_primary_multiplier, 12), max(top_k * self.config.bm25_secondary_multiplier, 6)
        # hybrid / math / unknown: symmetric
        budget = max(top_k * self.config.bm25_primary_multiplier, 12)
        return budget, budget

    def _search_pair(
        self, query: str, bm25_k: int, dense_k: int, node_types: List[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Runs BM25 and dense retrieval in parallel.

        `node_types`, if given, is forwarded to both stores to restrict the
        search to chunks/nodes whose `node_type` metadata matches (e.g.
        ["summary"] to search only RAPTOR summary nodes for "global" queries).
        Left as None for the default behavior of searching everything,
        identical to pre-RAPTOR retrieval.
        """
        with ThreadPoolExecutor(max_workers=2) as ex:
            bm25_kwargs = {"node_types": node_types} if node_types else {}
            dense_kwargs = {"node_types": node_types} if node_types else {}
            futures = {
                ex.submit(self.bm25_store.search, query, bm25_k, **bm25_kwargs): "bm25",
                ex.submit(self.vector_store.search, query, dense_k, **dense_kwargs): "dense",
            }
            bm25_results: List[Dict[str, Any]] = []
            dense_results: List[Dict[str, Any]] = []
            for fut in as_completed(futures):
                label = futures[fut]
                try:
                    results = fut.result() or []
                    if label == "bm25":
                        bm25_results = results
                    else:
                        dense_results = results
                except Exception as e:
                    logger.warning(f"{label.upper()} retrieval failed for '{query}': {e}")
                    if label == "bm25":
                        bm25_results = []
                    else:
                        dense_results = []
        return bm25_results, dense_results

    # -----------------------------
    # Fusion
    # -----------------------------
    def _weighted_rrf(
        self,
        ranked_lists: List[Tuple[List[Dict[str, Any]], float]],
        k: int,
        score_threshold: float,
    ) -> List[Dict[str, Any]]:
        fused_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        for results, weight in ranked_lists:
            for rank, doc in enumerate(results):
                chunk_id = doc["chunk_id"]
                doc_map.setdefault(chunk_id, doc)
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (weight / (k + rank + 1))

        ordered = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        output: List[Dict[str, Any]] = []
        for chunk_id, fused_score in ordered:
            if fused_score < score_threshold:
                break
            doc = dict(doc_map[chunk_id])
            doc["score"] = float(fused_score)
            doc["fused_score"] = float(fused_score)
            output.append(doc)
        return output

    def _query_weights(self, route: str, features: QueryFeatures) -> Dict[str, float]:
        """
        Heuristic weights for lexical vs semantic signals.
        """
        bm25_w = 1.0
        dense_w = 1.0

        if features.has_numbers or features.has_quotes or features.has_code_tokens or features.has_capitalized_tokens:
            bm25_w += 0.25
        if features.is_semantic:
            dense_w += 0.20
        if features.is_comparative:
            bm25_w += 0.10
            dense_w += 0.10
        if features.has_negation:
            bm25_w += 0.20
            dense_w += 0.05

        if route == "bm25":
            bm25_w += 0.35
        elif route == "dense":
            dense_w += 0.35

        return {"bm25": bm25_w, "dense": dense_w}

    def _should_include_summary_nodes(
        self,
        route: str,
        features: QueryFeatures,
        complexity: Optional[str],
        use_summary_nodes: bool,
    ) -> bool:
        """Gate RAPTOR summaries so they help synthesis queries without leaking into
        the plain hybrid baseline.

        The caller must opt in via ``use_summary_nodes``. We then restrict the
        summaries to genuinely multi-hop / synthesis-style retrievals.
        """
        if not use_summary_nodes:
            return False
        if complexity == "global":
            return True
        if complexity == "multi_hop":
            if route == "bm25" and not (features.has_negation or features.is_comparative):
                return False
            # E-4 FIX: tightened the trigger condition.
            # Old: word_count >= 10 OR has_negation OR is_comparative OR is_semantic
            # Problem: is_semantic fires on almost any "what is / how / why" question
            # (simple lookups) causing verbose-but-simple questions to invoke the
            # expensive 4-variant multi-hop path, receive incoherent packed context
            # from 4 disconnected sections, and hallucinate to fill gaps.
            # This is the primary driver of the 94.3% -> 70.1% faithfulness collapse.
            # New rule: require at least ONE genuinely complex signal:
            #   - word_count >= 12  (likely multi-clause)
            #   - is_comparative    (explicit vs / compare / difference)
            #   - negation that is NOT also a plain semantic lookup
            return (
                features.word_count >= 12
                or features.is_comparative
                or (features.has_negation and not features.is_semantic)
            )
        if complexity == "simple":
            # RAPTOR-specific opt-in for large documents: conservative so the
            # summary tree helps synthesis without flooding short atomic lookups.
            if route == "bm25":
                return False
            # E-4 FIX: raised from 8->12 and removed is_semantic as standalone
            # trigger for the same reason as the multi_hop branch above.
            return features.word_count >= 12 or features.is_comparative
        return False


    @staticmethod
    def _merge_candidate_lists(
        primary: List[Dict[str, Any]],
        secondary: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        merged: List[Dict[str, Any]] = []
        for doc in primary + secondary:
            chunk_id = doc.get("chunk_id")
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            merged.append(doc)
        return merged

    @staticmethod
    def _sort_summary_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def _level(doc: Dict[str, Any]) -> int:
            try:
                return int(doc.get("metadata", {}).get("raptor_level", 0))
            except (TypeError, ValueError):
                return 0

        return sorted(
            candidates,
            key=lambda d: (-_level(d), -float(d.get("score", d.get("fused_score", 0.0)))),
        )
    
    # -----------------------------
    # Reranking
    # -----------------------------
    def _load_reranker(self):
        # Delegates to the module-level singleton so weights are loaded only
        # once even when a new HybridRetriever is created per document.
        self._cross_encoder = _get_cross_encoder(self.config.reranker_model)

    def _rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        shortlist = candidates[: max(self.config.rerank_pool_size, top_k)]
        if len(shortlist) <= 1:
            return shortlist[:top_k]

        try:
            self._load_reranker()
            # E-6 FIX: use context_text (parent-expanded text the LLM sees)
            # instead of the raw child text field when scoring pairs.
            pairs = [[query, doc.get("context_text") or doc.get("text", "")] for doc in shortlist]

            scores = self._cross_encoder.predict(pairs, batch_size=self.config.rerank_batch_size)
            for i, doc in enumerate(shortlist):
                doc["rerank_score"] = float(scores[i])
            self._last_rerank_failed = False
            return sorted(shortlist, key=lambda x: x.get("rerank_score", x.get("score", 0.0)), reverse=True)[:top_k]
        except Exception as e:
            # FIX (#4): a reranker failure used to just log a warning and
            # silently return fused-order results, which look identical to a
            # successful-but-low-confidence rerank to any downstream
            # consumer. Track it explicitly so it can be surfaced in the
            # retrieval trace (same pattern as _last_search_pair_failures)
            # rather than only existing as a log line.
            logger.error(f"[RERANK DEGRADED] Cross-encoder reranking failed, falling back to fused order: {e}")
            self._last_rerank_failed = True
            return shortlist[:top_k]

    # -----------------------------
    # Diversity / context expansion
    # -----------------------------
    def _diversify(self, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        selected: List[Dict[str, Any]] = []
        doc_counts = defaultdict(int)
        parent_counts = defaultdict(int)

        # First pass: enforce diversity caps.
        for doc in candidates:
            meta = doc.get("metadata", {})
            doc_id = str(meta.get("document_id", "unknown"))
            parent_id = str(meta.get("parent_id", doc.get("parent_id", "unknown")))

            if doc_counts[doc_id] >= self.config.max_per_document:
                continue
            if parent_counts[parent_id] >= self.config.max_per_parent:
                continue

            selected.append(doc)
            doc_counts[doc_id] += 1
            parent_counts[parent_id] += 1

            if len(selected) >= top_k:
                return selected

        # Backfill if diversity caps were too strict.
        if len(selected) < top_k:
            for doc in candidates:
                if doc in selected:
                    continue
                selected.append(doc)
                if len(selected) >= top_k:
                    break

        return selected[:top_k]

    def _expand_parent_context(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        meta = doc.get("metadata", {}) or {}
        parent_text = meta.get("parent_text")
        if not parent_text:
            parent_id = meta.get("parent_id")
            if parent_id and hasattr(self.vector_store, "parent_cache"):
                parent_text = self.vector_store.parent_cache.get(parent_id)

        expanded = dict(doc)
        if parent_text:
            expanded["context_text"] = parent_text
        else:
            expanded["context_text"] = doc.get("text", "")

        return expanded

    # -----------------------------
    # Public API
    # -----------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        complexity: str = None,
        use_summary_nodes: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        `complexity`, if given, is one of "simple" / "multi_hop" / "global"
        (see query_understanding.py's classify_retrieval_complexity). It's an
        optional gate on top of the existing route_for() lexical/dense
        routing:

          - "simple":    skip multi-query variant generation and the
                          cross-encoder reranker. Single dense-only lookup,
                          fused-order results. This is the fast path for
                          single-fact lookups where the extra stages mostly
                          add latency without changing the top result.
          - "global":    search RAPTOR summary nodes (node_type="summary")
                          instead of raw chunks, biased toward the highest
                          available abstraction level. Falls back to the
                          normal full pipeline over raw chunks if no summary
                          nodes exist for this document (e.g. RAPTOR wasn't
                          run, or the document was too short to cluster).
          - "multi_hop" or None (default): unchanged -- the existing full
                          RAG-Fusion pipeline (multi-query, hybrid RRF,
                          rerank, diversify) over raw chunks. Passing no
                          complexity argument at all preserves the exact
                          pre-RAPTOR retrieve() behavior.
        """
        if complexity == "simple" and self._should_force_standard_pipeline(query, complexity):
            logger.info("Simple path overridden to full pipeline for clause-heavy / legal / negation query.")
            complexity = "multi_hop"

        if complexity == "simple":
            return self._retrieve_simple(query, top_k)
        if complexity == "global":
            global_results = self._retrieve_global(query, top_k)
            if global_results is not None:
                return global_results
            # No RAPTOR tree for this document -- fall through to the
            # standard pipeline below rather than returning nothing.

        return self._retrieve_standard(query, top_k, complexity=complexity, use_summary_nodes=use_summary_nodes)

    # -----------------------------
    # Complexity-gated retrieval paths
    # -----------------------------
    # NOTE: _retrieve_simple, _retrieve_global, _retrieve_standard, and
    # _expand_parent_context defined in this class body are overridden at
    # module import time by the monkey-patch assignments near the bottom of
    # this file (HybridRetriever._retrieve_standard = _hr_retrieve_standard,
    # etc). The free-function `_hr_*` versions are what actually execute at
    # runtime. Both copies are kept in sync for trace/candidate logging, but
    # if you need to change retrieval behavior, edit the `_hr_*` functions.
    def _retrieve_simple(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fast path for simple factual lookups.

        The old implementation used dense-only retrieval, which missed exact
        clause wording and short legal facts too often. This version keeps the
        path lightweight, but uses a small BM25+dense fusion so the answer can
        still recover explicit phrasing when the embedding signal is noisy.
        """
        t0 = _now_ms()
        trace = RetrievalTrace(query=query, route="dense_fast_path", variants=[query])

        bm25_results, dense_results = self._search_pair(
            query,
            bm25_k=max(top_k * 2, top_k),
            dense_k=max(top_k * 2, top_k),
        )
        trace.candidate_counts["bm25"] = len(bm25_results)
        trace.candidate_counts["dense"] = len(dense_results)
        trace.bm25_candidates = _summarize_candidates_for_trace(bm25_results)
        trace.embedding_candidates = _summarize_candidates_for_trace(dense_results)
        trace.notes.append("complexity=simple: used small hybrid fusion without multi-query expansion.")

        fused = self._weighted_rrf(
            ranked_lists=[(bm25_results, 1.0), (dense_results, 1.0)],
            k=self.config.rrf_k,
            score_threshold=min(self.config.rrf_threshold, 0.0),
        )
        ordered = fused if fused else (dense_results or bm25_results)
        trace.after_rrf = _summarize_candidates_for_trace(ordered)

        final_results = []
        for doc in ordered[:top_k]:
            doc = self._expand_parent_context(doc)
            doc["retrieval_route"] = "dense_fast_path"
            doc["retrieval_variants"] = [query]
            final_results.append(doc)

        trace.stage_timings_ms["total"] = _now_ms() - t0
        trace.candidate_counts["final"] = len(final_results)
        trace.submitted_to_llm = _summarize_candidates_for_trace(final_results)
        self.last_trace = trace
        logger.info(
            "Retrieval complete | route=dense_fast_path | final=%d | total=%.1fms",
            len(final_results),
            trace.stage_timings_ms["total"],
        )
        return final_results

    def _retrieve_global(self, query: str, top_k: int) -> Optional[List[Dict[str, Any]]]:
        """Searches RAPTOR summary nodes for 'global'/summary-style queries.

        Prefers the highest-level summary nodes available (the most
        abstractive, closest to a whole-document view), and only descends to
        lower RAPTOR levels if the top level doesn't return enough results.
        Returns None (rather than []) when no summary nodes exist at all, so
        the caller can fall back to the standard raw-chunk pipeline instead
        of returning an empty answer.
        """
        t0 = _now_ms()
        bm25_results, dense_results = self._search_pair(
            query, bm25_k=max(top_k * 3, 10), dense_k=max(top_k * 3, 10), node_types=["summary"],
        )

        if not bm25_results and not dense_results:
            return None  # No RAPTOR tree for this document.

        fused = self._weighted_rrf(
            ranked_lists=[(bm25_results, 1.0), (dense_results, 1.0)],
            k=self.config.rrf_k,
            score_threshold=0.0,  # summary node pools are small; don't filter further
        )

        # Prefer the highest raptor_level present (most abstractive), then
        # backfill with lower levels if we don't have enough.
        def _level(doc):
            try:
                return int(doc.get("metadata", {}).get("raptor_level", 0))
            except (TypeError, ValueError):
                return 0

        fused_by_level_desc = sorted(fused, key=lambda d: (-_level(d), -d.get("score", 0.0)))

        trace = RetrievalTrace(
            query=query, route="raptor_global", variants=[query],
            candidate_counts={"summary_bm25": len(bm25_results), "summary_dense": len(dense_results), "fused": len(fused)},
            notes=["complexity=global: searched RAPTOR summary nodes only, preferring higher abstraction levels."],
        )
        trace.bm25_candidates = _summarize_candidates_for_trace(bm25_results)
        trace.embedding_candidates = _summarize_candidates_for_trace(dense_results)
        trace.after_rrf = _summarize_candidates_for_trace(fused_by_level_desc)

        final_results = []
        for doc in fused_by_level_desc[:top_k]:
            doc = dict(doc)
            doc["context_text"] = doc.get("text", "")
            doc["retrieval_route"] = "raptor_global"
            doc["retrieval_variants"] = [query]
            final_results.append(doc)

        trace.stage_timings_ms["total"] = _now_ms() - t0
        trace.candidate_counts["final"] = len(final_results)
        trace.submitted_to_llm = _summarize_candidates_for_trace(final_results)
        self.last_trace = trace
        logger.info("Retrieval complete | route=raptor_global | final=%d | total=%.1fms",
                    len(final_results), trace.stage_timings_ms["total"])
        return final_results

    def _retrieve_standard(
        self,
        query: str,
        top_k: int,
        complexity: Optional[str] = None,
        use_summary_nodes: bool = False,
    ) -> List[Dict[str, Any]]:
        """The original full RAG-Fusion pipeline: multi-query variants,
        hybrid BM25+dense, weighted RRF, cross-encoder rerank, diversify.
        Used for complexity="multi_hop" and as the default when no
        complexity argument is given at all."""
        t0 = _now_ms()
        route = self.route_for(query)
        features = self._analyze_query(query)
        variants = self._generate_query_variants(query, features)
        include_summary_nodes = self._should_include_summary_nodes(
            route=route,
            features=features,
            complexity=complexity,
            use_summary_nodes=use_summary_nodes,
        )

        trace = RetrievalTrace(
            query=query,
            route=route,
            features=features.__dict__.copy(),
            variants=variants[:],
        )

        logger.info(f"Retrieval route selected: {route}")
        logger.info(f"Query variants: {variants}")

        bm25_primary_k, dense_primary_k = self._route_budgets(route, top_k)
        weights = self._query_weights(route, features)

        t1 = _now_ms()
        ranked_lists: List[Tuple[List[Dict[str, Any]], float]] = []
        candidate_pool: List[Dict[str, Any]] = []
        bm25_pool: List[Dict[str, Any]] = []
        dense_pool: List[Dict[str, Any]] = []
        summary_ranked_lists: List[Tuple[List[Dict[str, Any]], float]] = []
        summary_pool: List[Dict[str, Any]] = []

        # Retrieve using multiple variants and both retrievers in parallel.
        # Raw chunks are always part of the main pool. Summary nodes are only
        # injected when the caller explicitly opts in (e.g. doc-RAG / RAPTOR
        # routes), which keeps the plain hybrid baseline intact while still
        # letting the tree survive reranking for synthesis questions.
        for idx, variant in enumerate(variants):
            variant_weight = 1.0
            if idx == 0:
                variant_weight = 1.0
            elif idx == 1:
                variant_weight = 0.92
            else:
                variant_weight = 0.85

            bm25_results, dense_results = self._search_pair(
                variant,
                bm25_k=bm25_primary_k if route != "dense" else bm25_primary_k // 2,
                dense_k=dense_primary_k if route != "bm25" else dense_primary_k // 2,
            )
            bm25_results = [d for d in bm25_results if d.get("metadata", {}).get("node_type") != "summary"]
            dense_results = [d for d in dense_results if d.get("metadata", {}).get("node_type") != "summary"]

            trace.candidate_counts[f"bm25_variant_{idx}"] = len(bm25_results)
            trace.candidate_counts[f"dense_variant_{idx}"] = len(dense_results)

            if bm25_results:
                ranked_lists.append((bm25_results, weights["bm25"] * variant_weight))
            if dense_results:
                ranked_lists.append((dense_results, weights["dense"] * variant_weight))

            candidate_pool.extend(bm25_results)
            candidate_pool.extend(dense_results)
            bm25_pool.extend(bm25_results)
            dense_pool.extend(dense_results)

            if include_summary_nodes:
                summary_bm25, summary_dense = self._search_pair(
                    variant,
                    bm25_k=max(bm25_primary_k // 2, 6),
                    dense_k=max(dense_primary_k // 2, 6),
                    node_types=["summary"],
                )
                trace.candidate_counts[f"summary_bm25_variant_{idx}"] = len(summary_bm25)
                trace.candidate_counts[f"summary_dense_variant_{idx}"] = len(summary_dense)

                if summary_bm25:
                    summary_ranked_lists.append((summary_bm25, weights["dense"] * self.config.summary_fusion_weight * variant_weight))
                if summary_dense:
                    summary_ranked_lists.append((summary_dense, weights["dense"] * self.config.summary_fusion_weight * variant_weight))

                summary_pool.extend(summary_bm25)
                summary_pool.extend(summary_dense)

        trace.stage_timings_ms["first_stage_retrieval"] = _now_ms() - t1
        trace.bm25_candidates = _summarize_candidates_for_trace(bm25_pool)
        trace.embedding_candidates = _summarize_candidates_for_trace(dense_pool)

        unique_pool: Dict[str, Dict[str, Any]] = {}
        for doc in candidate_pool:
            unique_pool.setdefault(doc["chunk_id"], doc)

        if not ranked_lists:
            trace.stage_timings_ms["total"] = _now_ms() - t0
            trace.notes.append("No first-stage candidates found.")
            self.last_trace = trace
            return []

        t2 = _now_ms()
        fused = self._weighted_rrf(
            ranked_lists=ranked_lists,
            k=self.config.rrf_k,
            score_threshold=self.config.rrf_threshold,
        )
        trace.stage_timings_ms["fusion"] = _now_ms() - t2
        trace.candidate_counts["fused"] = len(fused)
        trace.after_rrf = _summarize_candidates_for_trace(fused)

        ordered_candidates = fused if fused else list(unique_pool.values())
        ordered_candidates = ordered_candidates[: self.config.fused_candidate_cap]

        if include_summary_nodes and summary_pool:
            summary_fused = self._weighted_rrf(
                ranked_lists=summary_ranked_lists,
                k=self.config.rrf_k,
                score_threshold=0.0,
            ) if summary_ranked_lists else []
            summary_fused = self._sort_summary_candidates(summary_fused)
            summary_reserve = min(
                max(self.config.summary_reserve_k, max(top_k // 2, 1)),
                len(summary_fused),
            )
            if summary_reserve > 0:
                ordered_candidates = self._merge_candidate_lists(
                    summary_fused[:summary_reserve],
                    ordered_candidates,
                )[: self.config.fused_candidate_cap]
                trace.candidate_counts["summary_fused"] = len(summary_fused)
                trace.notes.append(
                    f"summary_nodes_enabled=True: reserved {summary_reserve} RAPTOR summaries for rerank consideration."
                )

        t3 = _now_ms()
        if features.has_negation:
            trace.notes.append(
                "Negation detected in query; kept cross-encoder reranking but will rely on polarity-aware answer generation."
            )
        reranked = self._rerank(query, ordered_candidates, top_k=max(top_k, self.config.rerank_pool_size))
        trace.stage_timings_ms["rerank"] = _now_ms() - t3
        trace.candidate_counts["reranked_input"] = len(ordered_candidates)
        trace.candidate_counts["reranked_output"] = len(reranked)
        trace.after_rerank = _summarize_candidates_for_trace(reranked)

        diversified = self._diversify(reranked, top_k=max(top_k, 1))
        trace.stage_timings_ms["diversify"] = _now_ms() - t3 - trace.stage_timings_ms["rerank"]

        final_results: List[Dict[str, Any]] = []
        for doc in diversified[:top_k]:
            doc = self._expand_parent_context(doc)
            if "rerank_score" in doc:
                doc["score"] = float(doc["rerank_score"])
            doc["retrieval_route"] = route
            doc["retrieval_variants"] = variants
            final_results.append(doc)

        trace.stage_timings_ms["total"] = _now_ms() - t0
        trace.candidate_counts["final"] = len(final_results)
        trace.submitted_to_llm = _summarize_candidates_for_trace(final_results)
        self.last_trace = trace

        logger.info(
            "Retrieval complete | route=%s | candidates=%d | final=%d | total=%.1fms",
            route,
            trace.candidate_counts.get("fused", 0),
            len(final_results),
            trace.stage_timings_ms["total"],
        )

        return final_results


# Backward-compatible alias for callers that expect the old name.
HybridRetrieverV2 = HybridRetriever

# ---------------------------------------------------------------------
# Hierarchical routed hybrid overrides
# ---------------------------------------------------------------------

def _hr_chunk_kind(doc: Dict[str, Any]) -> str:
    meta = doc.get("metadata", {}) or {}
    raw_kind = (
        meta.get("node_type")
        or meta.get("chunk_type")
        or doc.get("node_type")
        or doc.get("chunk_type")
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


def _hr_search_pair(
    self,
    query: str,
    bm25_k: int,
    dense_k: int,
    node_types: List[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run BM25 and dense retrieval in parallel.

    We keep node filtering in Python rather than pushing it down to the stores
    for the hierarchical raw/parent paths. That keeps older chunks that do not
    yet have a node_type field compatible with the new router.

    FIX (visibility): a failed arm used to be indistinguishable from an arm
    that legitimately found nothing -- both produced []. We now record which
    arm(s) raised an exception on `self._last_search_pair_failures` (a set,
    reset each call) so callers (retrieve()/the eval wrappers/the pipeline)
    can propagate "degraded retrieval" into the trace and the user-facing
    confidence field instead of silently returning a plausible-looking but
    incomplete result set.
    """
    search_multiplier = 3 if node_types else 1
    bm25_fetch_k = max(bm25_k, 8) * search_multiplier
    dense_fetch_k = max(dense_k, 8) * search_multiplier

    failures: set = set()

    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {
            ex.submit(self.bm25_store.search, query, bm25_fetch_k): "bm25",
            ex.submit(self.vector_store.search, query, dense_fetch_k): "dense",
        }
        bm25_results: List[Dict[str, Any]] = []
        dense_results: List[Dict[str, Any]] = []
        for fut in as_completed(futures):
            label = futures[fut]
            try:
                results = fut.result() or []
                if node_types:
                    allowed = {str(n).lower() for n in node_types}
                    results = [
                        doc for doc in results
                        if _hr_chunk_kind(doc) in allowed
                    ]
                if label == "bm25":
                    bm25_results = results[:bm25_k]
                else:
                    dense_results = results[:dense_k]
            except Exception as e:
                logger.error(f"[RETRIEVAL DEGRADED] {label.upper()} arm raised an exception for query "
                             f"'{query[:80]}': {e}. Continuing with the surviving arm only.")
                failures.add(label)
                if label == "bm25":
                    bm25_results = []
                else:
                    dense_results = []

    # Accumulate across the whole retrieve() call (multiple variants/passes
    # may invoke _search_pair several times); the wrappers reset this set at
    # the start of each top-level retrieve() call.
    existing = getattr(self, "_last_search_pair_failures", None)
    self._last_search_pair_failures = (existing or set()) | failures

    return bm25_results, dense_results



def _hr_expand_parent_context(self, doc: Dict[str, Any], prefer_parent: bool = True) -> Dict[str, Any]:
    meta = doc.get("metadata", {}) or {}
    kind = _hr_chunk_kind(doc)
    parent_text = None

    if kind in {"summary", "parent"}:
        parent_text = doc.get("text", "")
    elif prefer_parent:
        parent_text = meta.get("parent_text") or doc.get("parent_text")
        if not parent_text:
            parent_id = meta.get("parent_id")
            if parent_id and hasattr(self.vector_store, "parent_cache"):
                parent_text = self.vector_store.parent_cache.get(parent_id)

    expanded = dict(doc)
    expanded["chunk_kind"] = kind
    if parent_text:
        expanded["context_text"] = parent_text
    else:
        expanded["context_text"] = doc.get("text", "")
    return expanded


def _hr_retrieve_simple(self, query: str, top_k: int) -> List[Dict[str, Any]]:
    """Fast path for atomic factual lookups.

    Only raw chunks are kept unless the corpus has not yet been re-ingested with
    hierarchical node_type metadata.
    """
    t0 = _now_ms()
    trace = RetrievalTrace(query=query, route="dense_fast_path", variants=[query])

    bm25_results, dense_results = self._search_pair(
        query,
        bm25_k=max(top_k * 3, 12),
        dense_k=max(top_k * 3, 12),
    )

    raw_bm25 = [d for d in bm25_results if _hr_chunk_kind(d) == "raw"]
    raw_dense = [d for d in dense_results if _hr_chunk_kind(d) == "raw"]

    # Backward compatibility: if the corpus has not been re-chunked yet, fall
    # back to the full candidate lists instead of returning nothing.
    bm25_pool = raw_bm25 or bm25_results
    dense_pool = raw_dense or dense_results

    trace.candidate_counts["bm25"] = len(bm25_pool)
    trace.candidate_counts["dense"] = len(dense_pool)
    trace.bm25_candidates = _summarize_candidates_for_trace(bm25_pool)
    trace.embedding_candidates = _summarize_candidates_for_trace(dense_pool)
    trace.notes.append("simple route: raw chunks only, no multi-query expansion.")

    fused = self._weighted_rrf(
        ranked_lists=[(bm25_pool, 1.0), (dense_pool, 1.0)],
        k=self.config.rrf_k,
        score_threshold=min(self.config.rrf_threshold, 0.0),
    )
    ordered = fused if fused else (dense_pool or bm25_pool)
    trace.after_rrf = _summarize_candidates_for_trace(ordered)

    final_results = []
    for doc in ordered[:top_k]:
        doc = self._expand_parent_context(doc, prefer_parent=False)
        doc["retrieval_route"] = "dense_fast_path"
        doc["retrieval_variants"] = [query]
        final_results.append(doc)

    trace.stage_timings_ms["total"] = _now_ms() - t0
    trace.candidate_counts["final"] = len(final_results)
    trace.submitted_to_llm = _summarize_candidates_for_trace(final_results)
    self.last_trace = trace
    logger.info(
        "Retrieval complete | route=dense_fast_path | final=%d | total=%.1fms",
        len(final_results),
        trace.stage_timings_ms["total"],
    )
    return final_results


def _hr_retrieve_global(self, query: str, top_k: int) -> Optional[List[Dict[str, Any]]]:
    """Prefer the highest-level available hierarchy for global questions."""
    t0 = _now_ms()

    summary_bm25, summary_dense = self._search_pair(
        query,
        bm25_k=max(top_k * 4, 12),
        dense_k=max(top_k * 4, 12),
        node_types=["summary"],
    )

    if not summary_bm25 and not summary_dense:
        # New hierarchical path fallback: section-level parent nodes.
        summary_bm25, summary_dense = self._search_pair(
            query,
            bm25_k=max(top_k * 4, 12),
            dense_k=max(top_k * 4, 12),
            node_types=["parent"],
        )

    if not summary_bm25 and not summary_dense:
        return None

    fused = self._weighted_rrf(
        ranked_lists=[(summary_bm25, 1.0), (summary_dense, 1.0)],
        k=self.config.rrf_k,
        score_threshold=0.0,
    )

    def _level(doc):
        meta = doc.get("metadata", {}) or {}
        try:
            return int(meta.get("raptor_level", meta.get("hierarchy_level", 1)))
        except (TypeError, ValueError):
            return 0

    fused_by_level_desc = sorted(fused, key=lambda d: (-_level(d), -d.get("score", 0.0)))

    trace = RetrievalTrace(
        query=query,
        route="hierarchical_global",
        variants=[query],
        candidate_counts={
            "summary_bm25": len(summary_bm25),
            "summary_dense": len(summary_dense),
            "fused": len(fused),
        },
        notes=["global route: searched summary nodes first, then parent-level fallbacks."],
    )
    trace.bm25_candidates = _summarize_candidates_for_trace(summary_bm25)
    trace.embedding_candidates = _summarize_candidates_for_trace(summary_dense)
    trace.after_rrf = _summarize_candidates_for_trace(fused_by_level_desc)

    final_results: List[Dict[str, Any]] = []
    for doc in fused_by_level_desc[:top_k]:
        doc = dict(doc)
        doc["context_text"] = doc.get("text", "")
        doc["retrieval_route"] = "hierarchical_global"
        doc["retrieval_variants"] = [query]
        final_results.append(doc)

    trace.stage_timings_ms["total"] = _now_ms() - t0
    trace.candidate_counts["final"] = len(final_results)
    trace.submitted_to_llm = _summarize_candidates_for_trace(final_results)
    self.last_trace = trace
    logger.info(
        "Retrieval complete | route=hierarchical_global | final=%d | total=%.1fms",
        len(final_results),
        trace.stage_timings_ms["total"],
    )
    return final_results


def _hr_retrieve_standard(
    self,
    query: str,
    top_k: int,
    complexity: Optional[str] = None,
    use_summary_nodes: bool = False,
) -> List[Dict[str, Any]]:
    """Routed hybrid retrieval over raw + section-level chunks."""
    t0 = _now_ms()
    # Reset the per-call failure tracker (see _hr_search_pair) so this
    # retrieve() call's trace only reflects failures from this call, not a
    # stale set left over from a previous query on the same retriever
    # instance (the eval harness reuses one retriever per document).
    self._last_search_pair_failures = set()
    self._last_rerank_failed = False
    route = self.route_for(query)
    features = self._analyze_query(query)
    variants = self._generate_query_variants(query, features)
    include_summary_nodes = self._should_include_summary_nodes(
        route=route,
        features=features,
        complexity=complexity,
        use_summary_nodes=use_summary_nodes,
    )

    include_parent_nodes = (
        complexity in {"multi_hop", "global"}
        or features.is_comparative
        or features.has_negation
        or features.word_count >= 12
    )

    trace = RetrievalTrace(
        query=query,
        route=route,
        features=features.__dict__.copy(),
        variants=variants[:],
    )

    logger.info(f"Retrieval route selected: {route}")
    logger.info(f"Query variants: {variants}")
    logger.info(f"include_parent_nodes={include_parent_nodes}, include_summary_nodes={include_summary_nodes}")

    bm25_primary_k, dense_primary_k = self._route_budgets(route, top_k)
    weights = self._query_weights(route, features)

    t1 = _now_ms()
    raw_ranked_lists: List[Tuple[List[Dict[str, Any]], float]] = []
    parent_ranked_lists: List[Tuple[List[Dict[str, Any]], float]] = []
    summary_ranked_lists: List[Tuple[List[Dict[str, Any]], float]] = []
    candidate_pool: List[Dict[str, Any]] = []
    summary_pool: List[Dict[str, Any]] = []
    bm25_pool: List[Dict[str, Any]] = []
    dense_pool: List[Dict[str, Any]] = []

    for idx, variant in enumerate(variants):
        variant_weight = 1.0 if idx == 0 else (0.92 if idx == 1 else 0.85)

        bm25_results, dense_results = self._search_pair(
            variant,
            bm25_k=bm25_primary_k if route != "dense" else max(bm25_primary_k // 2, 6),
            dense_k=dense_primary_k if route != "bm25" else max(dense_primary_k // 2, 6),
        )
        bm25_pool.extend(bm25_results)
        dense_pool.extend(dense_results)

        raw_bm25 = [d for d in bm25_results if _hr_chunk_kind(d) == "raw"]
        raw_dense = [d for d in dense_results if _hr_chunk_kind(d) == "raw"]
        parent_bm25 = [d for d in bm25_results if _hr_chunk_kind(d) == "parent"]
        parent_dense = [d for d in dense_results if _hr_chunk_kind(d) == "parent"]

        # Backward compatibility: if old chunks are missing node metadata, keep
        # them in the raw pool instead of discarding them.
        if not raw_bm25 and bm25_results:
            raw_bm25 = bm25_results
        if not raw_dense and dense_results:
            raw_dense = dense_results

        trace.candidate_counts[f"bm25_variant_{idx}"] = len(raw_bm25)
        trace.candidate_counts[f"dense_variant_{idx}"] = len(raw_dense)
        if include_parent_nodes:
            trace.candidate_counts[f"parent_bm25_variant_{idx}"] = len(parent_bm25)
            trace.candidate_counts[f"parent_dense_variant_{idx}"] = len(parent_dense)

        if raw_bm25:
            raw_ranked_lists.append((raw_bm25, weights["bm25"] * variant_weight))
        if raw_dense:
            raw_ranked_lists.append((raw_dense, weights["dense"] * variant_weight))

        candidate_pool.extend(raw_bm25)
        candidate_pool.extend(raw_dense)

        if include_parent_nodes:
            parent_weight = 0.72
            if parent_bm25:
                parent_ranked_lists.append((parent_bm25, weights["bm25"] * parent_weight * variant_weight))
            if parent_dense:
                parent_ranked_lists.append((parent_dense, weights["dense"] * parent_weight * variant_weight))
            candidate_pool.extend(parent_bm25)
            candidate_pool.extend(parent_dense)

        if include_summary_nodes:
            summary_bm25, summary_dense = self._search_pair(
                variant,
                bm25_k=max(bm25_primary_k // 2, 6),
                dense_k=max(dense_primary_k // 2, 6),
                node_types=["summary"],
            )
            trace.candidate_counts[f"summary_bm25_variant_{idx}"] = len(summary_bm25)
            trace.candidate_counts[f"summary_dense_variant_{idx}"] = len(summary_dense)

            if summary_bm25:
                summary_ranked_lists.append((summary_bm25, weights["dense"] * self.config.summary_fusion_weight * variant_weight))
            if summary_dense:
                summary_ranked_lists.append((summary_dense, weights["dense"] * self.config.summary_fusion_weight * variant_weight))

            summary_pool.extend(summary_bm25)
            summary_pool.extend(summary_dense)

    trace.stage_timings_ms["first_stage_retrieval"] = _now_ms() - t1
    trace.bm25_candidates = _summarize_candidates_for_trace(bm25_pool)
    trace.embedding_candidates = _summarize_candidates_for_trace(dense_pool)

    unique_pool: Dict[str, Dict[str, Any]] = {}
    for doc in candidate_pool:
        unique_pool.setdefault(doc["chunk_id"], doc)

    if not raw_ranked_lists and not parent_ranked_lists and not summary_ranked_lists:
        trace.stage_timings_ms["total"] = _now_ms() - t0
        trace.notes.append("No first-stage candidates found.")
        self.last_trace = trace
        return []

    t2 = _now_ms()
    fused_raw = self._weighted_rrf(
        ranked_lists=raw_ranked_lists,
        k=self.config.rrf_k,
        score_threshold=self.config.rrf_threshold,
    ) if raw_ranked_lists else []
    fused_parent = self._weighted_rrf(
        ranked_lists=parent_ranked_lists,
        k=self.config.rrf_k,
        score_threshold=max(0.0, self.config.rrf_threshold * 0.7),
    ) if parent_ranked_lists else []

    trace.stage_timings_ms["fusion"] = _now_ms() - t2
    trace.candidate_counts["fused_raw"] = len(fused_raw)
    trace.candidate_counts["fused_parent"] = len(fused_parent)

    ordered_candidates = self._merge_candidate_lists(fused_raw, fused_parent)
    if not ordered_candidates:
        ordered_candidates = list(unique_pool.values())
    ordered_candidates = ordered_candidates[: self.config.fused_candidate_cap]
    trace.after_rrf = _summarize_candidates_for_trace(ordered_candidates)

    if include_summary_nodes and summary_pool:
        summary_fused = self._weighted_rrf(
            ranked_lists=summary_ranked_lists,
            k=self.config.rrf_k,
            score_threshold=0.0,
        ) if summary_ranked_lists else []
        summary_fused = self._sort_summary_candidates(summary_fused)
        summary_reserve = min(
            max(self.config.summary_reserve_k, max(top_k // 2, 1)),
            len(summary_fused),
        )
        if summary_reserve > 0:
            ordered_candidates = self._merge_candidate_lists(
                summary_fused[:summary_reserve],
                ordered_candidates,
            )[: self.config.fused_candidate_cap]
            trace.candidate_counts["summary_fused"] = len(summary_fused)
            trace.notes.append(
                f"summary_nodes_enabled=True: reserved {summary_reserve} summary nodes for rerank consideration."
            )

    t3 = _now_ms()
    if features.has_negation:
        trace.notes.append(
            "Negation detected in query; reranker kept, answer generation must preserve polarity."
        )
    reranked = self._rerank(query, ordered_candidates, top_k=max(top_k, self.config.rerank_pool_size))
    trace.stage_timings_ms["rerank"] = _now_ms() - t3
    trace.candidate_counts["reranked_input"] = len(ordered_candidates)
    trace.candidate_counts["reranked_output"] = len(reranked)
    trace.after_rerank = _summarize_candidates_for_trace(reranked)

    diversified = self._diversify(reranked, top_k=max(top_k, 1))
    trace.stage_timings_ms["diversify"] = _now_ms() - t3 - trace.stage_timings_ms["rerank"]

    final_results: List[Dict[str, Any]] = []
    prefer_parent_context = include_parent_nodes or include_summary_nodes
    for doc in diversified[:top_k]:
        doc = self._expand_parent_context(doc, prefer_parent=prefer_parent_context)
        if "rerank_score" in doc:
            doc["score"] = float(doc["rerank_score"])
        doc["retrieval_route"] = route
        doc["retrieval_variants"] = variants
        final_results.append(doc)

    trace.stage_timings_ms["total"] = _now_ms() - t0
    trace.candidate_counts["final"] = len(final_results)
    trace.submitted_to_llm = _summarize_candidates_for_trace(final_results)

    failed_arms = getattr(self, "_last_search_pair_failures", None) or set()
    if failed_arms:
        trace.notes.append(
            f"DEGRADED RETRIEVAL: the following arm(s) raised exceptions and returned no "
            f"results for at least one query variant: {sorted(failed_arms)}. Results below "
            f"reflect only the surviving arm(s) and should not be treated as a normal "
            f"empty-result case."
        )
        trace.candidate_counts["degraded_arms"] = len(failed_arms)

    if getattr(self, "_last_rerank_failed", False):
        trace.notes.append(
            "DEGRADED RETRIEVAL: cross-encoder reranking failed; results below are in "
            "fused (RRF) order, not reranked order."
        )
        trace.candidate_counts["rerank_failed"] = 1

    self.last_trace = trace

    logger.info(
        "Retrieval complete | route=%s | candidates=%d | final=%d | total=%.1fms",
        route,
        trace.candidate_counts.get("fused_raw", 0) + trace.candidate_counts.get("fused_parent", 0),
        len(final_results),
        trace.stage_timings_ms["total"],
    )

    return final_results


HybridRetriever._search_pair = _hr_search_pair
HybridRetriever._expand_parent_context = _hr_expand_parent_context
HybridRetriever._retrieve_simple = _hr_retrieve_simple
HybridRetriever._retrieve_global = _hr_retrieve_global
HybridRetriever._retrieve_standard = _hr_retrieve_standard
