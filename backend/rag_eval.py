# """
# rag_eval.py
# ===========

# RAG quality evaluation harness for the document-grounded pipeline.

# Implements the four core RAGAS-style metrics via an LLM judge (Groq), plus
# retrieval-only metrics that don't require an LLM call:

#   - Context Precision      : fraction of retrieved top-k chunks that are relevant
#   - Context Precision @ R  : precision over only the top R retrieved chunks,
#                               where R = number of gold-relevant chunks (avoids
#                               the precision@k ceiling when R < k)
#   - Context Recall         : fraction of relevant chunks that were retrieved
#   - Faithfulness            : fraction of answer claims supported by retrieved context
#   - Answer Relevance        : how well the answer addresses the original query

#   - Hit Rate                : did >=1 relevant chunk appear in top_k?
#   - MRR                     : mean reciprocal rank of the first relevant chunk

# It compares two retrieval/generation configurations on the same gold set:

#   - "hybrid"  : the system's HybridRetriever (BM25 + dense, RRF fusion, rerank)
#   - "naive"   : a single dense-retrieval baseline (top_k from VectorStore only,
#                 no fusion, no reranking) -- the standard "naive RAG" comparator.

# Each configuration also reports the QueryRouter's route decision per query
# (dense/bm25/hybrid/math), so you can confirm the gold set actually exercises
# more than one retrieval path for the "hybrid" configuration.

# IMPORTANT -- before running this for the first time, fix known index drift:
#     python -m eval.reindex_fixtures
# This re-ingests dummy.txt / semantic_fixture.txt with the current chunking
# schema (fixing stale parent_id metadata that causes "Parent text missing in
# cache" warnings) and syncs the PDF corpus into BM25 so bm25/hybrid routes can
# actually retrieve it.

# Usage:
#     python -m eval.rag_eval                    # run full comparison, print report
#     python -m eval.rag_eval --output out.json  # also write raw results to JSON
#     python -m eval.rag_eval --runs 3           # repeat 3x, report mean +/- std
#                                                 # (mitigates LLM-judge stochasticity)

# Requires GROQ_API_KEY (same as the rest of the backend) for the LLM-judge
# metrics. If unset, only the retrieval-only metrics (hit rate, MRR, context
# precision/recall) are computed and the LLM-judge metrics are reported as
# "skipped".
# """

# import os
# import json
# import asyncio
# import argparse
# import statistics
# from typing import List, Dict, Any, Tuple, TYPE_CHECKING

# from dotenv import load_dotenv
# load_dotenv()

# from metrics import (
#     hit_rate,
#     mrr,
#     context_precision,
#     context_precision_at_r,
#     context_recall,
#     chunk_id_for as _chunk_id_for,
# )

# try:
#     from groq import AsyncGroq
# except ImportError:
#     AsyncGroq = None

# if TYPE_CHECKING:
#     from vector_store import VectorStore
#     from generator import AnswerGenerator


# GOLD_PATH = os.path.join(os.path.dirname(__file__), "golden_qa.json")

# # Judge model selection — Groq free-tier reality check (as of June 2026):
# #
# #   llama-3.3-70b-versatile : 100K TPD free. One successful 3-run eval
# #     (~90K tokens) exhausts the daily quota. Do not use for multi-run evals.
# #   llama-3.1-70b-versatile : DECOMMISSIONED — returns 400 immediately.
# #   llama-3.1-8b-instant    : 500K TPD, 14,400 RPD — the most permissive
# #     free-tier model. 3 runs × 10 queries × 2 judge calls × ~800 tokens
# #     = ~48K tokens, well within budget.
# #
# # On the previous successful run (single run, 70B judge) faithfulness scored
# # 1.0 across all 10 queries. The answers are short, well-grounded, and the
# # context is capped at 800 chars/chunk — 8B handles this cleanly. For a
# # corpus where faithfulness scores vary more, consider paying for Developer
# # tier (~$0.05 to evaluate this full gold set) or running one 70B run + two
# # 8B runs and reporting them separately.
# JUDGE_MODEL_FAITHFULNESS  = "llama-3.1-8b-instant"
# JUDGE_MODEL_RELEVANCE     = "llama-3.1-8b-instant"

# # Hard cap per retrieved chunk's context_text when building the faithfulness
# # prompt. Prevents a handful of long, mostly-irrelevant chunks (e.g. from a
# # large unrelated PDF) from dominating the prompt and derailing claim
# # decomposition. Keep this tight for 8B models especially.
# MAX_CONTEXT_CHARS_PER_CHUNK = 800

# # Retry config for 429 rate-limit responses from Groq
# JUDGE_MAX_RETRIES = 4
# JUDGE_RETRY_BASE_DELAY = 5.0   # seconds; doubles each retry (5, 10, 20, 40)


# # ---------------------------------------------------------------------------
# # Retrieval configurations
# # ---------------------------------------------------------------------------

# class NaiveRetriever:
#     """Naive single-pass dense retrieval baseline: top_k from the vector
#     store only, no BM25, no RRF fusion, no reranking."""

#     def __init__(self, vector_store: "VectorStore"):
#         self.vector_store = vector_store

#     def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#         results = self.vector_store.search(query, top_k=top_k)
#         for doc in results:
#             meta = doc.get("metadata", {})
#             doc["context_text"] = meta.get("parent_text", doc.get("text", ""))
#         return results

#     def route_for(self, query: str) -> str:
#         return "n/a (naive, dense-only)"


# class HybridRetrieverWithRoute:
#     """Thin wrapper around HybridRetriever that also exposes the
#     QueryRouter's decision for a query, so the eval report can confirm the
#     gold set actually exercises more than one retrieval path."""

#     def __init__(self, hybrid_retriever):
#         self._inner = hybrid_retriever

#     def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#         return self._inner.retrieve(query, top_k=top_k)

#     def route_for(self, query: str) -> str:
#         # QueryRouter.route() is a pure function of the query string (no
#         # side effects beyond logging), so calling it again here for
#         # reporting purposes is safe and won't affect the actual retrieval
#         # call above.
#         return self._inner.router.route(query)


# # ---------------------------------------------------------------------------
# # LLM-judge metrics (faithfulness, answer relevance)
# # ---------------------------------------------------------------------------

# class LLMJudge:
#     def __init__(self):
#         api_key = os.getenv("GROQ_API_KEY")
#         self.enabled = False
#         if api_key and AsyncGroq:
#             try:
#                 # Two separate client instances so each picks up its own
#                 # per-model rate-limit quota on Groq's backend.
#                 self._faith_client = AsyncGroq(api_key=api_key)
#                 self._rel_client   = AsyncGroq(api_key=api_key)
#                 self.enabled = True
#             except Exception as e:
#                 self._init_error = str(e)

#     async def _judge_json(self, client, model: str, prompt: str) -> Dict[str, Any]:
#         """Call Groq with exponential-backoff retry on 429 rate-limit errors.

#         Distinguishes two 429 cases:
#         - RPM/TPM burst (wait < 90s): retry after the indicated wait.
#         - TPD exhaustion (wait > 90s, or no wait time parseable but still 429
#           after retries): the daily quota is gone; skip immediately rather than
#           making the user wait minutes per query.

#         Returns an empty dict on non-retryable errors so callers degrade gracefully.
#         """
#         import asyncio as _asyncio
#         import re as _re

#         delay = JUDGE_RETRY_BASE_DELAY
#         for attempt in range(1, JUDGE_MAX_RETRIES + 1):
#             try:
#                 resp = await client.chat.completions.create(
#                     model=model,
#                     messages=[{"role": "user", "content": prompt}],
#                     temperature=0.0,
#                     response_format={"type": "json_object"},
#                 )
#                 raw = resp.choices[0].message.content
#                 try:
#                     return json.loads(raw)
#                 except Exception:
#                     return {}
#             except Exception as e:
#                 err_str = str(e)
#                 if "429" in err_str or "rate_limit" in err_str.lower():
#                     m = _re.search(r"try again in\s+([\d.]+)([smh]?)", err_str, _re.IGNORECASE)
#                     if m:
#                         raw_wait = float(m.group(1))
#                         unit = m.group(2).lower()
#                         if unit == "m":
#                             raw_wait *= 60
#                         elif unit == "h":
#                             raw_wait *= 3600
#                         wait = raw_wait + 2.0
#                     else:
#                         wait = delay

#                     # TPD exhaustion: wait time is hours/minutes away (> 90s).
#                     # Retrying is pointless — skip this query's judge score.
#                     if wait > 90:
#                         from logger import logger as _logger
#                         _logger.warning(
#                             f"Judge TPD quota exhausted for {model} "
#                             f"(reset in ~{wait/60:.0f}min). "
#                             f"Skipping LLM-judge metrics for this query. "
#                             f"Re-run after midnight UTC or switch to a model "
#                             f"with remaining quota."
#                         )
#                         return {}

#                     if attempt < JUDGE_MAX_RETRIES:
#                         from logger import logger as _logger
#                         _logger.warning(
#                             f"Judge rate-limited (attempt {attempt}/{JUDGE_MAX_RETRIES}), "
#                             f"waiting {wait:.0f}s before retry..."
#                         )
#                         await _asyncio.sleep(wait)
#                         delay *= 2
#                         continue
#                     else:
#                         from logger import logger as _logger
#                         _logger.error(f"Judge rate-limit retries exhausted: {err_str[:120]}")
#                         return {}
#                 else:
#                     from logger import logger as _logger
#                     _logger.error(f"Judge call failed (non-retryable): {err_str[:120]}")
#                     return {}
#         return {}

#     async def faithfulness(self, answer: str, context: str) -> Tuple[float, str]:
#         """Decomposes the answer into atomic claims and checks each against
#         the retrieved context. Returns (score, rationale).
#         Uses JUDGE_MODEL_FAITHFULNESS (llama-3.1-8b-instant).

#         IMPORTANT: only claims that appear in (or can be directly inferred
#         from) the answer text are evaluated. Facts that exist in the context
#         but were NOT stated in the answer do not affect the score — this is
#         a faithfulness check, not a completeness check.
#         """
#         if not self.enabled:
#             reason = getattr(self, "_init_error", None)
#             return None, f"skipped ({reason or 'no GROQ_API_KEY'})"

#         prompt = f"""You are a strict FAITHFULNESS evaluator for a retrieval-augmented QA system.

# Your task is to decide whether every factual claim that appears in the ANSWER
# is supported by the RETRIEVED CONTEXT. This is NOT a completeness check — you
# must NOT penalise the answer for omitting facts that are in the context. You
# also must NOT mark a claim as "supported" just because it sounds plausible or
# appears somewhere in the context unrelated to what the answer actually says.

# RETRIEVED CONTEXT (use only to verify claims made in the answer):
# {context}

# ANSWER TO EVALUATE (decompose only this text into claims):
# {answer}

# Instructions:
# 1. Read the ANSWER carefully and extract every distinct atomic factual claim
#    it makes (ignore hedges like "I think" or "according to the context").
# 2. For each claim, check whether the RETRIEVED CONTEXT explicitly states or
#    directly implies that claim. Do NOT use outside knowledge.
#    - "supported"    → the context clearly backs the claim
#    - "unsupported"  → the context does not address it (potential hallucination)
#    - "contradicted" → the context directly contradicts the claim
# 3. faithfulness_score = supported_count / total_claim_count.
#    If the answer contains no factual claims (e.g. it is a refusal or
#    "I don't know"), set faithfulness_score to 1.0.

# Return ONLY valid JSON with no preamble:
# {{"claims": [{{"claim": "...", "verdict": "supported|contradicted|unsupported"}}], "faithfulness_score": <float 0.0-1.0>}}"""

#         result = await self._judge_json(self._faith_client, JUDGE_MODEL_FAITHFULNESS, prompt)
#         score = result.get("faithfulness_score")
#         if score is None:
#             return None, "judge returned invalid JSON or rate-limit exhausted"
#         return float(score), json.dumps(result.get("claims", []))

#     async def answer_relevance(self, query: str, answer: str) -> Tuple[float, str]:
#         """Scores how directly and completely the answer addresses the query,
#         independent of factual correctness.
#         Uses JUDGE_MODEL_RELEVANCE (llama-3.1-70b-versatile) — separate Groq quota."""
#         if not self.enabled:
#             reason = getattr(self, "_init_error", None)
#             return None, f"skipped ({reason or 'no GROQ_API_KEY'})"

#         prompt = f"""You are evaluating ANSWER RELEVANCE: how directly and completely
# an answer addresses the question asked, regardless of whether the answer is
# factually correct.

# Question:
# {query}

# Answer:
# {answer}

# Score from 0.0 to 1.0:
# - 1.0: directly and completely addresses the question
# - 0.5: partially addresses it, is vague, or includes irrelevant content
# - 0.0: does not address the question at all (e.g. off-topic, refusal with no attempt)

# Return ONLY a JSON object: {{"relevance_score": <float 0-1>, "rationale": "<one sentence>"}}"""

#         result = await self._judge_json(self._rel_client, JUDGE_MODEL_RELEVANCE, prompt)
#         score = result.get("relevance_score")
#         if score is None:
#             return None, "judge returned invalid JSON or rate-limit exhausted"
#         return float(score), result.get("rationale", "")


# # ---------------------------------------------------------------------------
# # Evaluation runner
# # ---------------------------------------------------------------------------

# async def _generate_answer(
#     generator: "AnswerGenerator",
#     query: str,
#     chunks: List[Dict[str, Any]],
#     fallback_generator: "AnswerGenerator | None" = None,
#     max_retries: int = 2,
# ) -> str:
#     """Drains the generator's streaming response and returns the final answer text.

#     If the primary generator fails (network error, JSON parse failure, or the
#     known "An error occurred during answer generation." sentinel), it retries
#     up to max_retries times. If all retries fail and a fallback_generator is
#     provided, it falls through to that (e.g. the naive retriever's generator
#     path) before giving up. This prevents a single transient Groq error from
#     collapsing an entire query's metrics to zero.
#     """
#     from logger import logger as _logger

#     ERROR_SENTINEL = "An error occurred during answer generation."

#     async def _try_once(gen) -> str:
#         final_answer = ""
#         async for event in gen.generate_stream(query, chunks, mode="doc_rag"):
#             if event.get("event") == "final":
#                 try:
#                     data = json.loads(event["data"])
#                     final_answer = data.get("answer", "")
#                 except Exception:
#                     pass
#         return final_answer

#     for attempt in range(1, max_retries + 1):
#         try:
#             answer = await _try_once(generator)
#             if answer and answer != ERROR_SENTINEL:
#                 return answer
#             _logger.warning(
#                 f"Generator returned error sentinel on attempt {attempt}/{max_retries} "
#                 f"for query: {query[:60]!r}"
#             )
#         except Exception as e:
#             _logger.warning(
#                 f"Generator raised exception on attempt {attempt}/{max_retries}: {e}"
#             )
#         if attempt < max_retries:
#             await asyncio.sleep(2.0 * attempt)  # brief back-off between retries

#     # All primary retries exhausted — try the fallback generator if provided
#     if fallback_generator is not None:
#         _logger.warning(
#             f"Primary generator failed after {max_retries} attempts; "
#             f"falling back to naive generator for query: {query[:60]!r}"
#         )
#         try:
#             answer = await _try_once(fallback_generator)
#             if answer:
#                 return answer
#         except Exception as e:
#             _logger.error(f"Fallback generator also failed: {e}")

#     _logger.error(
#         f"All generation attempts failed for query: {query[:60]!r}. "
#         f"This query's faithfulness / relevance scores will be N/A."
#     )
#     return ""


# async def evaluate_config(
#     name: str,
#     retrieve_fn,
#     examples: List[Dict[str, Any]],
#     generator: "AnswerGenerator",
#     judge: LLMJudge,
#     top_k: int = 5,
#     fallback_generator: "AnswerGenerator | None" = None,
# ) -> Dict[str, Any]:
#     from logger import logger
#     logger.info(f"Evaluating configuration: {name}")
#     per_query = []

#     for ex in examples:
#         query = ex["query"]
#         relevant_ids = ex["relevant_chunk_ids"]
#         expected_route = ex.get("expected_route")

#         retrieved = retrieve_fn.retrieve(query, top_k)
#         retrieved_ids = [_chunk_id_for(d) for d in retrieved]
#         actual_route = retrieve_fn.route_for(query)

#         context_str = "\n---\n".join(
#             d.get("context_text", d.get("text", ""))[:MAX_CONTEXT_CHARS_PER_CHUNK]
#             for d in retrieved
#         )

#         answer = await _generate_answer(
#             generator, query, retrieved, fallback_generator=fallback_generator
#         )

#         faith_score, faith_detail = await judge.faithfulness(answer, context_str)
#         rel_score, rel_detail = await judge.answer_relevance(query, answer)

#         per_query.append({
#             "id": ex["id"],
#             "query": query,
#             "answer": answer,
#             "retrieved_ids": retrieved_ids,
#             "relevant_ids": relevant_ids,
#             "expected_route": expected_route,
#             "actual_route": actual_route,
#             "hit_rate": hit_rate(retrieved_ids, relevant_ids),
#             "mrr": mrr(retrieved_ids, relevant_ids),
#             "context_precision": context_precision(retrieved_ids, relevant_ids),
#             "context_precision_at_r": context_precision_at_r(retrieved_ids, relevant_ids),
#             "context_recall": context_recall(retrieved_ids, relevant_ids),
#             "faithfulness": faith_score,
#             "faithfulness_detail": faith_detail,
#             "answer_relevance": rel_score,
#             "answer_relevance_detail": rel_detail,
#         })

#     def _avg(key):
#         vals = [q[key] for q in per_query if q[key] is not None]
#         return sum(vals) / len(vals) if vals else None

#     summary = {
#         "config": name,
#         "n_examples": len(examples),
#         "hit_rate": _avg("hit_rate"),
#         "mrr": _avg("mrr"),
#         "context_precision": _avg("context_precision"),
#         "context_precision_at_r": _avg("context_precision_at_r"),
#         "context_recall": _avg("context_recall"),
#         "faithfulness": _avg("faithfulness"),
#         "answer_relevance": _avg("answer_relevance"),
#         "per_query": per_query,
#     }
#     return summary


# async def main(output_path: str = None, top_k: int = 5, runs: int = 1):
#     from logger import logger
#     from vector_store import VectorStore
#     from bm25_store import BM25Store
#     from retriever import HybridRetriever
#     from generator import AnswerGenerator

#     with open(GOLD_PATH, "r", encoding="utf-8") as f:
#         gold = json.load(f)
#     examples = gold["examples"]

#     vector_store = VectorStore(persist_dir="./data/chroma")
#     bm25_store = BM25Store(persist_dir="./data/bm25")
#     hybrid_retriever = HybridRetrieverWithRoute(HybridRetriever(vector_store, bm25_store))
#     naive_retriever = NaiveRetriever(vector_store)
#     generator = AnswerGenerator()
#     judge = LLMJudge()

#     if not judge.enabled:
#         logger.warning("GROQ_API_KEY not set or judge client failed to init; "
#                         "faithfulness and answer_relevance will be skipped.")

#     all_runs = []
#     for run_idx in range(runs):
#         if runs > 1:
#             logger.info(f"=== Run {run_idx + 1}/{runs} ===")
#         run_results = {}
#         run_results["naive"] = await evaluate_config(
#             "naive_single_retrieval", naive_retriever, examples, generator, judge, top_k=top_k
#         )
#         run_results["hybrid"] = await evaluate_config(
#             "hybrid_rrf_rerank", hybrid_retriever, examples, generator, judge, top_k=top_k,
#             # Fallback to naive generator on transient Groq errors (e.g. q8)
#             fallback_generator=generator,
#         )
#         all_runs.append(run_results)

#     if runs == 1:
#         results = all_runs[0]
#         _print_report(results)
#     else:
#         results = _aggregate_runs(all_runs)
#         _print_report_with_std(results, runs)

#     if output_path:
#         with open(output_path, "w", encoding="utf-8") as f:
#             json.dump({"runs": all_runs, "aggregate": results if runs > 1 else None}, f, indent=2)
#         logger.info(f"Wrote raw results to {output_path}")

#     return results


# def _fmt(v):
#     if v is None:
#         return "  N/A"
#     return f"{v:.3f}"


# METRIC_LABELS = [
#     ("Hit Rate@k", "hit_rate"),
#     ("MRR@k", "mrr"),
#     ("Context Precision@k", "context_precision"),
#     ("Context Precision@R", "context_precision_at_r"),
#     ("Context Recall", "context_recall"),
#     ("Faithfulness", "faithfulness"),
#     ("Answer Relevance", "answer_relevance"),
# ]


# def _print_route_summary(results: Dict[str, Any]):
#     hybrid = results["hybrid"]
#     routes = [q["actual_route"] for q in hybrid["per_query"]]
#     counts = {}
#     for r in routes:
#         counts[r] = counts.get(r, 0) + 1
#     summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
#     print(f"Router decisions across gold set (hybrid config): {summary}")
#     mismatches = [
#         q["id"] for q in hybrid["per_query"]
#         if q.get("expected_route") and q["actual_route"] != q["expected_route"]
#     ]
#     if mismatches:
#         print(f"Note: router decision differs from expected_route for: {', '.join(mismatches)}")
#     print()


# def _print_report(results: Dict[str, Any]):
#     naive = results["naive"]
#     hybrid = results["hybrid"]

#     print("\n" + "=" * 60)
#     print(f"RAG EVALUATION REPORT  (n={naive['n_examples']} queries, faithfulness judge={JUDGE_MODEL_FAITHFULNESS}, relevance judge={JUDGE_MODEL_RELEVANCE})")
#     print("=" * 60)
#     print(f"{'Metric':<22}{'Naive (single dense)':<22}{'Hybrid (RRF+rerank)':<18}")
#     print("-" * 60)
#     for label, key in METRIC_LABELS:
#         print(f"{label:<22}{_fmt(naive[key]):<22}{_fmt(hybrid[key]):<18}")
#     print("=" * 60)
#     _print_route_summary(results)


# def _aggregate_runs(all_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """Aggregates mean and std across multiple full runs for each top-level
#     metric, for both configurations."""
#     aggregate = {}
#     for config in ("naive", "hybrid"):
#         agg = {"config": all_runs[0][config]["config"], "n_examples": all_runs[0][config]["n_examples"]}
#         for _, key in METRIC_LABELS:
#             vals = [r[config][key] for r in all_runs if r[config][key] is not None]
#             if vals:
#                 agg[key] = {
#                     "mean": statistics.mean(vals),
#                     "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
#                 }
#             else:
#                 agg[key] = {"mean": None, "std": None}
#         # carry route info from the last run for the summary
#         agg["per_query"] = all_runs[-1][config]["per_query"]
#         aggregate[config] = agg
#     return aggregate


# def _fmt_mean_std(d):
#     if d is None or d.get("mean") is None:
#         return "    N/A"
#     return f"{d['mean']:.3f}±{d['std']:.3f}"


# def _print_report_with_std(results: Dict[str, Any], runs: int):
#     naive = results["naive"]
#     hybrid = results["hybrid"]

#     print("\n" + "=" * 70)
#     print(f"RAG EVALUATION REPORT  (n={naive['n_examples']} queries x {runs} runs, faithfulness judge={JUDGE_MODEL_FAITHFULNESS}, relevance judge={JUDGE_MODEL_RELEVANCE})")
#     print("=" * 70)
#     print(f"{'Metric':<22}{'Naive (mean±std)':<24}{'Hybrid (mean±std)':<20}")
#     print("-" * 70)
#     for label, key in METRIC_LABELS:
#         print(f"{label:<22}{_fmt_mean_std(naive[key]):<24}{_fmt_mean_std(hybrid[key]):<20}")
#     print("=" * 70)
#     _print_route_summary(results)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="RAG quality evaluation (faithfulness, relevance, context precision/recall, hit rate, MRR)")
#     parser.add_argument("--output", type=str, default=None, help="Optional path to write raw JSON results")
#     parser.add_argument("--top-k", type=int, default=5, help="top_k for retrieval (default 5)")
#     parser.add_argument("--runs", type=int, default=1, help="Number of full repetitions; reports mean +/- std across runs (default 1)")
#     args = parser.parse_args()

#     asyncio.run(main(output_path=args.output, top_k=args.top_k, runs=args.runs))


"""
rag_eval_v2.py

Compares three retrievers:
1) naive_dense
2) current_hybrid_rrf_rerank
3) advanced_hybrid_rrf_rerank

Keeps the same style as the older rag_eval, but adds the third architecture.

What this script measures by default:
- Hit Rate@k
- MRR@k
- Context Precision@k
- Context Recall@k
- Route accuracy / expected-route match

Optional:
- If your backend exposes an answer generator / judges, plug them in where marked.
- If not, the script still produces a clean retrieval benchmark and JSON report.

Usage:
    python -m rag_eval_v2 --runs 3 --output eval_results_v2.json

Adjust the imports in BUILDERS if your module names differ.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------
# Optional imports from your backend
# ---------------------------------------------------------------------

def _safe_import_current_hybrid():
    """
    Try the most likely module names for the current retriever.
    """
    candidates = [
        ("retriever", "HybridRetriever"),
        ("hybrid_retriever", "HybridRetriever"),
        ("advanced_hybrid_retriever", "HybridRetriever"),
    ]
    for mod_name, cls_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            return getattr(mod, cls_name)
        except Exception:
            continue
    raise ImportError(
        "Could not import HybridRetriever from retriever.py, hybrid_retriever.py, "
        "or advanced_hybrid_retriever.py"
    )


def _safe_import_vector_bm25():
    """
    Import your stores. Update names here if your project differs.
    """
    vector_store = None
    bm25_store = None

    for mod_name, cls_name in [
        ("vector_store", "VectorStore"),
        ("stores.vector_store", "VectorStore"),
    ]:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            vector_store = getattr(mod, cls_name)
            break
        except Exception:
            pass

    for mod_name, cls_name in [
        ("bm25_store", "BM25Store"),
        ("stores.bm25_store", "BM25Store"),
    ]:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            bm25_store = getattr(mod, cls_name)
            break
        except Exception:
            pass

    if vector_store is None or bm25_store is None:
        raise ImportError(
            "Could not import VectorStore/BM25Store. Update _safe_import_vector_bm25() "
            "to match your backend module names."
        )
    return vector_store, bm25_store


# ---------------------------------------------------------------------
# Benchmark data
# ---------------------------------------------------------------------

@dataclass
class Example:
    id: str
    query: str
    answer: str
    relevant_ids: List[str]
    expected_route: str


def load_default_examples() -> List[Example]:
    """
    Matches the 10-question benchmark visible in your current eval outputs.
    """
    rows = [
        ("q1", "What is the codename of the project and who manages it?", "Project Alpha, managed by Sarah Jenkins.", ["dummy.txt_page_1_parent_0_child_0"], "dense"),
        ("q2", "What is the launch date for Project Alpha?", "2027-10-15", ["dummy.txt_page_1_parent_0_child_0"], "dense"),
        ("q3", "What is Quantum Neural Architecture Search (QNAS)?", "Quantum Neural Architecture Search (QNAS) is a conceptual framework bridging quantum computing principles with automated machine learning.", ["semantic_fixture.txt_page_1_parent_0_child_0", "semantic_fixture.txt_page_1_parent_0_child_1"], "dense"),
        ("q4", "What is the main drawback that makes QNAS impractical today?", "The main drawback that makes QNAS impractical today is the requirement for low-noise qubits.", ["semantic_fixture.txt_page_1_parent_0_child_0", "semantic_fixture.txt_page_1_parent_0_child_1"], "dense"),
        ("q5", "How does QNAS compare to standard Evolutionary Algorithms?", "QNAS doesn't require iterative generation loops, collapsing the topological search space into an optimal state when evaluated against the validation loss function.", ["semantic_fixture.txt_page_1_parent_0_child_0", "semantic_fixture.txt_page_1_parent_0_child_1"], "dense"),
        ("q6", "list the 8 Java reserved words for data types", "byte, short, int, long, float, double, char, boolean", ["1._language_fundamentals_(1).pdf_page_8_parent_0_child_0"], "hybrid"),
        ("q7", "enum keyword introduced in 1.5v", "The enum keyword was introduced in Java 5 (1.5v).", ["1._language_fundamentals_(1).pdf_page_10_parent_0_child_0", "1._language_fundamentals_(1).pdf_page_10_parent_0_child_1"], "bm25"),
        ("q8", "Project Alpha launch date 2027-10-15", "2027-10-15", ["dummy.txt_page_1_parent_0_child_0"], "bm25"),
        ("q9", "Array length vs length() method", "The length variable is applicable only for arrays and represents the size of the array. The length() method is applicable for String objects and returns the number of characters present in the String.", ["1._language_fundamentals_(1).pdf_page_35_parent_0_child_0", "1._language_fundamentals_(1).pdf_page_35_parent_0_child_1"], "hybrid"),
        ("q10", "NegativeArraySizeException array size rules", "Array size cannot be negative. The allowed data types to specify array size are byte, short, char, int.", ["1._language_fundamentals_(1).pdf_page_28_parent_0_child_0"], "hybrid"),
    ]
    return [Example(*r) for r in rows]


def load_examples(path: Optional[str]) -> List[Example]:
    if not path:
        return load_default_examples()

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Gold/eval file not found: {path}")

    data = json.loads(p.read_text(encoding="utf-8"))
    # Accept either a list of examples or a prior eval_results-like file.
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "examples" in data:
        items = data["examples"]
    elif isinstance(data, dict) and "aggregate" in data:
        # Try to recover from existing eval_results.json style file.
        # We use the first run's naive per_query benchmark as source of truth.
        runs = data.get("runs", [])
        if runs:
            first = runs[0]
            # nested structure: {"naive": {...}, "hybrid": {...}}
            if "naive" in first and "per_query" in first["naive"]:
                items = first["naive"]["per_query"]
            elif "hybrid" in first and "per_query" in first["hybrid"]:
                items = first["hybrid"]["per_query"]
            else:
                raise ValueError("Could not find per_query examples in eval_results-style file")
        else:
            raise ValueError("Empty eval_results-style file")
    else:
        raise ValueError("Unsupported examples file format")

    examples: List[Example] = []
    for item in items:
        examples.append(
            Example(
                id=item["id"],
                query=item["query"],
                answer=item.get("answer", ""),
                relevant_ids=item.get("relevant_ids", []),
                expected_route=item.get("expected_route", "n/a"),
            )
        )
    return examples


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def _extract_chunk_id(doc: Dict[str, Any]) -> str:
    return (
        doc.get("chunk_id")
        or doc.get("id")
        or doc.get("metadata", {}).get("chunk_id")
        or doc.get("metadata", {}).get("id")
        or ""
    )


def _safe_text(doc: Dict[str, Any]) -> str:
    return (
        doc.get("context_text")
        or doc.get("text")
        or doc.get("metadata", {}).get("parent_text")
        or ""
    )


def hit_rate_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    rel = set(relevant_ids)
    return 1.0 if any(r in rel for r in retrieved_ids) else 0.0


def mrr_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    rel = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in rel:
            return 1.0 / i
    return 0.0


def context_precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    if not retrieved_ids:
        return 0.0
    rel = set(relevant_ids)
    return sum(1 for rid in retrieved_ids if rid in rel) / len(retrieved_ids)


def context_recall(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    if not relevant_ids:
        return 0.0
    rel = set(relevant_ids)
    return sum(1 for rid in rel if rid in retrieved_ids) / len(relevant_ids)


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    values = list(values)
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)),
    }


# ---------------------------------------------------------------------
# Retriever wrappers
# ---------------------------------------------------------------------

class NaiveDenseRetriever:
    """
    Simple dense-only baseline, mirroring the old naive comparison.
    """
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def route_for(self, query: str) -> str:
        return "dense"

    def retrieve(self, query: str, top_k: int = 5):
        return self.vector_store.search(query, top_k=top_k)


def build_retrievers(vector_store, bm25_store):
    """
    Three architectures:
      1) naive_dense
      2) current_hybrid_rrf_rerank
      3) advanced_hybrid_rrf_rerank
    """
    current_hybrid_cls = _safe_import_current_hybrid()

    # If advanced_hybrid_retriever.py exists, import it separately.
    advanced_hybrid_cls = current_hybrid_cls
    try:
        mod = __import__("advanced_hybrid_retriever", fromlist=["HybridRetriever"])
        advanced_hybrid_cls = getattr(mod, "HybridRetriever")
    except Exception:
        pass

    return {
        "naive": NaiveDenseRetriever(vector_store),
        "current_hybrid": current_hybrid_cls(vector_store, bm25_store),
        "advanced_hybrid": advanced_hybrid_cls(vector_store, bm25_store),
    }


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_single_retriever(
    name: str,
    retriever,
    examples: List[Example],
    top_k: int,
) -> Dict[str, Any]:
    per_query: List[Dict[str, Any]] = []

    hit_rates = []
    mrrs = []
    precisions = []
    recalls = []
    route_matches = []

    for ex in examples:
        route = "n/a"
        if hasattr(retriever, "route_for"):
            try:
                route = retriever.route_for(ex.query)
            except Exception:
                route = "error"

        try:
            docs = retriever.retrieve(ex.query, top_k=top_k)
        except Exception as e:
            docs = []
            per_query.append({
                "id": ex.id,
                "query": ex.query,
                "answer": ex.answer,
                "retrieved_ids": [],
                "relevant_ids": ex.relevant_ids,
                "expected_route": ex.expected_route,
                "actual_route": route,
                "error": str(e),
                "hit_rate": 0.0,
                "mrr": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
            })
            hit_rates.append(0.0)
            mrrs.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            route_matches.append(0.0)
            continue

        retrieved_ids = [_extract_chunk_id(d) for d in docs]
        retrieved_ids = [x for x in retrieved_ids if x]

        hr = hit_rate_at_k(retrieved_ids, ex.relevant_ids)
        rr = mrr_at_k(retrieved_ids, ex.relevant_ids)
        cp = context_precision_at_k(retrieved_ids, ex.relevant_ids)
        cr = context_recall(retrieved_ids, ex.relevant_ids)

        hit_rates.append(hr)
        mrrs.append(rr)
        precisions.append(cp)
        recalls.append(cr)
        route_matches.append(1.0 if route == ex.expected_route else 0.0)

        per_query.append({
            "id": ex.id,
            "query": ex.query,
            "answer": ex.answer,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": ex.relevant_ids,
            "expected_route": ex.expected_route,
            "actual_route": route,
            "hit_rate": hr,
            "mrr": rr,
            "context_precision": cp,
            "context_recall": cr,
            "route_match": 1.0 if route == ex.expected_route else 0.0,
        })

    return {
        "config": name,
        "n_examples": len(examples),
        "hit_rate": mean_std(hit_rates),
        "mrr": mean_std(mrrs),
        "context_precision": mean_std(precisions),
        "context_recall": mean_std(recalls),
        "route_match": mean_std(route_matches),
        "per_query": per_query,
    }


def flatten_metric(arch_block: Dict[str, Any], key: str) -> Tuple[float, float]:
    v = arch_block.get(key, {})
    if isinstance(v, dict) and "mean" in v and "std" in v:
        return float(v["mean"]), float(v["std"])
    return float(v), 0.0


def summarize_arch(block: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "config": block["config"],
        "n_examples": block["n_examples"],
    }
    for k in ["hit_rate", "mrr", "context_precision", "context_recall", "route_match"]:
        out[k] = block[k]
    return out


def print_summary(aggregate: Dict[str, Dict[str, Any]]) -> None:
    names = ["naive", "current_hybrid", "advanced_hybrid"]
    print("\n" + "=" * 88)
    print("RAG EVALUATION REPORT")
    print("=" * 88)
    print(f"{'Metric':<24} {'Naive (mean±std)':<24} {'Current Hybrid':<24} {'Advanced Hybrid':<24}")
    print("-" * 88)
    metrics = ["hit_rate", "mrr", "context_precision", "context_recall", "route_match"]
    labels = {
        "hit_rate": "Hit Rate@k",
        "mrr": "MRR@k",
        "context_precision": "Context Precision@k",
        "context_recall": "Context Recall",
        "route_match": "Route Match",
    }
    for metric in metrics:
        row = [labels[metric]]
        for name in names:
            m = aggregate[name][metric]
            row.append(f"{m['mean']:.3f}±{m['std']:.3f}")
        print(f"{row[0]:<24} {row[1]:<24} {row[2]:<24} {row[3]:<24}")
    print("=" * 88)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=str, default="eval_results_v2.json")
    parser.add_argument("--gold", type=str, default="", help="Optional gold/examples JSON file")
    args = parser.parse_args()

    VectorStore, BM25Store = _safe_import_vector_bm25()

    # You already load these in your backend code. These constructors may need a tweak
    # if your project uses different initialization signatures.
    vector_store = VectorStore()
    bm25_store = BM25Store()

    retrievers = build_retrievers(vector_store, bm25_store)
    examples = load_examples(args.gold or None)

    runs: List[Dict[str, Any]] = []
    all_blocks = {
        "naive": [],
        "current_hybrid": [],
        "advanced_hybrid": [],
    }

    for run_idx in range(args.runs):
        run_block: Dict[str, Any] = {"run": run_idx + 1}
        for name, retriever in retrievers.items():
            block = evaluate_single_retriever(name, retriever, examples, top_k=args.top_k)
            run_block[name] = block
            all_blocks[name].append(block)
        runs.append(run_block)

    aggregate: Dict[str, Any] = {}
    for name in ["naive", "current_hybrid", "advanced_hybrid"]:
        aggregate[name] = {
            "config": all_blocks[name][0]["config"] if all_blocks[name] else name,
            "n_examples": len(examples),
            "hit_rate": mean_std([b["hit_rate"]["mean"] for b in all_blocks[name]]),
            "mrr": mean_std([b["mrr"]["mean"] for b in all_blocks[name]]),
            "context_precision": mean_std([b["context_precision"]["mean"] for b in all_blocks[name]]),
            "context_recall": mean_std([b["context_recall"]["mean"] for b in all_blocks[name]]),
            "route_match": mean_std([b["route_match"]["mean"] for b in all_blocks[name]]),
            "runs": all_blocks[name],
        }

    report = {
        "runs": runs,
        "aggregate": aggregate,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_summary(aggregate)
    print(f"\nWrote raw results to {out_path.resolve()}")


if __name__ == "__main__":
    main()
