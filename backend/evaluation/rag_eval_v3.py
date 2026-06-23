"""
rag_eval_v3.py — Multi-Document RAG Answer Quality Evaluation
=============================================================

Comprehensive evaluation harness that:
1. Downloads/loads diverse PDFs (5-page to 600-page documents)
2. Ingests each into isolated vector + BM25 stores
3. Generates answers from four architectures:
   - Naive dense retrieval
   - Current hybrid (RRF + rerank)
   - RAPTOR hybrid (RRF + rerank + summary nodes)
   - Routed hybrid (hierarchical chunks + route-aware retrieval + adaptive packing)
4. Compares generated answers against ground truth using an LLM judge
5. Reports per-document, per-category, and aggregate metrics

Usage:
    python rag_eval_v3.py                          # full eval, print report
    python rag_eval_v3.py --output results.json    # also write JSON
    python rag_eval_v3.py --subset 2               # only first 2 documents
    python rag_eval_v3.py --skip-download          # skip PDF download step
"""


import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _path_setup  # noqa: F401

import argparse
import asyncio
import json
import os
import re
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
load_dotenv()

# VLM retriever — optional, requires PyMuPDF + opencv + Ollama VLM model
try:
    from vlm_retriever import VLMRetriever
    HAS_VLM = True
except ImportError as _vlm_err:
    HAS_VLM = False
    _vlm_import_error = str(_vlm_err)

# RAPTOR tree builder — optional, requires scikit-learn
try:
    from raptor import build_and_store_raptor_tree
    HAS_RAPTOR = True
except ImportError as _raptor_err:
    HAS_RAPTOR = False
    _raptor_import_error = str(_raptor_err)

# Adaptive context packer — optional, used by the routed hybrid architecture
try:
    from context_packer import AdaptiveContextPacker
    HAS_CONTEXT_PACKER = True
except ImportError as _packer_err:
    HAS_CONTEXT_PACKER = False
    _packer_import_error = str(_packer_err)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVAL_DIR = Path(__file__).parent.parent / "eval_corpus"
GOLDEN_QA_PATH = Path(__file__).parent / "eval_golden_qa.json"
ISOLATED_DATA_DIR = Path(__file__).parent.parent / "eval_data"

# LLM Judge config
JUDGE_MODEL = "llama3.1:8b"
JUDGE_MAX_RETRIES = 3
JUDGE_RETRY_BASE_DELAY = 5.0
MAX_CONTEXT_CHARS_PER_CHUNK = 800

# Rate limit delay between LLM calls (seconds)
INTER_CALL_DELAY = 1.5

# Ollama config (used when --ollama-model is passed)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Question:
    id: str
    query: str
    ground_truth_answer: str
    question_type: str
    difficulty: str
    relevant_pages: List[int]


@dataclass
class EvalDocument:
    id: str
    filename: str
    category: str
    pages: int
    description: str
    questions: List[Question]


@dataclass
class PerQueryResult:
    question_id: str
    document_id: str
    query: str
    ground_truth: str
    generated_answer: str
    question_type: str
    difficulty: str
    category: str
    relevant_pages: List[int] = field(default_factory=list)
    # Retrieval metrics
    retrieved_ids: List[str] = field(default_factory=list)
    hit_rate: float = 0.0
    mrr: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    # Answer quality metrics (LLM judge)
    answer_correctness: Optional[float] = None
    answer_completeness: Optional[float] = None
    faithfulness: Optional[float] = None
    correctness_detail: str = ""
    completeness_detail: str = ""
    faithfulness_detail: str = ""
    # -----------------------------------------------------------------------
    # Per-step pipeline trace (populated for architectures that expose it)
    # -----------------------------------------------------------------------
    query_variants: List[str] = field(default_factory=list)
    route: str = ""
    query_features: Dict[str, Any] = field(default_factory=dict)
    embedding_candidates: List[Dict[str, Any]] = field(default_factory=list)
    bm25_candidates: List[Dict[str, Any]] = field(default_factory=list)
    after_rrf: List[Dict[str, Any]] = field(default_factory=list)
    after_rerank: List[Dict[str, Any]] = field(default_factory=list)
    submitted_to_llm: List[Dict[str, Any]] = field(default_factory=list)
    candidate_counts: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    stage_timings_ms: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metrics

# ---------------------------------------------------------------------------
# Metrics (retrieval)
# ---------------------------------------------------------------------------

def _extract_chunk_id(doc: Dict[str, Any]) -> str:
    return (
        doc.get("chunk_id")
        or doc.get("id")
        or doc.get("metadata", {}).get("chunk_id")
        or doc.get("metadata", {}).get("id")
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


def context_recall_score(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    if not relevant_ids:
        return 0.0
    rel = set(relevant_ids)
    return sum(1 for rid in rel if rid in retrieved_ids) / len(relevant_ids)


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)),
    }

# FIX (#7): category_breakdown and question_type_breakdown can have buckets
# with as few as 5-15 questions (the eval corpus is intentionally skewed
# toward us_constitution -- 65/100 questions -- leaving thin samples for
# every other document/category/type). A std of 0.3-0.48 on a mean computed
# from 5-15 points is large relative to the mean and is consistent with
# small-sample volatility, not necessarily genuine answer-quality variance.
# Reporting mean/std alone invites reading those numbers with more
# confidence than the sample size supports. This adds `n` (sample size) and
# a `low_sample_size` flag (n < LOW_SAMPLE_SIZE_THRESHOLD) next to mean/std
# so a reader of the JSON can see at a glance which buckets are statistically
# thin, instead of having to cross-reference question counts separately.
LOW_SAMPLE_SIZE_THRESHOLD = 15

def mean_std_n(values: Sequence[float]) -> Dict[str, float]:
    values = [v for v in values if v is not None]
    n = len(values)
    base = mean_std(values)
    base["n"] = n
    base["low_sample_size"] = n < LOW_SAMPLE_SIZE_THRESHOLD
    # Standard error of the mean -- useful alongside std for judging how
    # precisely the mean itself is known, which matters more than std when
    # comparing two thin-sample buckets against each other.
    base["sem"] = float(base["std"] / (n ** 0.5)) if n > 0 else 0.0
    return base

def _trace_to_dict(trace: Any) -> Dict[str, Any]:
    """Normalize dict/dataclass traces into plain JSON-safe dictionaries."""
    if not trace:
        return {}
    if isinstance(trace, dict):
        return dict(trace)
    try:
        return asdict(trace)
    except Exception:
        if hasattr(trace, "__dict__"):
            return dict(vars(trace))
    return {}





def _classify_retrieval_complexity(query: str) -> str:
    """Deterministic query complexity gate used by the routed hybrid eval path.

    Mirrors the heuristic used in query_understanding.py closely enough for
    evaluation, but without requiring the Groq-backed planner.
    """
    q = (query or "").strip().lower()
    if not q:
        return "simple"

    global_patterns = [
        "overall", "in general", "summarize", "summary", "across the document",
        "across documents", "global", "whole document", "entire document",
    ]
    # Genuine comparative/multi-hop phrasing only. Previous version also
    # included bare "why", "how does", "how do", "and ", " or " — these match
    # nearly any natural-language question (e.g. "Why must HTTP/1.1 servers
    # send a Content-Length header?" or "What does Section 3.2 require and
    # how is it enforced?"), which is what drove this classifier to label
    # ~99% of queries as multi_hop and over-trigger multi-query expansion
    # even on simple technical lookups (rfc_q1 correctness 0.8 -> 0.2).
    multi_hop_patterns = [
        "compare", "comparison", "difference between", "differences between",
        "relationship between", "versus", " vs ",
        "how does it compare", "how does this compare",
        "both ", "across multiple", "across several", "across all",
        "first ", "before ", "then after",
    ]
    # Unambiguous constitutional/legal-domain terms only.
    legal_patterns = [
        "article", "amendment", "clause", "constitution",
        "congress", "senate", "house of representatives",
        "jurisdiction", "shall not",
        "ratification", "apportion", "succession", "voting age",
        "inferior federal courts",
    ]
    # Ambiguous terms that also show up routinely in technical specs (RFCs
    # say "Section 3.2", API docs discuss "regulation" of traffic, etc).
    # These only count as a legal signal when no technical-domain term is
    # also present in the same query.
    legal_ambiguous_patterns = [
        "section", "court", "statute", "regulation", "ordinance",
        "power to", "treaty",
    ]
    technical_domain_patterns = [
        "rfc", "http/", "protocol", "header field", "status code",
        "specification", "syntax", "grammar", "implementation",
        "client", "server", "request", "response", "endpoint", "api",
        "payload", "encoding",
    ]
    negation_patterns = [
        " not ", "n't", " never ", " no ", " none ", " without ", " except ",
        " excluding ", " isn't ", " doesn't ", " don't ", " didn't ",
        " cannot ", " can't ", " won't ", " wouldn't ", " shouldn't ",
    ]

    def is_legal(text: str) -> bool:
        if any(p in text for p in legal_patterns):
            return True
        if any(p in text for p in legal_ambiguous_patterns):
            return not any(p in text for p in technical_domain_patterns)
        return False

    if any(p in q for p in global_patterns):
        return "global"
    if any(p in q for p in multi_hop_patterns):
        return "multi_hop"
    if any(p in q for p in negation_patterns) or is_legal(q):
        return "multi_hop"

    word_count = len(q.split())
    # Genuine multi-clause structure: the query asks two distinct things
    # ("...and what...", "...and how...") or contains multiple question
    # marks. A single incidental "and"/"or" inside an otherwise single-fact
    # question (e.g. "qualifications... House of Representatives") does not
    # by itself indicate multi-hop reasoning is required, so it's no longer
    # sufficient on its own. Sheer length is also not a complexity signal:
    # precisely-worded single-fact questions are often long.
    multi_clause_connectors = (
        " and what ", " and how ", " and which ", " and why ", " and when ",
        " and does ", " and is ", " and are ", " and what's ",
        " or what ", " or how ", " or which ",
    )
    has_multi_clause = (
        any(c in q for c in multi_clause_connectors)
        or q.count("?") >= 2
    )
    if has_multi_clause:
        return "multi_hop"

    simple_prefixes = ("what is ", "who is ", "when is ", "where is ", "define ", "what does ")
    if word_count <= 8 and q.startswith(simple_prefixes):
        return "simple"

    # Nothing above flagged this as global, negated, legal/ambiguous-legal,
    # or genuinely multi-clause — it's a single, direct lookup regardless of
    # how many words it took to phrase precisely. Only fall back to
    # multi_hop if it's still touching legal/ambiguous-domain vocabulary
    # without a clear technical-domain override (handled inside is_legal).
    if not is_legal(q):
        return "simple"

    return "multi_hop"


def _prepare_chunks_for_llm(retriever: Any, query: str, retrieved: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Pack the retrieved context only when the retriever exposes a packer."""
    if not retrieved:
        return []

    pack_for_llm = getattr(retriever, "pack_for_llm", None)
    if callable(pack_for_llm):
        try:
            packed = pack_for_llm(query, retrieved, top_k=top_k)
            if isinstance(packed, list):
                return packed
        except TypeError:
            try:
                packed = pack_for_llm(query, retrieved)
                if isinstance(packed, list):
                    return packed
            except Exception:
                pass
        except Exception:
            pass

    return retrieved


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """Evaluates answer quality against ground truth.

    Supports two backends selected via the ``ollama_model`` constructor arg:

    * **Groq** (default) — requires ``GROQ_API_KEY`` in env.
    * **Ollama** — connects to a local Ollama server.  Pass
      ``ollama_model="<model_name>"`` (e.g. ``"llama3.2"``).  The base URL
      defaults to ``http://localhost:11434`` but can be overridden via the
      ``OLLAMA_BASE_URL`` env var.
    """

    def __init__(self, ollama_model: Optional[str] = None):
        self._backend = "ollama" if ollama_model else "groq"

        # FIX (#6): the judge's regex-salvage fallback (see _parse_raw_json)
        # was good defensive engineering, but the fraction of scores that
        # came from a clean JSON parse vs. a regex-salvaged partial parse
        # was never surfaced anywhere -- a reader of eval_results_*.json had
        # no way to know how much of any architecture's score came from a
        # less-trustworthy salvage path. These counters are read out into
        # the final report's metadata (see run_eval's output assembly).
        self.parse_stats = {"clean_json": 0, "salvaged": 0, "repaired": 0, "empty": 0}

        if ollama_model:
            # ---- Ollama backend (OpenAI-compatible /v1 endpoint) ----
            try:
                from openai import AsyncOpenAI  # openai>=1.0 required
                self._ollama_model = ollama_model
                base_url = OLLAMA_BASE_URL.rstrip("/") + "/v1"
                self.client = AsyncOpenAI(
                    base_url=base_url,
                    api_key="ollama",  # Ollama ignores the key; field is required
                )
                self.enabled = True
                print(f"  [JUDGE] Using Ollama backend: {ollama_model} @ {base_url}")
            except ImportError:
                self.enabled = False
                self._init_error = "openai package not installed (pip install openai)"
                self._ollama_model = None
        else:
            # ---- Groq backend (original behaviour) ----
            self._ollama_model = None
            try:
                from groq import AsyncGroq
                api_key = os.getenv("GROQ_API_KEY")
                if api_key:
                    self.client = AsyncGroq(api_key=api_key)
                    self.enabled = True
                else:
                    self.enabled = False
                    self._init_error = "no GROQ_API_KEY"
            except ImportError:
                self.enabled = False
                self._init_error = "groq package not installed"

    def _parse_raw_json(self, raw: Optional[str], _count: bool = True) -> Dict[str, Any]:
        """Parse a raw LLM response as JSON, with tolerant fallback extraction.

        The judge sometimes emits nearly-JSON output with one or more fields
        malformed or truncated. When that happens, we salvage any score/detail
        fields we can find instead of dropping the whole result to an empty dict.

        `_count`, when True, records the outcome in self.parse_stats so the
        final report can state what fraction of judge calls needed the
        salvage path (see FIX #6 note in __init__).
        """
        if not raw:
            if _count:
                self.parse_stats["empty"] += 1
            return {}

        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                if _count:
                    self.parse_stats["clean_json"] += 1
                return parsed
        except Exception:
            pass

        salvaged: Dict[str, Any] = {}

        score_patterns = {
            "correctness_score": r'"correctness_score"\s*:\s*([\d.]+)',
            "completeness_score": r'"completeness_score"\s*:\s*([\d.]+)',
            "faithfulness_score": r'"faithfulness_score"\s*:\s*([\d.]+)',
        }
        for key, pat in score_patterns.items():
            m = re.search(pat, cleaned, flags=re.IGNORECASE)
            if m:
                try:
                    salvaged[key] = float(m.group(1))
                except Exception:
                    continue

        detail_patterns = {
            "correctness_detail": r'"correctness_detail"\s*:\s*"([^"]*)"',
            "completeness_detail": r'"completeness_detail"\s*:\s*"([^"]*)"',
            "faithfulness_detail": r'"faithfulness_detail"\s*:\s*"([^"]*)"',
            "rationale": r'"rationale"\s*:\s*"([^"]*)"',
        }
        for key, pat in detail_patterns.items():
            m = re.search(pat, cleaned, flags=re.IGNORECASE | re.DOTALL)
            if m:
                salvaged[key] = m.group(1).strip()

        if not salvaged:
            score_match = re.search(
                r'"(?:correctness|completeness|faithfulness)_score"\s*:\s*([\d.]+)',
                cleaned,
                flags=re.IGNORECASE,
            )
            if score_match:
                key_match = re.search(r'"(\w+_score)"', cleaned)
                if key_match:
                    try:
                        salvaged[key_match.group(1)] = float(score_match.group(1))
                    except Exception:
                        pass

        if _count:
            if salvaged:
                self.parse_stats["salvaged"] += 1
            else:
                self.parse_stats["empty"] += 1

        return salvaged
    
    @staticmethod
    def _normalize_judge_score(result: Dict[str, Any], score_key: str, claims_key: str = "claims", supported_labels: Optional[set] = None) -> Optional[float]:
        """Normalize judge scores defensively.

        If the judge emits claim-level verdicts, recompute the score locally so
        we never trust a summed/unnormalized score above 1.0.
        """
        supported_labels = supported_labels or set()
        claims = result.get(claims_key)
        if isinstance(claims, list) and claims:
            verdicts = []
            for item in claims:
                if isinstance(item, dict):
                    verdict = str(item.get("verdict", "")).strip().lower()
                    if verdict:
                        verdicts.append(verdict)
                elif isinstance(item, str):
                    verdicts.append(item.strip().lower())
            if verdicts:
                supported = sum(1 for v in verdicts if v in supported_labels)
                return max(0.0, min(1.0, supported / len(verdicts)))

        score = result.get(score_key)
        if score is None:
            return None
        try:
            return max(0.0, min(1.0, float(score)))
        except Exception:
            return None
        
    async def _repair_json_output(self, prompt: str, raw: str) -> Dict[str, Any]:
        """Ask the judge model to reformat a broken response into valid JSON."""
        repair_prompt = f"""The previous response was malformed or incomplete.
Return ONLY valid JSON that matches the original schema.
Do not add commentary, markdown, or extra keys.

Original task:
{prompt}

Broken response:
{raw}
"""
        try:
            if self._backend == "ollama":
                resp = await self.client.chat.completions.create(
                    model=self._ollama_model,
                    messages=[{"role": "user", "content": repair_prompt}],
                    temperature=0.0,
                    max_tokens=512,
                    extra_body={"options": {"num_ctx": 2048}},
                )
                repaired = resp.choices[0].message.content or ""
                # FIX (#6): the triggering parse failure on the original
                # response was already recorded by the caller's
                # _parse_raw_json(raw) call. Count this repair-pass parse
                # under its own bucket instead of double-counting it as a
                # second "salvaged"/"empty" outcome for the same judge call.
                result = self._parse_raw_json(repaired, _count=False)
                if result:
                    self.parse_stats["repaired"] += 1
                return result

            resp = await self.client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "You output only valid JSON."},
                    {"role": "user", "content": repair_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            repaired = resp.choices[0].message.content or ""
            result = self._parse_raw_json(repaired, _count=False)
            if result:
                self.parse_stats["repaired"] += 1
            return result
        except Exception as exc:
            print(f"  [JUDGE] Repair pass failed: {exc}")
            return {}

    async def _judge_json(self, prompt: str) -> Dict[str, Any]:
        """Call the configured LLM backend with retry logic and return parsed JSON."""
        if self._backend == "ollama":
            return await self._judge_json_ollama(prompt)
        return await self._judge_json_groq(prompt)

    async def _judge_json_ollama(self, prompt: str) -> Dict[str, Any]:
        """Call Ollama via OpenAI-compatible endpoint. Simple retry on connection errors."""
        delay = JUDGE_RETRY_BASE_DELAY
        for attempt in range(1, JUDGE_MAX_RETRIES + 1):
            try:
                # Ollama doesn't support response_format=json_object for all models,
                # so we rely on prompt instructions + regex fallback parsing.
                #
                # IMPORTANT: num_ctx MUST match generator.py's OLLAMA_NUM_CTX (2048)
                # and raptor.py's RAPTOR_OLLAMA_NUM_CTX (2048) exactly. Ollama keys
                # its loaded-model cache on the full request config, not just the
                # model name -- a different num_ctx (Ollama's bare default is 4096)
                # makes Ollama treat this as a DIFFERENT model load and spin up a
                # second llama-server process alongside the one already holding
                # ~1.9GB of VRAM for the generator/raptor calls. On a 4GB card that
                # second allocation is exactly the cudaMalloc OOM seen in eval runs.
                resp = await self.client.chat.completions.create(
                    model=self._ollama_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512,  # judge responses are short JSON; enough headroom
                    extra_body={"options": {"num_ctx": 2048}},
                )
                raw = resp.choices[0].message.content
                result = self._parse_raw_json(raw)
                if result:
                    return result
                repair = await self._repair_json_output(prompt, raw)
                if repair:
                    return repair
                # Empty parse — retry
                if attempt < JUDGE_MAX_RETRIES:
                    print(f"  [JUDGE/ollama] Empty JSON parse (attempt {attempt}), retrying...")
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    print(f"  [JUDGE/ollama] Could not parse JSON after {JUDGE_MAX_RETRIES} attempts.")
                    return {}
            except Exception as e:
                err_str = str(e)
                if attempt < JUDGE_MAX_RETRIES:
                    print(f"  [JUDGE/ollama] Error (attempt {attempt}): {err_str[:120]}. Retrying in {delay:.0f}s...")
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    print(f"  [JUDGE/ollama] Non-retryable error: {err_str[:120]}")
                    return {}
        return {}

    async def _judge_json_groq(self, prompt: str) -> Dict[str, Any]:
        """Call Groq with retry on rate limits (original behaviour)."""
        delay = JUDGE_RETRY_BASE_DELAY
        for attempt in range(1, JUDGE_MAX_RETRIES + 1):
            try:
                resp = await self.client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content
                result = self._parse_raw_json(raw)
                if result:
                    return result
                repair = await self._repair_json_output(prompt, raw or "")
                if repair:
                    return repair
                return {}
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str.lower():
                    # FIX (High): Groq error messages use mixed formats like
                    # "10m23.808s" or "623.808s". Parse minutes + seconds separately.
                    wait = delay  # fallback
                    m_full = re.search(
                        r"try again in\s+(?:(\d+)m\s*)?(?:([\d.]+)s)?",
                        err_str,
                        re.IGNORECASE,
                    )
                    if m_full and (m_full.group(1) or m_full.group(2)):
                        minutes = float(m_full.group(1) or 0)
                        seconds = float(m_full.group(2) or 0)
                        wait = minutes * 60 + seconds + 2.0
                    else:
                        m_h = re.search(r"try again in\s+([\d.]+)h", err_str, re.IGNORECASE)
                        if m_h:
                            wait = float(m_h.group(1)) * 3600 + 2.0

                    if wait > 90:
                        print(f"  [JUDGE] TPD quota exhausted (reset in ~{wait/60:.0f}min). Skipping.")
                        return {}

                    if attempt < JUDGE_MAX_RETRIES:
                        print(f"  [JUDGE] Rate-limited (attempt {attempt}/{JUDGE_MAX_RETRIES}), waiting {wait:.0f}s...")
                        await asyncio.sleep(wait)
                        delay *= 2
                        continue
                    else:
                        print(f"  [JUDGE] Rate-limit retries exhausted.")
                        return {}
                elif "400" in err_str or "Failed to generate JSON" in err_str:
                    print(f"  [JUDGE] Non-retryable JSON error: {err_str[:120]}")
                    return {}
                else:
                    print(f"  [JUDGE] Non-retryable error: {err_str[:120]}")
                    return {}
        return {}

    async def combined_judge(
        self,
        query: str,
        ground_truth: str,
        generated: str,
        context: str,
    ) -> Tuple[
        Tuple[Optional[float], str],
        Tuple[Optional[float], str],
        Tuple[Optional[float], str],
    ]:
        """Score correctness, completeness, and faithfulness in two LLM calls.

        FIX (Critical — context bleed): The original single-prompt design included
        RETRIEVED CONTEXT alongside correctness/completeness instructions.  Even
        though the prompt said "for faithfulness only", the judge LLM's
        correctness scores bled over from differing retrieved chunks across
        architectures, producing inconsistent scores for identical generated
        answers (e.g. the negation-category 1.0 → 0.0 flip on uc_q4).

        The fix splits into two calls:
          Call 1: correctness + completeness — sees ONLY query, ground truth,
                  and generated answer (no retrieved context).
          Call 2: faithfulness — sees query, generated answer, and retrieved
                  context (no ground truth, to avoid the reverse bleed).

        This is still ~33 % cheaper than the original 3-call design while
        eliminating the context→correctness contamination.

        Returns three (score, detail) tuples in order:
            (correctness, completeness, faithfulness)
        """
        if not self.enabled:
            msg = f"skipped ({getattr(self, '_init_error', 'unknown')})"
            return (None, msg), (None, msg), (None, msg)

        # --- Call 1: Correctness + Completeness (NO retrieved context) ---
        cc_prompt = f"""You are a rigorous QA evaluator.  Score the GENERATED ANSWER on TWO dimensions.

QUESTION: {query}

GROUND TRUTH: {ground_truth}

GENERATED ANSWER: {generated}

--- SCORING INSTRUCTIONS ---

1. CORRECTNESS (correctness_score 0.0-1.0)
   Decompose GROUND TRUTH into atomic claims.
   correctness_score = correct_count / total_claims.  Empty/refusal answer -> 0.0.

2. COMPLETENESS (completeness_score 0.0-1.0)
   What fraction of GROUND TRUTH key facts appear in the GENERATED ANSWER?
   1.0 = all facts covered, 0.0 = none covered.

Return ONLY valid JSON with exactly these keys (no markdown fences):
{{
  "correctness_score": <float 0-1>,
  "correctness_detail": "<one sentence>",
  "completeness_score": <float 0-1>,
  "completeness_detail": "<one sentence>"
}}"""

        cc_result = await self._judge_json(cc_prompt)

        # --- Call 2: Faithfulness (WITH retrieved context, NO ground truth) ---
        # FIX (#6): faithfulness was scoring lower than correctness for most
        # architectures in eval_results_v3.json, an unusual and
        # counterintuitive pattern. The likely cause: a generated claim that
        # is factually correct but phrased differently from the retrieved
        # chunk's exact wording (different word order, synonym, summarized
        # number) could get marked "unsupported" by a strict literal-match
        # reading, even though it IS grounded in that context. We do NOT
        # re-add ground truth here -- that would reintroduce the exact
        # context-bleed contamination this two-call split was built to
        # eliminate (see the FIX note above on combined_judge). Instead we
        # make the existing "supported if context backs it" instruction
        # explicit about semantic equivalence, so paraphrase isn't conflated
        # with fabrication.
        faith_prompt = f"""You are a strict FAITHFULNESS evaluator.

QUESTION: {query}

GENERATED ANSWER: {generated}

RETRIEVED CONTEXT:
{context}

--- SCORING INSTRUCTIONS ---

Every claim in GENERATED ANSWER: "supported" if RETRIEVED CONTEXT backs it,
"unsupported" otherwise. faithfulness_score = supported_count / total_claims.
No claims -> 1.0. Do NOT penalise for omitting facts.

IMPORTANT: judge support by MEANING, not by matching exact wording. A claim
is "supported" if RETRIEVED CONTEXT entails the same fact, even when the
generated answer uses different words, a different sentence structure, a
rounded number, or summarizes/paraphrases the context. Only mark a claim
"unsupported" if RETRIEVED CONTEXT contradicts it OR contains no information
relevant to it at all -- not merely because the phrasing differs.

Return ONLY valid JSON with exactly these keys (no markdown fences):
{{
  "faithfulness_score": <float 0-1>,
  "faithfulness_detail": "<one sentence>"
}}"""

        faith_result = await self._judge_json(faith_prompt)

        def _extract(result: dict, score_key: str, detail_key: str, claims_labels: Optional[set] = None) -> Tuple[Optional[float], str]:
            score = self._normalize_judge_score(result, score_key, supported_labels=claims_labels or set())
            detail = result.get(detail_key, "")
            if score is None:
                return None, "judge returned invalid JSON"
            return float(score), str(detail)

        return (
            _extract(cc_result, "correctness_score", "correctness_detail", {"correct", "supported"}),
            _extract(cc_result, "completeness_score", "completeness_detail"),
            _extract(faith_result, "faithfulness_score", "faithfulness_detail", {"supported"}),
        )

    async def answer_correctness(self, query: str, ground_truth: str, generated: str) -> Tuple[float, str]:
        """Score how correct the generated answer is vs ground truth."""
        if not self.enabled:
            return None, f"skipped ({getattr(self, '_init_error', 'unknown')})"

        prompt = f"""You are a strict ANSWER CORRECTNESS evaluator for a QA system.

Your task: Compare the GENERATED ANSWER against the GROUND TRUTH ANSWER and determine
how factually correct the generated answer is.

QUESTION: {query}

GROUND TRUTH ANSWER: {ground_truth}

GENERATED ANSWER: {generated}

Instructions:
1. Decompose the GROUND TRUTH into atomic factual claims.
2. For each claim, check if the GENERATED ANSWER states the same fact (even if phrased differently).
   - "correct": the generated answer states this fact correctly
   - "missing": the generated answer doesn't mention this fact
   - "incorrect": the generated answer contradicts this fact
3. correctness_score = correct_count / total_ground_truth_claims
4. If the generated answer is empty or a refusal, score 0.0.
5. If ground truth has no factual claims, score 1.0.

Return ONLY valid JSON:
{{"claims": [{{"claim": "...", "verdict": "correct|missing|incorrect"}}], "correctness_score": <float 0.0-1.0>}}"""

        result = await self._judge_json(prompt)
        score = self._normalize_judge_score(result, "correctness_score", supported_labels={"correct"})
        if score is None:
            return None, "judge returned invalid JSON"
        return float(score), json.dumps(result.get("claims", []))
    
    async def answer_completeness(self, query: str, ground_truth: str, generated: str) -> Tuple[float, str]:
        """Score how complete the generated answer is."""
        if not self.enabled:
            return None, f"skipped ({getattr(self, '_init_error', 'unknown')})"

        prompt = f"""You are an ANSWER COMPLETENESS evaluator.

Score how thoroughly the GENERATED ANSWER covers the key information from the GROUND TRUTH.

QUESTION: {query}
GROUND TRUTH: {ground_truth}
GENERATED ANSWER: {generated}

Score from 0.0 to 1.0:
- 1.0: covers all key facts from ground truth
- 0.5: covers roughly half the key information
- 0.0: misses all key information or is empty

Return ONLY JSON: {{"completeness_score": <float 0-1>, "rationale": "<one sentence>"}}"""

        result = await self._judge_json(prompt)
        score = self._normalize_judge_score(result, "completeness_score")
        if score is None:
            return None, "judge returned invalid JSON"
        return float(score), result.get("rationale", "")
    
    async def faithfulness(self, answer: str, context: str) -> Tuple[float, str]:
        """Score whether the answer only contains facts from the retrieved context."""
        if not self.enabled:
            return None, f"skipped ({getattr(self, '_init_error', 'unknown')})"

        prompt = f"""You are a strict FAITHFULNESS evaluator.

Check if every claim in the ANSWER is supported by the RETRIEVED CONTEXT.
Do NOT penalize for omitting facts. Only check what IS stated.

RETRIEVED CONTEXT:
{context}

ANSWER TO EVALUATE:
{answer}

Instructions:
1. Extract atomic claims from the ANSWER.
2. For each: "supported" if context backs it, "unsupported" if not.
3. faithfulness_score = supported_count / total_claims. If no claims, score 1.0.

Return ONLY JSON:
{{"claims": [{{"claim": "...", "verdict": "supported|unsupported"}}], "faithfulness_score": <float 0.0-1.0>}}"""

        result = await self._judge_json(prompt)
        score = self._normalize_judge_score(result, "faithfulness_score", supported_labels={"supported"})
        if score is None:
            return None, "judge returned invalid JSON"
        return float(score), json.dumps(result.get("claims", []))

# ---------------------------------------------------------------------------
# Retriever wrappers
# ---------------------------------------------------------------------------

class NaiveDenseRetriever:
    """Simple dense-only baseline."""
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.last_trace = None

    def retrieve(self, query: str, top_k: int = 5):
        import time as _time
        t0 = _time.perf_counter() * 1000.0
        results = self.vector_store.search(query, top_k=top_k)
        t1 = _time.perf_counter() * 1000.0
        for doc in results:
            meta = doc.get("metadata", {})
            doc["context_text"] = meta.get("parent_text", doc.get("text", ""))

        def _chunk_summary(docs):
            out = []
            for d in docs:
                meta = d.get("metadata", {})
                out.append({
                    "chunk_id": d.get("chunk_id", ""),
                    "text": (d.get("text", ""))[:300],
                    "score": float(d.get("score", 0.0)),
                    "page": meta.get("page_start", 0),
                    "source": meta.get("document_id", ""),
                })
            return out

        self.last_trace = {
            "route": "dense",
            "query_variants": [query],
            "query_features": {},
            "embedding_candidates": _chunk_summary(results),
            "bm25_candidates": [],
            "after_rrf": [],
            "after_rerank": _chunk_summary(results),
            "submitted_to_llm": _chunk_summary(results),
            "candidate_counts": {"dense": len(results), "final": len(results)},
            "notes": [],
            "stage_timings_ms": {
                "first_stage_retrieval": round(t1 - t0, 2),
                "total": round(t1 - t0, 2),
            },
        }
        return results

    def route_for(self, query: str) -> str:
        return "dense"


class HybridRetrieverWrapper:
    """Wraps the HybridRetriever to also expose route info and detailed trace."""
    def __init__(self, hybrid_retriever):
        self._inner = hybrid_retriever
        self.last_trace = None

    def retrieve(self, query: str, top_k: int = 5):
        results = self._inner.retrieve(query, top_k=top_k)
        raw_trace = _trace_to_dict(getattr(self._inner, "last_trace", None))

        def _chunk_summary(docs):
            out = []
            for d in docs:
                meta = d.get("metadata", {})
                out.append({
                    "chunk_id": d.get("chunk_id", ""),
                    "text": (d.get("text", ""))[:300],
                    "score": float(d.get("score", d.get("fused_score", d.get("rerank_score", 0.0)))),
                    "rerank_score": float(d.get("rerank_score", 0.0)) if "rerank_score" in d else None,
                    "fused_score": float(d.get("fused_score", 0.0)) if "fused_score" in d else None,
                    "page": meta.get("page_start", 0),
                    "source": meta.get("document_id", ""),
                })
            return out

        route = raw_trace.get("route", "hybrid") or "hybrid"
        variants = raw_trace.get("variants", raw_trace.get("query_variants", [query]))
        features = raw_trace.get("features", raw_trace.get("query_features", {}))
        timings = raw_trace.get("stage_timings_ms", {})
        notes = raw_trace.get("notes", [])
        candidate_counts = raw_trace.get("candidate_counts", {})

        self.last_trace = {
            "route": route,
            "query_variants": variants,
            "query_features": features,
            "embedding_candidates": raw_trace.get("embedding_candidates", []),
            "bm25_candidates": raw_trace.get("bm25_candidates", []),
            "after_rrf": raw_trace.get("after_rrf", []),
            "after_rerank": raw_trace.get("after_rerank", _chunk_summary(results)),
            "submitted_to_llm": raw_trace.get("submitted_to_llm", _chunk_summary(results)),
            "candidate_counts": candidate_counts,
            "notes": notes,
            "stage_timings_ms": timings,
        }
        return results

    def route_for(self, query: str) -> str:
        try:
            return self._inner.router.route(query)
        except Exception:
            return "unknown"


class RaptorRetrieverWrapper:
    """Wraps HybridRetriever with a RAPTOR tree pre-built on the same chunks.

    At evaluation time, query complexity is auto-classified so that:
    - global/summary queries route to RAPTOR summary nodes only
    - simple factual queries use the fast dense-only path (skip reranker)
    - multi-hop queries use the full hybrid pipeline (same as HybridRetrieverWrapper)
    """

    def __init__(self, hybrid_retriever, raptor_tree=None):
        self._inner = hybrid_retriever
        self._tree = raptor_tree  # RaptorTree | None (if tree-build failed)
        self.last_trace = None
        # Instantiate the classifier without __init__ to avoid needing GROQ_API_KEY.
        # classify_retrieval_complexity is a pure deterministic method, no network calls.
        import query_understanding as _qu_mod
        self._qu_obj = _qu_mod.QueryUnderstandingEngine.__new__(_qu_mod.QueryUnderstandingEngine)
        self._classify = lambda q: _qu_mod.QueryUnderstandingEngine.classify_retrieval_complexity(self._qu_obj, q)

    def retrieve(self, query: str, top_k: int = 5):
        import time as _time
        t0 = _time.perf_counter() * 1000.0

        complexity = self._classify(query.lower())
        use_summary_nodes = self._tree is not None and not self._tree.is_empty()
        # If no RAPTOR tree was built (e.g. too few chunks to cluster),
        # fall back to the standard hybrid path regardless of classification.
        if self._tree is None or self._tree.is_empty():
            complexity = None
            use_summary_nodes = False

        results = self._inner.retrieve(
            query,
            top_k=top_k,
            complexity=complexity,
            use_summary_nodes=use_summary_nodes,
        )
        t1 = _time.perf_counter() * 1000.0

        raw_trace = _trace_to_dict(getattr(self._inner, "last_trace", None))

        def _chunk_summary(docs):
            out = []
            for d in docs:
                meta = d.get("metadata", {})
                out.append({
                    "chunk_id": d.get("chunk_id", ""),
                    "text": (d.get("text", ""))[:300],
                    "score": float(d.get("score", d.get("fused_score", 0.0))),
                    "page": meta.get("page_start", 0),
                    "source": meta.get("document_id", ""),
                    "node_type": meta.get("node_type", "raw"),
                })
            return out

        self.last_trace = {
            "route": "raptor/" + (complexity or "hybrid"),
            "query_variants": raw_trace.get("variants", raw_trace.get("query_variants", [query])),
            "query_features": {"retrieval_complexity": complexity},
            "embedding_candidates": raw_trace.get("embedding_candidates", []),
            "bm25_candidates": raw_trace.get("bm25_candidates", []),
            "after_rrf": raw_trace.get("after_rrf", []),
            "after_rerank": raw_trace.get("after_rerank", _chunk_summary(results)),
            "submitted_to_llm": raw_trace.get("submitted_to_llm", _chunk_summary(results)),
            "candidate_counts": raw_trace.get("candidate_counts", {"final": len(results)}),
            "notes": raw_trace.get("notes", []) + [
                "raptor_tree=" + ("built" if self._tree and not self._tree.is_empty() else "missing")
            ],
            "stage_timings_ms": raw_trace.get("stage_timings_ms", {"total": round(t1 - t0, 2)}),
        }
        return results

    def route_for(self, query: str) -> str:
        complexity = self._classify(query.lower())
        return "raptor/" + complexity


class RoutedHybridRetrieverWrapper:
    """Route-aware wrapper that adds hierarchical packing to the advanced hybrid retriever.

    This is the new architecture being evaluated:
      - query routing / complexity gating
      - hierarchical chunks (raw + parent)
      - BM25 + dense retrieval
      - late reranker inside the underlying retriever
      - adaptive context packing before generation
    """

    def __init__(self, hybrid_retriever, packer: Optional[Any] = None):
        self._inner = hybrid_retriever
        self._packer = packer or (AdaptiveContextPacker() if HAS_CONTEXT_PACKER else None)
        self.last_trace = None
        self.last_packed_chunks: List[Dict[str, Any]] = []
        self.last_pack_trace: Dict[str, Any] = {}

    def _chunk_summary(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for d in docs:
            meta = d.get("metadata", {}) or {}
            out.append({
                "chunk_id": d.get("chunk_id", ""),
                "text": (d.get("context_text") or d.get("text", ""))[:300],
                "score": float(d.get("score", d.get("fused_score", d.get("rerank_score", 0.0)))),
                "rerank_score": float(d.get("rerank_score", 0.0)) if "rerank_score" in d else None,
                "fused_score": float(d.get("fused_score", 0.0)) if "fused_score" in d else None,
                "page": meta.get("page_start", 0),
                "source": meta.get("document_id", ""),
                "node_type": meta.get("node_type", d.get("node_type", "raw")),
            })
        return out

    def route_for(self, query: str) -> str:
        complexity = _classify_retrieval_complexity(query)
        return f"routed_hybrid/{complexity}"

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        complexity: Optional[str] = None,
        use_summary_nodes: bool = False,
    ) -> List[Dict[str, Any]]:
        import time as _time

        t0 = _time.perf_counter() * 1000.0
        route_complexity = complexity or _classify_retrieval_complexity(query)
        kwargs = {"top_k": top_k, "complexity": route_complexity, "use_summary_nodes": use_summary_nodes}
        try:
            results = self._inner.retrieve(query, **kwargs)
        except TypeError:
            kwargs.pop("use_summary_nodes", None)
            try:
                results = self._inner.retrieve(query, **kwargs)
            except TypeError:
                kwargs.pop("complexity", None)
                results = self._inner.retrieve(query, top_k=top_k)
        t1 = _time.perf_counter() * 1000.0

        raw_trace = _trace_to_dict(getattr(self._inner, "last_trace", None))
        self.last_trace = {
            "route": f"routed_hybrid/{route_complexity}",
            "query_variants": raw_trace.get("variants", raw_trace.get("query_variants", [query])),
            "query_features": {
                **raw_trace.get("features", raw_trace.get("query_features", {})),
                "retrieval_complexity": route_complexity,
                "use_summary_nodes": bool(use_summary_nodes),
                "has_context_packer": bool(self._packer),
            },
            "embedding_candidates": raw_trace.get("embedding_candidates", []),
            "bm25_candidates": raw_trace.get("bm25_candidates", []),
            "after_rrf": raw_trace.get("after_rrf", []),
            "after_rerank": raw_trace.get("after_rerank", self._chunk_summary(results)),
            "submitted_to_llm": raw_trace.get("submitted_to_llm", []),
            "candidate_counts": dict(raw_trace.get("candidate_counts", {})),
            "notes": list(raw_trace.get("notes", [])),
            "stage_timings_ms": dict(raw_trace.get("stage_timings_ms", {})),
        }
        self.last_trace["stage_timings_ms"]["total"] = round(t1 - t0, 2)
        self.last_trace["candidate_counts"]["raw"] = len(results)
        self.last_packed_chunks = list(results)
        self.last_pack_trace = {}
        return results

    def pack_for_llm(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not chunks:
            self.last_packed_chunks = []
            self.last_pack_trace = {
                "packed": 0,
                "token_budget": 0,
                "mode": _classify_retrieval_complexity(query),
            }
            if self.last_trace is not None:
                self.last_trace["submitted_to_llm"] = []
                self.last_trace["candidate_counts"]["packed"] = 0
            return []

        mode = _classify_retrieval_complexity(query)
        if self._packer is None:
            packed = chunks[:top_k]
            token_budget = 0
        else:
            # FIX (#9): previously this always passed a hardcoded
            # max_tokens=1650/1100 straight through, which BYPASSES the
            # context packer's own density-aware budget logic (max_tokens
            # short-circuits _target_tokens entirely). We now pass the base
            # budget via answer_plan only, and let AdaptiveContextPacker.pack
            # inspect the retrieved chunks' `doc_shape` (set by chunking.py)
            # to widen the budget for equation-dense or clause-dense
            # documents -- so Newton's Principia / the Attention paper don't
            # get truncated at the same flat ceiling as a sparse document.
            base_budget = 1650 if mode != "simple" else 1100
            packed = self._packer.pack(
                query=query,
                chunks=chunks,
                answer_plan={"retrieval_complexity": mode, "mode": mode, "budget": {"max_tokens_for_context": base_budget}},
                mode=mode,
            )
            token_budget = self._packer._target_tokens(
                {"retrieval_complexity": mode, "mode": mode, "budget": {"max_tokens_for_context": base_budget}},
                mode,
                chunks=chunks,
            )

        self.last_packed_chunks = list(packed)

        def _summarize_for_trace(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return self._chunk_summary(docs)

        self.last_pack_trace = {
            "packed": len(packed),
            "token_budget": token_budget,
            "mode": mode,
        }

        if self.last_trace is not None:
            self.last_trace["submitted_to_llm"] = _summarize_for_trace(packed)
            self.last_trace["candidate_counts"]["packed"] = len(packed)
            notes = self.last_trace.setdefault("notes", [])
            notes.append(
                f"context_packer={'enabled' if self._packer is not None else 'fallback'}; mode={mode}; packed={len(packed)}; budget={token_budget}"
            )

        return packed




def _import_stores():
    """Import VectorStore and BM25Store."""
    from vector_store import VectorStore
    from bm25_store import BM25Store
    return VectorStore, BM25Store


def _import_hybrid_retriever():
    """Import the HybridRetriever class."""
    from retriever import HybridRetriever
    return HybridRetriever


def _import_generator():
    """Import the AnswerGenerator."""
    from generator import AnswerGenerator
    return AnswerGenerator


# ---------------------------------------------------------------------------
# Document ingestion (isolated)
# ---------------------------------------------------------------------------

def ingest_document(
    pdf_path: str,
    vector_store,
    bm25_store,
) -> List[Dict[str, Any]]:
    """Ingest a single PDF into the provided stores."""
    from ingestion import DocumentIngestor

    ingestor = DocumentIngestor()
    doc_id = Path(pdf_path).stem.replace(" ", "_").lower()
    chunks = ingestor.parse_pdf(pdf_path, doc_id)

    if chunks:
        vector_store.add_chunks(chunks)
        bm25_store.add_chunks(chunks)

    return chunks


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

async def generate_answer(
    generator,
    query: str,
    chunks: List[Dict[str, Any]],
    max_retries: int = 2,
) -> str:
    """Generate an answer and return the text."""
    ERROR_SENTINEL = "An error occurred during answer generation."

    for attempt in range(1, max_retries + 1):
        try:
            final_answer = ""
            async for event in generator.generate_stream(query, chunks, mode="doc_rag"):
                if event.get("event") == "final":
                    try:
                        data = json.loads(event["data"])
                        final_answer = data.get("answer", "")
                    except Exception:
                        pass
            if final_answer and final_answer != ERROR_SENTINEL:
                return final_answer
        except Exception as e:
            print(f"    Generator error (attempt {attempt}/{max_retries}): {e}")

        if attempt < max_retries:
            await asyncio.sleep(2.0 * attempt)

    return ""


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

async def evaluate_document(
    doc: EvalDocument,
    retrievers: Dict[str, Any],
    generator,
    judge: LLMJudge,
    top_k: int = 5,
) -> Dict[str, List[PerQueryResult]]:
    """Evaluate all questions for a single document across all architectures."""

    results_by_arch = {name: [] for name in retrievers}

    for q_idx, question in enumerate(doc.questions):
        print(f"    Q{q_idx+1}/{len(doc.questions)}: {question.query[:60]}...")

        for arch_name, retriever in retrievers.items():
            try:
                retrieved = retriever.retrieve(question.query, top_k=top_k)
            except Exception as e:
                print(f"      [{arch_name}] Retrieval error: {e}")
                retrieved = []

            llm_chunks = _prepare_chunks_for_llm(retriever, question.query, retrieved, top_k=top_k)
            retriever_trace = _trace_to_dict(getattr(retriever, "last_trace", None))
            retrieved_ids = [_extract_chunk_id(d) for d in retrieved if _extract_chunk_id(d)]

            relevant_pages = set(question.relevant_pages)
            retrieved_pages = set()
            retrieved_pages_ordered = []
            for d in retrieved:
                meta = d.get("metadata", {})
                page = meta.get("page_start", 0)
                if page == 0:
                    cid = _extract_chunk_id(d)
                    m = re.search(r"page_(\d+)", cid)
                    if m:
                        page = int(m.group(1))
                retrieved_pages.add(page)
                if page not in retrieved_pages_ordered:
                    retrieved_pages_ordered.append(page)

            page_hit = 1.0 if relevant_pages & retrieved_pages else 0.0

            page_mrr = 0.0
            for i, d in enumerate(retrieved, start=1):
                meta = d.get("metadata", {})
                page = meta.get("page_start", 0)
                if page == 0:
                    cid = _extract_chunk_id(d)
                    m_match = re.search(r"page_(\d+)", cid)
                    if m_match:
                        page = int(m_match.group(1))
                if page in relevant_pages:
                    page_mrr = 1.0 / i
                    break

            r = len(relevant_pages)
            if r == 0:
                page_precision = 1.0
            else:
                top_r_pages = retrieved_pages_ordered[:r]
                if not top_r_pages:
                    page_precision = 0.0
                else:
                    page_precision = sum(1 for p in top_r_pages if p in relevant_pages) / len(top_r_pages)

            if relevant_pages:
                page_recall = len(relevant_pages & retrieved_pages) / len(relevant_pages)
            else:
                page_recall = 0.0

            answer = await generate_answer(generator, question.query, llm_chunks)
            await asyncio.sleep(INTER_CALL_DELAY)

            context_str = "\n---\n".join(
                (d.get("context_text", d.get("text", "")))[:MAX_CONTEXT_CHARS_PER_CHUNK]
                for d in llm_chunks
            )

            (correct_score, correct_detail), (complete_score, complete_detail), (faith_score, faith_detail) = (
                await judge.combined_judge(
                    question.query,
                    question.ground_truth_answer,
                    answer,
                    context_str,
                )
            )
            await asyncio.sleep(INTER_CALL_DELAY)

            result = PerQueryResult(
                question_id=question.id,
                document_id=doc.id,
                query=question.query,
                ground_truth=question.ground_truth_answer,
                generated_answer=answer,
                question_type=question.question_type,
                difficulty=question.difficulty,
                category=doc.category,
                relevant_pages=list(question.relevant_pages),
                retrieved_ids=retrieved_ids,
                hit_rate=page_hit,
                mrr=page_mrr,
                context_precision=page_precision,
                context_recall=page_recall,
                answer_correctness=correct_score,
                answer_completeness=complete_score,
                faithfulness=faith_score,
                correctness_detail=correct_detail,
                completeness_detail=complete_detail,
                faithfulness_detail=faith_detail,
                route=retriever_trace.get("route", ""),
                query_variants=retriever_trace.get("query_variants", []),
                query_features=retriever_trace.get("query_features", {}),
                embedding_candidates=retriever_trace.get("embedding_candidates", []),
                bm25_candidates=retriever_trace.get("bm25_candidates", []),
                after_rrf=retriever_trace.get("after_rrf", []),
                after_rerank=retriever_trace.get("after_rerank", []),
                submitted_to_llm=retriever_trace.get("submitted_to_llm", []),
                candidate_counts=retriever_trace.get("candidate_counts", {}),
                notes=retriever_trace.get("notes", []),
                stage_timings_ms=retriever_trace.get("stage_timings_ms", {}),
            )
            results_by_arch[arch_name].append(result)

    return results_by_arch

# ---------------------------------------------------------------------------
# Detailed per-question pipeline trace report
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int = 200) -> str:
    """Truncate text for display, appending … if cut."""
    text = (text or "").replace("\n", " ").strip()
    return text[:max_len] + ("…" if len(text) > max_len else "")


def _score_bar(score: Optional[float], width: int = 20) -> str:
    """Render a simple ASCII progress bar for a 0-1 score."""
    if score is None:
        return "[" + " " * width + "] N/A"
    filled = round(score * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.2f}"


def _candidate_preview(candidate: Dict[str, Any]) -> str:
    page = candidate.get("page", "?")
    score = candidate.get("score", candidate.get("fused_score", candidate.get("rerank_score", 0.0)))
    cid = candidate.get("chunk_id", "")
    text = _truncate(candidate.get("text", ""), 180)
    return f"page={page}  score={float(score):.4f}  id={cid[:48]}  {text}"


def build_detailed_per_question_report(
    all_results: Dict[str, List["PerQueryResult"]],
) -> Dict[str, Any]:
    """Serialize the full per-question trace report for JSON output."""
    arch_names = list(all_results.keys())
    question_map: Dict[str, Dict[str, Any]] = {}

    for arch, results in all_results.items():
        for r in results:
            slot = question_map.setdefault(r.question_id, {"_meta": r})
            slot[arch] = r

    questions = []
    for _, arch_map in question_map.items():
        meta: "PerQueryResult" = arch_map["_meta"]
        arch_payload = {}
        for arch in arch_names:
            r = arch_map.get(arch)
            if r is None:
                continue
            arch_payload[arch] = {
                "route": r.route,
                "query_variants": r.query_variants,
                "query_features": r.query_features,
                "candidate_counts": r.candidate_counts,
                "notes": r.notes,
                "stage_timings_ms": r.stage_timings_ms,
                "embedding_candidates": r.embedding_candidates,
                "bm25_candidates": r.bm25_candidates,
                "after_rrf": r.after_rrf,
                "after_rerank": r.after_rerank,
                "submitted_to_llm": r.submitted_to_llm,
                "generated_answer": r.generated_answer,
                "answer_correctness": r.answer_correctness,
                "answer_completeness": r.answer_completeness,
                "faithfulness": r.faithfulness,
                "correctness_detail": r.correctness_detail,
                "completeness_detail": r.completeness_detail,
                "faithfulness_detail": r.faithfulness_detail,
            }

        questions.append({
            "question_id": meta.question_id,
            "document_id": meta.document_id,
            "query": meta.query,
            "ground_truth": meta.ground_truth,
            "question_type": meta.question_type,
            "difficulty": meta.difficulty,
            "category": meta.category,
            "relevant_pages": meta.relevant_pages,
            "retrieved_ids": meta.retrieved_ids,
            "hit_rate": meta.hit_rate,
            "mrr": meta.mrr,
            "context_precision": meta.context_precision,
            "context_recall": meta.context_recall,
            "architectures": arch_payload,
        })

    return {
        "architectures": arch_names,
        "questions": questions,
    }


def print_detailed_per_question_report(
    all_results: Dict[str, List["PerQueryResult"]],
) -> None:
    """Print a detailed, human-readable pipeline trace for every question."""
    report = build_detailed_per_question_report(all_results)
    arch_names = report["architectures"]
    questions = report["questions"]

    DIVIDER = "=" * 110

    print("\n" + DIVIDER)
    print("  DETAILED PER-QUESTION PIPELINE TRACE REPORT")
    print(f"  {len(questions)} questions · {len(arch_names)} architectures: {', '.join(arch_names)}")
    print(DIVIDER)

    for q_num, item in enumerate(questions, start=1):
        print(f"\n{'━' * 110}")
        print(
            f"  Q{q_num:03d}  [{item['question_id']}]  |  doc: {item['document_id']}  |  "
            f"type: {item['question_type']}  |  difficulty: {item['difficulty']}"
        )
        print(f"  QUERY : {item['query']}")
        print(f"  RELEVANT PAGES: {item['relevant_pages']}")
        print(f"  GROUND TRUTH: {_truncate(item['ground_truth'], 300)}")
        print(f"{'━' * 110}")

        for arch in arch_names:
            r = item["architectures"].get(arch)
            if not r:
                continue

            print(f"\n  ┌─ {arch.upper()} {'─' * (98 - len(arch))}┐")
            feat_str = ", ".join(
                k for k, v in r["query_features"].items()
                if v and k != "word_count"
            ) or "—"
            print(f"  │ route: {r['route'] or '?'}")
            print(f"  │ query variants: {' | '.join(r['query_variants']) if r['query_variants'] else item['query']}")
            print(f"  │ features: {feat_str}")
            if r["notes"]:
                print(f"  │ notes: {'; '.join(r['notes'])}")

            def _print_candidates(title: str, candidates: List[Dict[str, Any]]):
                print(f"  │ {title}:")
                if not candidates:
                    print("  │   (none)")
                    return
                for ci, c in enumerate(candidates[:5], 1):
                    print(f"  │   {ci}. {_candidate_preview(c)}")

            _print_candidates("embedding candidates", r["embedding_candidates"])
            _print_candidates("bm25 candidates", r["bm25_candidates"])
            _print_candidates("after rrf", r["after_rrf"])
            _print_candidates("after rerank", r["after_rerank"])
            _print_candidates("submitted to llm", r["submitted_to_llm"])

            print("  │ generated answer:")
            answer_lines = (r["generated_answer"] or "(no answer)").splitlines()[:10]
            for line in answer_lines:
                print(f"  │   {line}")
            if len((r["generated_answer"] or "").splitlines()) > 10:
                print("  │   …")

            print(
                "  │ scores: "
                f"correctness={_score_bar(r['answer_correctness'])}  "
                f"completeness={_score_bar(r['answer_completeness'])}  "
                f"faithfulness={_score_bar(r['faithfulness'])}"
            )
            if r["correctness_detail"]:
                print(f"  │ correctness note: {_truncate(r['correctness_detail'], 120)}")
            if r["completeness_detail"]:
                print(f"  │ completeness note: {_truncate(r['completeness_detail'], 120)}")
            if r["faithfulness_detail"]:
                print(f"  │ faithfulness note: {_truncate(r['faithfulness_detail'], 120)}")

            if r["stage_timings_ms"]:
                timing_parts = [
                    f"{k}={v:.1f}ms"
                    for k, v in r["stage_timings_ms"].items()
                    if isinstance(v, (int, float))
                ]
                print(f"  │ timings: {' | '.join(timing_parts)}")
            print(f"  └{'─' * 108}┘")

    print(f"\n{DIVIDER}")
    print("  END OF DETAILED PIPELINE TRACE REPORT")
    print(DIVIDER + "\n")


# ---------------------------------------------------------------------------
# Report generation# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    ("Answer Correctness", "answer_correctness"),
    ("Answer Completeness", "answer_completeness"),
    ("Faithfulness", "faithfulness"),
    ("Hit Rate@5", "hit_rate"),
    ("MRR@5", "mrr"),
    ("Context Precision@5", "context_precision"),
    ("Context Recall", "context_recall"),
]


def _fmt_ms(d: Dict[str, float]) -> str:
    if d["mean"] == 0.0 and d["std"] == 0.0:
        return "  N/A"
    return f"{d['mean']:.3f}±{d['std']:.3f}"


def compute_aggregate(
    all_results: Dict[str, List[PerQueryResult]],
) -> Dict[str, Dict[str, Any]]:
    """Compute aggregate metrics for each architecture."""
    aggregate = {}
    for arch_name, results in all_results.items():
        agg = {"config": arch_name, "n_questions": len(results)}
        for label, key in METRIC_KEYS:
            vals = [getattr(r, key) for r in results if getattr(r, key) is not None]
            agg[key] = mean_std(vals)
        aggregate[arch_name] = agg
    return aggregate


def compute_category_breakdown(
    all_results: Dict[str, List[PerQueryResult]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Compute per-category answer correctness for each architecture."""
    breakdown = {}
    for arch_name, results in all_results.items():
        categories = {}
        for r in results:
            if r.category not in categories:
                categories[r.category] = []
            categories[r.category].append(r)

        arch_breakdown = {}
        for cat, cat_results in sorted(categories.items()):
            vals = [r.answer_correctness for r in cat_results if r.answer_correctness is not None]
            arch_breakdown[cat] = mean_std_n(vals)

        breakdown[arch_name] = arch_breakdown
    return breakdown


def compute_question_type_breakdown(
    all_results: Dict[str, List[PerQueryResult]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Compute per-question-type answer correctness for each architecture."""
    breakdown = {}
    for arch_name, results in all_results.items():
        types = {}
        for r in results:
            if r.question_type not in types:
                types[r.question_type] = []
            types[r.question_type].append(r)

        arch_breakdown = {}
        for qtype, type_results in sorted(types.items()):
            vals = [r.answer_correctness for r in type_results if r.answer_correctness is not None]
            arch_breakdown[qtype] = mean_std_n(vals)

        breakdown[arch_name] = arch_breakdown
    return breakdown


def print_report(
    aggregate: Dict[str, Dict[str, Any]],
    cat_breakdown: Dict[str, Dict[str, Dict[str, Any]]],
    type_breakdown: Dict[str, Dict[str, Dict[str, Any]]],
    total_docs: int,
    total_questions: int,
) -> None:
    arch_names = list(aggregate.keys())

    print("\n" + "=" * 100)
    print(f"MULTI-DOCUMENT RAG ANSWER QUALITY REPORT")
    print(f"({total_docs} documents, {total_questions} questions, {len(arch_names)} architectures)")
    print("=" * 100)

    # Header
    header = f"{'Metric':<24}"
    for name in arch_names:
        header += f" {name:<24}"
    print(header)
    print("-" * max(100, 24 * (len(arch_names) + 1)))

    # Metrics
    for label, key in METRIC_KEYS:
        row = f"{label:<24}"
        for name in arch_names:
            row += f" {_fmt_ms(aggregate[name][key]):<24}"
        print(row)

    print("=" * 100)

    # Category breakdown
    print(f"\nPER-CATEGORY BREAKDOWN (Answer Correctness):")
    header = f"{'Category':<30}"
    for name in arch_names:
        header += f" {name:<24}"
    print(header)
    print("-" * 100)

    all_cats = set()
    for arch_bd in cat_breakdown.values():
        all_cats.update(arch_bd.keys())

    for cat in sorted(all_cats):
        row = f"{cat:<30}"
        for name in arch_names:
            if cat in cat_breakdown[name]:
                row += f" {_fmt_ms(cat_breakdown[name][cat]):<24}"
            else:
                row += f" {'  N/A':<24}"
        print(row)

    print("=" * 100)

    # Question type breakdown
    print(f"\nPER-QUESTION-TYPE BREAKDOWN (Answer Correctness):")
    header = f"{'Question Type':<24}"
    for name in arch_names:
        header += f" {name:<24}"
    print(header)
    print("-" * 100)

    all_types = set()
    for arch_bd in type_breakdown.values():
        all_types.update(arch_bd.keys())

    for qtype in sorted(all_types):
        row = f"{qtype:<24}"
        for name in arch_names:
            if qtype in type_breakdown[name]:
                row += f" {_fmt_ms(type_breakdown[name][qtype]):<24}"
            else:
                row += f" {'  N/A':<24}"
        print(row)

    print("=" * 100)


# ---------------------------------------------------------------------------
# Golden QA loading
# ---------------------------------------------------------------------------

def load_golden_qa(path: Optional[str] = None) -> List[EvalDocument]:
    """Load the golden QA dataset."""
    qa_path = Path(path) if path else GOLDEN_QA_PATH
    if not qa_path.exists():
        raise FileNotFoundError(f"Golden QA file not found: {qa_path}")

    data = json.loads(qa_path.read_text(encoding="utf-8"))
    documents = []

    for doc_data in data["documents"]:
        questions = [
            Question(
                id=q["id"],
                query=q["query"],
                ground_truth_answer=q["ground_truth_answer"],
                question_type=q["question_type"],
                difficulty=q["difficulty"],
                relevant_pages=q["relevant_pages"],
            )
            for q in doc_data["questions"]
        ]
        documents.append(EvalDocument(
            id=doc_data["id"],
            filename=doc_data["filename"],
            category=doc_data["category"],
            pages=doc_data["pages"],
            description=doc_data["description"],
            questions=questions,
        ))

    return documents


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(
    output_path: Optional[str] = None,
    top_k: int = 5,
    subset: Optional[int] = None,
    skip_download: bool = False,
    golden_qa_path: Optional[str] = None,
    ollama_model: Optional[str] = None,
    enable_vlm: bool = False,
    vlm_model: str = "qwen2.5vl:7b",
    enable_raptor: bool = False,
    raptor_model: Optional[str] = None,
):
    print("=" * 60)
    print("Multi-Document RAG Evaluation v3")
    print("=" * 60)

    if not skip_download:
        print("\n[Step 0] Downloading evaluation corpus...")
        try:
            from download_eval_corpus import main as download_all
            download_all()
        except Exception as e:
            print(f"  Warning: Download failed ({e}). Continuing with existing files.")

    print("\n[Step 1] Loading golden QA dataset...")
    documents = load_golden_qa(golden_qa_path)
    if subset:
        documents = documents[:subset]

    total_questions = sum(len(d.questions) for d in documents)
    print(f"  Loaded {len(documents)} documents with {total_questions} total questions")

    print("\n[Step 2] Importing backend components...")
    VectorStore, BM25Store = _import_stores()
    HybridRetriever = _import_hybrid_retriever()
    AnswerGenerator = _import_generator()
    generator = AnswerGenerator()
    judge = LLMJudge(ollama_model=ollama_model)

    if not judge.enabled:
        print("  WARNING: LLM judge is disabled. Only retrieval metrics will be computed.")

    # --- VLM architecture validation ---
    vlm_available = False
    if enable_vlm:
        if not HAS_VLM:
            print(f"  WARNING: --enable-vlm set but VLM dependencies missing: {_vlm_import_error}")
            print("           Install with: pip install PyMuPDF opencv-python-headless Pillow openai numpy")
        else:
            vlm_available = True
            print(f"  VLM architecture enabled (model: {vlm_model})")

    # --- RAPTOR architecture validation ---
    raptor_available = False
    # Resolve the model used for RAPTOR summarization.  Default: use the same
    # model as the generator (ollama_model) so Ollama only holds one set of
    # weights in VRAM at a time.  Falls back to raptor.RAPTOR_OLLAMA_MODEL if
    # neither --raptor-model nor --ollama-model was specified.
    if enable_raptor:
        from raptor import RAPTOR_OLLAMA_MODEL as _DEFAULT_RAPTOR_MODEL
        _resolved_raptor_model = raptor_model or ollama_model or _DEFAULT_RAPTOR_MODEL
        if not HAS_RAPTOR:
            print(f"  WARNING: --enable-raptor set but RAPTOR dependencies missing: {_raptor_import_error}")
            print("           Install with: pip install scikit-learn")
        else:
            raptor_available = True
            print(f"  RAPTOR architecture enabled (summarizer model: {_resolved_raptor_model}).")
    else:
        _resolved_raptor_model = None

    all_results: Dict[str, List[PerQueryResult]] = {
        "naive_dense": [],
        "hybrid": [],
        "routed_hybrid": [],
    }
    if vlm_available:
        all_results["vlm_maxsim"] = []
    if raptor_available:
        all_results["hybrid_raptor"] = []

    # FIX (#4): track, per document, whether the RAPTOR tree actually built
    # ("built"), fell back to plain hybrid retrieval ("empty_tree_fallback_to_hybrid"),
    # or errored out, so the final report can state plainly what fraction of
    # hybrid_raptor's results are "true RAPTOR" vs. hybrid-in-disguise instead
    # of that information existing only as a per-question trace note nobody
    # aggregates.
    raptor_fallback_log: Dict[str, str] = {}

    for doc_idx, doc in enumerate(documents):
        print(f"\n[Step 3.{doc_idx+1}] Processing: {doc.filename} ({doc.category}, ~{doc.pages} pages)")

        pdf_path = EVAL_DIR / doc.filename
        if not pdf_path.exists():
            print(f"  SKIP: PDF not found at {pdf_path}")
            continue

        iso_dir = ISOLATED_DATA_DIR / doc.id
        if iso_dir.exists():
            shutil.rmtree(iso_dir)
        iso_dir.mkdir(parents=True, exist_ok=True)

        chroma_dir = str(iso_dir / "chroma")
        bm25_dir = str(iso_dir / "bm25")

        try:
            vector_store = VectorStore(persist_dir=chroma_dir)
            bm25_store = BM25Store(persist_dir=bm25_dir)

            print(f"  Ingesting {doc.filename}...")
            chunks = ingest_document(str(pdf_path), vector_store, bm25_store)
            print(f"  Ingested {len(chunks)} chunks")

            if not chunks:
                print(f"  SKIP: No chunks extracted from {doc.filename}")
                continue

            hybrid_retriever = HybridRetriever(vector_store, bm25_store)

            retrievers = {
                "naive_dense": NaiveDenseRetriever(vector_store),
                "hybrid": HybridRetrieverWrapper(hybrid_retriever),
                "routed_hybrid": RoutedHybridRetrieverWrapper(hybrid_retriever),
            }

            # --- VLM architecture: separate ingestion pipeline ---
            if vlm_available:
                try:
                    print(f"  [VLM] Ingesting {doc.filename} through VLM pipeline...")
                    vlm_ret = VLMRetriever(vlm_model=vlm_model)
                    vlm_ret.ingest_pdf(str(pdf_path), document_id=Path(pdf_path).stem.replace(' ', '_').lower())
                    if vlm_ret.index and not vlm_ret.index.is_empty():
                        retrievers["vlm_maxsim"] = vlm_ret
                    else:
                        print(f"  [VLM] WARNING: No pages indexed for {doc.filename}")
                except Exception as vlm_exc:
                    print(f"  [VLM] ERROR during ingestion: {vlm_exc}")
                    import traceback
                    traceback.print_exc()

            # --- RAPTOR architecture: build summary tree on already-ingested chunks ---
            raptor_tree = None
            if raptor_available:
                try:
                    print(f"  [RAPTOR] Building summary tree for {doc.filename} ({len(chunks)} chunks)...")
                    raptor_doc_id = chunks[0]["document_id"]
                    raptor_tree = build_and_store_raptor_tree(
                        document_id=raptor_doc_id,
                        chunks=chunks,
                        vector_store=vector_store,
                        bm25_store=bm25_store,
                        source_file=doc.filename,
                        raptor_model=_resolved_raptor_model,
                    )
                    n_summary = sum(
                        len(level) for level in raptor_tree.levels.values()
                    ) if raptor_tree else 0
                    if raptor_tree and not raptor_tree.is_empty():
                        print(f"  [RAPTOR] Tree built: {n_summary} summary nodes across {len(raptor_tree.levels)} levels.")
                        raptor_hybrid = HybridRetriever(vector_store, bm25_store)
                        retrievers["hybrid_raptor"] = RaptorRetrieverWrapper(raptor_hybrid, raptor_tree)
                        raptor_fallback_log[doc.filename] = "built"
                    else:
                        # FIX (#4): previously this branch left `hybrid_raptor` out
                        # of `retrievers` entirely for this document, which meant
                        # this document's questions were silently ABSENT from the
                        # hybrid_raptor results in eval_results_*.json -- not
                        # degraded-but-present, just missing, with no record of
                        # why in the final report. We now register it anyway
                        # using a fallback wrapper (tree=None -> RaptorRetrieverWrapper
                        # already falls back to standard hybrid retrieval per-query),
                        # and log the fallback so it's visible in the run summary
                        # and can be cross-checked against per-architecture question
                        # counts in the output JSON.
                        print(f"  [RAPTOR] WARNING: Tree empty for {doc.filename} (too few chunks to cluster?). "
                              f"Registering hybrid_raptor as a hybrid-fallback for this document so its questions "
                              f"are not silently dropped from the architecture's results.")
                        raptor_hybrid = HybridRetriever(vector_store, bm25_store)
                        retrievers["hybrid_raptor"] = RaptorRetrieverWrapper(raptor_hybrid, raptor_tree=None)
                        raptor_fallback_log[doc.filename] = "empty_tree_fallback_to_hybrid"
                except Exception as raptor_exc:
                    print(f"  [RAPTOR] ERROR during tree build: {raptor_exc}")
                    import traceback
                    traceback.print_exc()
                    raptor_fallback_log[doc.filename] = f"build_error: {raptor_exc}"

            n_arch = len(retrievers)
            print(f"  Evaluating {len(doc.questions)} questions across {n_arch} architectures.")
            doc_results = await evaluate_document(doc, retrievers, generator, judge, top_k=top_k)

            for arch_name, results in doc_results.items():
                all_results[arch_name].extend(results)

        except Exception as e:
            print(f"  ERROR processing {doc.filename}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                if iso_dir.exists():
                    shutil.rmtree(iso_dir)
            except Exception:
                pass

    print("\n[Step 4] Computing aggregate metrics...")
    aggregate = compute_aggregate(all_results)
    cat_breakdown = compute_category_breakdown(all_results)
    type_breakdown = compute_question_type_breakdown(all_results)
    detailed_report = build_detailed_per_question_report(all_results)

    # FIX (#7): surface per-document question counts directly in metadata so
    # corpus imbalance (e.g. one document supplying the majority of
    # questions, others contributing only a handful) is visible up front
    # without having to cross-reference per_query counts by hand.
    questions_per_document: Dict[str, int] = {}
    if all_results:
        any_arch_results = next(iter(all_results.values()))
        for r in any_arch_results:
            doc_id = getattr(r, "document_id", "unknown")
            questions_per_document[doc_id] = questions_per_document.get(doc_id, 0) + 1

    print_report(
        aggregate, cat_breakdown, type_breakdown,
        total_docs=len(documents),
        total_questions=total_questions,
    )
    print_detailed_per_question_report(all_results)

    if output_path:
        output = {
            "metadata": {
                "total_documents": len(documents),
                "total_questions": total_questions,
                "architectures": list(all_results.keys()),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                # FIX (#7): see note above -- makes corpus skew visible
                # without cross-referencing per_query by hand.
                "questions_per_document": questions_per_document,
                # FIX (#4): per-document RAPTOR build status, so a reader of
                # this file can tell what fraction of hybrid_raptor's
                # questions were actually answered using a built RAPTOR tree
                # vs. a hybrid-retrieval fallback (or a build error), instead
                # of the two being silently blended into one architecture's
                # aggregate numbers.
                "raptor_fallback_log": raptor_fallback_log,
                # FIX (#6): clean-JSON vs. regex-salvaged vs. repaired vs.
                # totally-unparseable counts across every judge call this
                # run made, so a reader can tell how much of the final
                # scores rest on the tolerant-fallback parsing path instead
                # of a clean structured response.
                "judge_parse_stats": judge.parse_stats,
            },
            "aggregate": aggregate,
            "category_breakdown": cat_breakdown,
            "question_type_breakdown": type_breakdown,
            "detailed_per_question_report": detailed_report,
            "per_query": {
                arch: [asdict(r) for r in results]
                for arch, results in all_results.items()
            },
        }
        Path(output_path).write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote detailed results to {output_path}")

    return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-document RAG answer quality evaluation")
    parser.add_argument("--output", type=str, default="eval_results_v3.json", help="Output JSON path")
    parser.add_argument("--top-k", type=int, default=5, help="top_k for retrieval")
    parser.add_argument("--subset", type=int, default=None, help="Only evaluate first N documents")
    parser.add_argument("--skip-download", action="store_true", help="Skip PDF download step")
    parser.add_argument("--golden-qa", type=str, default=None, help="Path to golden QA JSON")
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=None,
        metavar="MODEL",
        help=(
            "Use a local Ollama model as the LLM judge instead of Groq. "
            "Pass the model name exactly as shown by `ollama list` "
            "(e.g. --ollama-model llama3.2 or --ollama-model mistral). "
            "The Ollama server URL defaults to http://localhost:11434 and "
            "can be overridden via the OLLAMA_BASE_URL env var."
        ),
    )
    parser.add_argument(
        "--enable-vlm",
        action="store_true",
        help=(
            "Enable the third VLM+MaxSim retrieval architecture. "
            "Requires Ollama running with a VLM model (e.g. qwen2.5vl:7b) "
            "plus PyMuPDF, opencv-python-headless, Pillow, openai, and numpy."
        ),
    )
    parser.add_argument(
        "--enable-raptor",
        action="store_true",
        help=(
            "Enable the RAPTOR hybrid+tree architecture. After each document is "
            "ingested, a recursive abstractive summary tree is built using GMM "
            "clustering and Ollama summarization. Global/summary queries are routed "
            "to RAPTOR summary nodes; simple/multi-hop use the standard hybrid path. "
            "Requires scikit-learn (pip install scikit-learn)."
        ),
    )
    parser.add_argument(
        "--raptor-model",
        type=str,
        default=None,
        metavar="RAPTOR_MODEL",
        help=(
            "Ollama model to use for RAPTOR cluster summarization. "
            "Defaults to the value of --ollama-model (if set) so both the "
            "generator and summarizer share the same loaded weights and Ollama "
            "only needs to hold one model in VRAM at a time. "
            "Override only when you explicitly want a different (e.g. smaller) "
            "model for summarization, such as --raptor-model llama3.2:1b."
        ),
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        default="qwen2.5vl:7b",
        metavar="VLM_MODEL",
        help=(
            "Ollama VLM model for the VLM+MaxSim architecture. "
            "Default: qwen2.5vl:7b.  Only used when --enable-vlm is set."
        ),
    )
    args = parser.parse_args()

    asyncio.run(main(
        output_path=args.output,
        top_k=args.top_k,
        subset=args.subset,
        skip_download=args.skip_download,
        golden_qa_path=args.golden_qa,
        ollama_model=args.ollama_model,
        enable_vlm=args.enable_vlm,
        vlm_model=args.vlm_model,
        enable_raptor=args.enable_raptor,
        raptor_model=args.raptor_model,
    ))