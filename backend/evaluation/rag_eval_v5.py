"""
rag_eval_routed_hybrid_gemini.py — Routed Hybrid RAG Answer Quality Evaluation
================================================================================

Stripped-down evaluation harness that:
1. Downloads / loads diverse PDFs from eval_corpus/
2. Ingests each into an isolated vector + BM25 store
3. Generates answers using ONE architecture:
       Routed Hybrid — hierarchical chunks + route-aware retrieval
                       + adaptive context packing
4. Judges generated answers against ground truth via the **Gemini 3.1 Flash Lite**
   model (google-generativeai SDK).
5. Reports per-document, per-category, and aggregate metrics.

LLM backend
-----------
ALL LLM calls (answer generation AND judging) go through Gemini 3.1 Flash Lite.
Set  GEMINI_API_KEY  in your environment (or .env file) before running.

Rate limits (as shown in Google AI Studio — free tier)
------------------------------------------------------
  RPM  =   15   requests per minute
  TPM  =  250 000 tokens per minute
  RPD  =  500   requests per day

The RateLimiter class below enforces all three limits so the eval never
triggers a 429. Each LLM call (generation + 2 judge calls = 3 per question)
is tracked against RPM and RPD counters; a per-call TPM estimate is used as
a soft guard before the call is issued.

Usage:
    python rag_eval_routed_hybrid_gemini.py
    python rag_eval_routed_hybrid_gemini.py --output results.json
    python rag_eval_routed_hybrid_gemini.py --subset 2
    python rag_eval_routed_hybrid_gemini.py --skip-download
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
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
load_dotenv()

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

EVAL_DIR         = Path(__file__).parent.parent / "eval_corpus"
GOLDEN_QA_PATH   = Path(__file__).parent / "eval_golden_qa.json"
ISOLATED_DATA_DIR = Path(__file__).parent.parent / "eval_data"

# Gemini model used for BOTH judging and answer generation
GEMINI_MODEL = "gemini-3.1-flash-lite"

# ---------------------------------------------------------------------------
# Rate-limit constants (sourced from Google AI Studio free-tier dashboard)
#   RPM =    15  requests / minute
#   TPM = 250000 tokens   / minute
#   RPD =   500  requests / day
# ---------------------------------------------------------------------------
RATE_LIMIT_RPM = 15
RATE_LIMIT_TPM = 250_000
RATE_LIMIT_RPD = 500

# Rough token estimate per LLM call (prompt + response).
# Used as a soft pre-flight guard against TPM exhaustion.
# Actual usage may vary; this is conservative.
ESTIMATED_TOKENS_PER_CALL = 1_200

# Judge retry config
JUDGE_MAX_RETRIES      = 3
JUDGE_RETRY_BASE_DELAY = 5.0

MAX_CONTEXT_CHARS_PER_CHUNK = 800

# Hard inter-call floor derived from RPM limit:
#   60 s / 15 RPM = 4.0 s minimum gap between any two API calls.
# We add a small buffer (0.2 s) for clock jitter.
INTER_CALL_DELAY = 60.0 / RATE_LIMIT_RPM + 0.2   # ≈ 4.2 s

# Sample-size threshold below which mean/std should be read cautiously
LOW_SAMPLE_SIZE_THRESHOLD = 15


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
    relevant_pages: List[int]          = field(default_factory=list)
    # Retrieval metrics
    retrieved_ids: List[str]           = field(default_factory=list)
    hit_rate: float                    = 0.0
    mrr: float                         = 0.0
    context_precision: float           = 0.0
    context_recall: float              = 0.0
    # Answer quality metrics (LLM judge)
    answer_correctness: Optional[float]   = None
    answer_completeness: Optional[float]  = None
    faithfulness: Optional[float]         = None
    correctness_detail: str            = ""
    completeness_detail: str           = ""
    faithfulness_detail: str           = ""
    # Per-step pipeline trace
    query_variants: List[str]          = field(default_factory=list)
    route: str                         = ""
    query_features: Dict[str, Any]     = field(default_factory=dict)
    embedding_candidates: List[Dict[str, Any]] = field(default_factory=list)
    bm25_candidates: List[Dict[str, Any]]      = field(default_factory=list)
    after_rrf: List[Dict[str, Any]]    = field(default_factory=list)
    after_rerank: List[Dict[str, Any]] = field(default_factory=list)
    submitted_to_llm: List[Dict[str, Any]] = field(default_factory=list)
    candidate_counts: Dict[str, int]   = field(default_factory=dict)
    notes: List[str]                   = field(default_factory=list)
    stage_timings_ms: Dict[str, float] = field(default_factory=dict)
    # E-5 FIX: explicit failure flag so judge failures count as 0.0 rather
    # than being silently excluded from aggregate means.
    _judge_failed: bool                = False



# ---------------------------------------------------------------------------
# Metric helpers
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
        "std":  float(statistics.pstdev(values)),
    }


def mean_std_n(values: Sequence[float]) -> Dict[str, Any]:
    """mean/std plus sample-size diagnostics."""
    values = [v for v in values if v is not None]
    n    = len(values)
    base = mean_std(values)
    base["n"]               = n
    base["low_sample_size"] = n < LOW_SAMPLE_SIZE_THRESHOLD
    base["sem"]             = float(base["std"] / (n ** 0.5)) if n > 0 else 0.0
    return base


def _trace_to_dict(trace: Any) -> Dict[str, Any]:
    """Normalise dict / dataclass traces into plain JSON-safe dicts."""
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


# ---------------------------------------------------------------------------
# Query complexity classifier  (mirrors query_understanding.py heuristic)
# ---------------------------------------------------------------------------

def _classify_retrieval_complexity(query: str) -> str:
    q = (query or "").strip().lower()
    if not q:
        return "simple"

    global_patterns = [
        "overall", "in general", "summarize", "summary", "across the document",
        "across documents", "global", "whole document", "entire document",
    ]
    multi_hop_patterns = [
        "compare", "comparison", "difference between", "differences between",
        "relationship between", "versus", " vs ",
        "how does it compare", "how does this compare",
        "both ", "across multiple", "across several", "across all",
        "first ", "before ", "then after",
    ]
    legal_patterns = [
        "article", "amendment", "clause", "constitution",
        "congress", "senate", "house of representatives",
        "jurisdiction", "shall not",
        "ratification", "apportion", "succession", "voting age",
        "inferior federal courts",
    ]
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

    word_count = len(q.split())
    simple_prefixes = ("what is ", "who is ", "when is ", "where is ", "define ", "what does ")
    # E-4 FIX: raised from <= 8 to <= 12 to align with the retriever.py
    # multi-hop trigger threshold. Previously, queries up to 8 words starting
    # with a simple prefix were correctly classified as simple but anything
    # 9-12 words would fall through to multi_hop even for plain semantic lookups.
    if word_count <= 12 and q.startswith(simple_prefixes):
        return "simple"

    if not is_legal(q):
        return "simple"

    return "multi_hop"


def _prepare_chunks_for_llm(
    retriever: Any,
    query: str,
    retrieved: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Use the retriever's packer when available, otherwise return as-is."""
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
# Rate limiter  (enforces RPM=15, TPM=250K, RPD=500 from the dashboard)
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Token-bucket + daily counter rate limiter for Gemini 3.1 Flash Lite.

    Limits enforced (free-tier values from Google AI Studio):
        RPM  =    15  requests / minute
        TPM  = 250000 tokens   / minute   (soft guard via token estimate)
        RPD  =   500  requests / day

    Usage:
        limiter = RateLimiter()
        await limiter.acquire(estimated_tokens=1200)
        # ... make the API call ...
        limiter.record(actual_tokens=response.usage.total_tokens)
    """

    def __init__(
        self,
        rpm: int = RATE_LIMIT_RPM,
        tpm: int = RATE_LIMIT_TPM,
        rpd: int = RATE_LIMIT_RPD,
    ):
        self._rpm = rpm
        self._tpm = tpm
        self._rpd = rpd

        # Sliding window: timestamps of the last N requests (minute window)
        self._request_times: list = []
        # Token usage in the current minute window
        self._token_times: list   = []   # list of (timestamp, tokens) tuples

        # Daily counters — reset when calendar date changes
        self._day_requests: int   = 0
        self._day_date: str       = time.strftime("%Y-%m-%d")

        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_day_if_needed(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._day_date:
            self._day_date     = today
            self._day_requests = 0
            print("  [RATE] Daily counter reset for new day.")

    def _prune_minute_window(self) -> None:
        """Drop entries older than 60 s from the sliding windows."""
        cutoff = time.monotonic() - 60.0
        self._request_times = [t for t in self._request_times if t > cutoff]
        self._token_times   = [(t, tok) for t, tok in self._token_times if t > cutoff]

    def _tokens_in_window(self) -> int:
        return sum(tok for _, tok in self._token_times)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def acquire(self, estimated_tokens: int = ESTIMATED_TOKENS_PER_CALL) -> None:
        """
        Block until it is safe to make one more API call.
        Checks RPM, TPM (soft), and RPD limits in order.
        """
        async with self._lock:
            while True:
                self._reset_day_if_needed()
                self._prune_minute_window()

                # --- RPD hard stop ---
                if self._day_requests >= self._rpd:
                    # Compute seconds until midnight for a helpful message
                    now       = time.localtime()
                    secs_left = (23 - now.tm_hour) * 3600 + (59 - now.tm_min) * 60 + (60 - now.tm_sec)
                    print(
                        f"  [RATE] RPD limit reached ({self._day_requests}/{self._rpd}). "
                        f"Waiting {secs_left}s until midnight reset..."
                    )
                    await asyncio.sleep(secs_left + 5)
                    self._reset_day_if_needed()
                    self._prune_minute_window()

                # --- RPM check ---
                if len(self._request_times) >= self._rpm:
                    oldest    = self._request_times[0]
                    wait_secs = 60.0 - (time.monotonic() - oldest) + 0.3
                    if wait_secs > 0:
                        print(f"  [RATE] RPM limit ({self._rpm}/min). Waiting {wait_secs:.1f}s...")
                        await asyncio.sleep(wait_secs)
                    self._prune_minute_window()
                    continue   # re-check all limits after sleep

                # --- TPM soft guard ---
                if self._tokens_in_window() + estimated_tokens > self._tpm:
                    oldest_tok = self._token_times[0][0]
                    wait_secs  = 60.0 - (time.monotonic() - oldest_tok) + 0.3
                    if wait_secs > 0:
                        print(f"  [RATE] TPM soft-limit. Waiting {wait_secs:.1f}s...")
                        await asyncio.sleep(wait_secs)
                    self._prune_minute_window()
                    continue

                # All checks passed — register this request
                now = time.monotonic()
                self._request_times.append(now)
                self._day_requests += 1
                # Reserve estimated tokens immediately; actual count recorded later
                self._token_times.append((now, estimated_tokens))
                break

    def record(self, actual_tokens: int) -> None:
        """
        Optionally update the most-recent token reservation with the real count.
        Call after a successful API response that exposes usage metadata.
        """
        if self._token_times:
            ts, _est = self._token_times[-1]
            self._token_times[-1] = (ts, actual_tokens)

    def status(self) -> Dict[str, Any]:
        """Return a snapshot of current usage for logging."""
        self._prune_minute_window()
        return {
            "rpm_used":  len(self._request_times),
            "rpm_limit": self._rpm,
            "tpm_used":  self._tokens_in_window(),
            "tpm_limit": self._tpm,
            "rpd_used":  self._day_requests,
            "rpd_limit": self._rpd,
        }


# ---------------------------------------------------------------------------
# Gemini 3.1 Flash Lite client (shared between judge and generator)
# ---------------------------------------------------------------------------

class _SingleKeyClient:
    """
    One Gemini key + its own isolated RateLimiter.
    Instantiated by KeyPool; not used directly by callers.
    """

    def __init__(self, api_key: str, label: str):
        self.label        = label
        self.rate_limiter = RateLimiter(
            rpm=RATE_LIMIT_RPM,
            tpm=RATE_LIMIT_TPM,
            rpd=RATE_LIMIT_RPD,
        )
        self._model = None
        self.enabled = False
        self._init_error = ""
        try:
            import google.generativeai as genai  # type: ignore
            # Each key needs its own configured client object.
            # google-generativeai ≥0.8 supports passing the key directly to
            # GenerativeModel via client_options / transport; for older versions
            # we re-call genai.configure() just before every call inside KeyPool.
            self._genai    = genai
            self._api_key  = api_key
            genai.configure(api_key=api_key)
            self._model    = genai.GenerativeModel(GEMINI_MODEL)
            self.enabled   = True
        except Exception as exc:
            self._init_error = str(exc)

    def is_daily_exhausted(self) -> bool:
        """Return True if this key has consumed its full RPD quota for today."""
        st = self.rate_limiter.status()
        return st["rpd_used"] >= st["rpd_limit"]

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int    = 1024,
    ) -> str:
        from google.generativeai.types import GenerationConfig  # type: ignore

        cfg = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Re-configure the global genai state to this key before calling.
        # (google-generativeai uses a module-level default; swapping is safe
        #  here because KeyPool serialises calls through a single asyncio task.)
        self._genai.configure(api_key=self._api_key)

        await self.rate_limiter.acquire(estimated_tokens=ESTIMATED_TOKENS_PER_CALL)

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self._model.generate_content(prompt, generation_config=cfg),
            )
            try:
                actual = response.usage_metadata.total_token_count
                if actual:
                    self.rate_limiter.record(actual)
            except Exception:
                pass
            return response.text or ""
        except Exception as exc:
            err = str(exc)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                print(
                    f"  [GEMINI/{self.label}] Rate-limit 429 despite throttling "
                    f"— backing off 60s: {err[:120]}"
                )
                await asyncio.sleep(60)
            else:
                print(f"  [GEMINI/{self.label}] generate() error: {err[:120]}")
            return ""


class KeyPool:
    """
    Round-robin pool of up to 3 Gemini API keys.

    Keys are read from the environment in order:
        GEMINI_API_KEY        — key 1 (required)
        GEMINI_API_KEY_2      — key 2 (optional)
        GEMINI_API_KEY_3      — key 3 (optional)

    When the active key's RPD quota is exhausted the pool automatically
    rotates to the next available key, so the eval continues without
    waiting for midnight.  If all keys are exhausted the original
    midnight-wait behaviour is preserved.
    """

    def __init__(self):
        self._clients: List[_SingleKeyClient] = []
        self._active_idx: int = 0
        self._lock = asyncio.Lock()

        key_env_names = ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"]
        for i, env_name in enumerate(key_env_names, start=1):
            key = os.getenv(env_name, "").strip()
            if not key:
                continue
            label  = f"KEY{i}"
            client = _SingleKeyClient(key, label)
            if client.enabled:
                self._clients.append(client)
                print(
                    f"  [KEYPOOL] {label} ({env_name}) loaded — "
                    f"RPD budget: {RATE_LIMIT_RPD} req/day"
                )
            else:
                print(
                    f"  [KEYPOOL] {label} ({env_name}) FAILED to init: "
                    f"{client._init_error}"
                )

    @property
    def enabled(self) -> bool:
        return bool(self._clients)

    @property
    def _init_error(self) -> str:
        return "No GEMINI_API_KEY found in environment."

    def _active(self) -> Optional["_SingleKeyClient"]:
        if not self._clients:
            return None
        return self._clients[self._active_idx % len(self._clients)]

    def _rotate(self) -> bool:
        """
        Try to find a non-exhausted key starting after the current one.
        Returns True if a fresh key was found, False if all are exhausted.
        """
        n = len(self._clients)
        for offset in range(1, n + 1):
            candidate_idx = (self._active_idx + offset) % n
            if not self._clients[candidate_idx].is_daily_exhausted():
                self._active_idx = candidate_idx
                print(
                    f"  [KEYPOOL] Rotated to "
                    f"{self._clients[self._active_idx].label} "
                    f"(prev key exhausted its {RATE_LIMIT_RPD} RPD quota)"
                )
                return True
        return False  # every key is exhausted

    @property
    def rate_limiter(self) -> RateLimiter:
        """Expose the *active* key's rate limiter for status logging."""
        c = self._active()
        return c.rate_limiter if c else RateLimiter()

    def pool_status(self) -> List[Dict[str, Any]]:
        """Return a status snapshot for every key — used in the output JSON."""
        return [
            {"label": c.label, **c.rate_limiter.status()}
            for c in self._clients
        ]

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int    = 1024,
    ) -> str:
        async with self._lock:
            client = self._active()
            if client is None:
                return ""

            # If the current key just hit RPD, try to rotate before calling.
            if client.is_daily_exhausted():
                if not self._rotate():
                    # All keys exhausted — fall back to midnight wait on the
                    # first key so existing RateLimiter logic handles it.
                    print(
                        "  [KEYPOOL] All keys exhausted for today. "
                        "Delegating to midnight-wait logic on KEY1..."
                    )
                    self._active_idx = 0
                client = self._active()

        # Release lock before the actual (potentially sleeping) API call so
        # other coroutines are not blocked while we wait for rate limits.
        return await client.generate(prompt, temperature=temperature, max_tokens=max_tokens)


class GeminiClient:
    """
    Public interface used by GeminiAnswerGenerator and LLMJudge.

    Backed by a KeyPool that transparently rotates across up to 3 API keys
    when the daily RPD quota (500 req/day) of the active key is exhausted.

    Environment variables
    ---------------------
        GEMINI_API_KEY        — required, key 1
        GEMINI_API_KEY_2      — optional, key 2
        GEMINI_API_KEY_3      — optional, key 3

    Install dependency:
        pip install google-generativeai
    """

    def __init__(self):
        self._pool = KeyPool()
        self.enabled = self._pool.enabled
        if self.enabled:
            n = len(self._pool._clients)
            print(
                f"  [GEMINI] Model  : {GEMINI_MODEL}\n"
                f"  [GEMINI] Keys   : {n} key(s) loaded  "
                f"(total RPD budget = {n * RATE_LIMIT_RPD} req/day)\n"
                f"  [GEMINI] Limits : RPM={RATE_LIMIT_RPM}  "
                f"TPM={RATE_LIMIT_TPM:,}  RPD={RATE_LIMIT_RPD} per key"
            )
        else:
            self._init_error = self._pool._init_error

    @property
    def rate_limiter(self) -> RateLimiter:
        """Expose active key's limiter for backward-compatible status logging."""
        return self._pool.rate_limiter

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int    = 1024,
    ) -> str:
        """
        Rate-limited call to Gemini 3.1 Flash Lite via the key pool.
        Automatically rotates to the next key when RPD is exhausted.
        """
        if not self.enabled:
            return ""
        return await self._pool.generate(prompt, temperature=temperature, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# LLM Judge  (Gemini 3.1 Flash Lite backend)
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Evaluates answer quality against ground truth using Gemini 3.1 Flash Lite.

    Uses the same two-call split as rag_eval_v3.py to avoid context bleed:
      Call 1 — correctness + completeness (no retrieved context)
      Call 2 — faithfulness (no ground truth)
    """

    def __init__(self, gemini_client: GeminiClient):
        self._gemini = gemini_client
        self.enabled = gemini_client.enabled
        # Parse-quality counters surfaced in the final report metadata
        self.parse_stats = {"clean_json": 0, "salvaged": 0, "repaired": 0, "empty": 0}

    # ------------------------------------------------------------------
    # JSON parsing with tolerant salvage fallback
    # ------------------------------------------------------------------

    def _parse_raw_json(self, raw: Optional[str], _count: bool = True) -> Dict[str, Any]:
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

        for key, pat in {
            "correctness_score":  r'"correctness_score"\s*:\s*([\d.]+)',
            "completeness_score": r'"completeness_score"\s*:\s*([\d.]+)',
            "faithfulness_score": r'"faithfulness_score"\s*:\s*([\d.]+)',
        }.items():
            m = re.search(pat, cleaned, flags=re.IGNORECASE)
            if m:
                try:
                    salvaged[key] = float(m.group(1))
                except Exception:
                    pass

        for key, pat in {
            "correctness_detail":  r'"correctness_detail"\s*:\s*"([^"]*)"',
            "completeness_detail": r'"completeness_detail"\s*:\s*"([^"]*)"',
            "faithfulness_detail": r'"faithfulness_detail"\s*:\s*"([^"]*)"',
            "rationale":           r'"rationale"\s*:\s*"([^"]*)"',
        }.items():
            m = re.search(pat, cleaned, flags=re.IGNORECASE | re.DOTALL)
            if m:
                salvaged[key] = m.group(1).strip()

        if _count:
            if salvaged:
                self.parse_stats["salvaged"] += 1
            else:
                self.parse_stats["empty"] += 1

        return salvaged

    @staticmethod
    def _normalize_score(
        result: Dict[str, Any],
        score_key: str,
        supported_labels: Optional[set] = None,
    ) -> Optional[float]:
        supported_labels = supported_labels or set()
        claims = result.get("claims")
        if isinstance(claims, list) and claims:
            verdicts = []
            for item in claims:
                if isinstance(item, dict):
                    v = str(item.get("verdict", "")).strip().lower()
                    if v:
                        verdicts.append(v)
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

    # ------------------------------------------------------------------
    # Core async call to Gemini with retry
    # ------------------------------------------------------------------

    async def _call_gemini_json(self, prompt: str) -> Dict[str, Any]:
        delay = JUDGE_RETRY_BASE_DELAY
        for attempt in range(1, JUDGE_MAX_RETRIES + 1):
            raw = await self._gemini.generate(prompt, temperature=0.0, max_tokens=512)
            result = self._parse_raw_json(raw)
            if result:
                return result

            # Try a repair pass on empty / malformed output
            repair_prompt = (
                "The previous response was malformed or incomplete.\n"
                "Return ONLY valid JSON that matches the original schema.\n"
                "Do not add commentary, markdown, or extra keys.\n\n"
                f"Original task:\n{prompt}\n\nBroken response:\n{raw}"
            )
            repair_raw = await self._gemini.generate(repair_prompt, temperature=0.0, max_tokens=512)
            repair_result = self._parse_raw_json(repair_raw, _count=False)
            if repair_result:
                self.parse_stats["repaired"] += 1
                return repair_result

            if attempt < JUDGE_MAX_RETRIES:
                print(f"  [JUDGE/gemini] Empty parse (attempt {attempt}), retrying in {delay:.0f}s...")
                await asyncio.sleep(delay)
                delay *= 2
            else:
                print(f"  [JUDGE/gemini] Could not parse JSON after {JUDGE_MAX_RETRIES} attempts.")

        # E-5 FIX: instead of returning {} (which becomes None scores that are
        # silently excluded from aggregate means), return a sentinel with explicit
        # 0.0 scores. The _judge_failed flag lets callers distinguish genuine
        # zero scores from failure-injected zeros in the output JSON.
        return {
            "_judge_failed":       True,
            "correctness_score":   0.0,
            "completeness_score":  0.0,
            "faithfulness_score":  0.0,
            "correctness_detail":  "judge_failed_after_retries",
            "completeness_detail": "judge_failed_after_retries",
            "faithfulness_detail": "judge_failed_after_retries",
        }


    # ------------------------------------------------------------------
    # Public evaluation methods
    # ------------------------------------------------------------------

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
        """
        Two-call judge returning (correctness, completeness, faithfulness).

        Call 1 — correctness + completeness, sees ONLY query/ground-truth/answer.
        Call 2 — faithfulness, sees query/answer/retrieved-context (no ground truth).
        """
        if not self.enabled:
            msg = f"skipped ({getattr(self._gemini, '_init_error', 'unknown')})"
            return (None, msg), (None, msg), (None, msg)

        # ---- Call 1: Correctness + Completeness ----
        cc_prompt = f"""You are a rigorous QA evaluator. Score the GENERATED ANSWER on TWO dimensions.

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

        cc_result = await self._call_gemini_json(cc_prompt)

        # ---- Call 2: Faithfulness ----
        faith_prompt = f"""You are a strict FAITHFULNESS evaluator.

QUESTION: {query}

GENERATED ANSWER: {generated}

RETRIEVED CONTEXT:
{context}

--- SCORING INSTRUCTIONS ---

Every claim in GENERATED ANSWER: "supported" if RETRIEVED CONTEXT backs it,
"unsupported" otherwise. faithfulness_score = supported_count / total_claims.
No claims -> 1.0. Do NOT penalise for omitting facts.

Judge support by MEANING, not by exact wording. A claim is "supported" if
RETRIEVED CONTEXT entails the same fact, even when phrased differently, rounded,
or summarised. Only mark a claim "unsupported" if RETRIEVED CONTEXT contradicts
it OR contains no information relevant to it at all.

Return ONLY valid JSON with exactly these keys (no markdown fences):
{{
  "faithfulness_score": <float 0-1>,
  "faithfulness_detail": "<one sentence>"
}}"""

        faith_result = await self._call_gemini_json(faith_prompt)

        def _extract(
            result: dict,
            score_key: str,
            detail_key: str,
            labels: Optional[set] = None,
        ) -> Tuple[Optional[float], str]:
            # E-5 FIX: treat judge-failure sentinel as explicit 0.0 (not None).
            if result.get("_judge_failed"):
                return 0.0, result.get(detail_key, "judge_failed_after_retries")
            score = self._normalize_score(result, score_key, supported_labels=labels or set())
            detail = result.get(detail_key, "")
            if score is None:
                return None, "judge returned invalid JSON"
            return float(score), str(detail)


        return (
            _extract(cc_result,    "correctness_score",  "correctness_detail",  {"correct", "supported"}),
            _extract(cc_result,    "completeness_score", "completeness_detail"),
            _extract(faith_result, "faithfulness_score", "faithfulness_detail", {"supported"}),
        )


# ---------------------------------------------------------------------------
# Answer generator  (Gemini 3.1 Flash Lite)
# ---------------------------------------------------------------------------

class GeminiAnswerGenerator:
    """
    Drop-in replacement for AnswerGenerator that calls Gemini 3.1 Flash Lite
    instead of the local Ollama generator.
    """

    SYSTEM_PROMPT = (
        "You are a precise document question-answering assistant. "
        "Answer the question using ONLY the provided context. "
        "Be concise but complete. "
        "If the context does not contain enough information, say so."
    )

    def __init__(self, gemini_client: GeminiClient):
        self._gemini = gemini_client

    def _build_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks, start=1):
            text = (chunk.get("context_text") or chunk.get("text") or "").strip()
            if text:
                context_parts.append(f"[Chunk {i}]\n{text}")

        context_str = "\n\n".join(context_parts) if context_parts else "(no context retrieved)"

        return (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER:"
        )

    async def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        max_retries: int = 2,
    ) -> str:
        prompt = self._build_prompt(query, chunks)
        for attempt in range(1, max_retries + 1):
            answer = await self._gemini.generate(prompt, temperature=0.1, max_tokens=1024)
            if answer.strip():
                return answer.strip()
            if attempt < max_retries:
                await asyncio.sleep(2.0 * attempt)
        return ""


# ---------------------------------------------------------------------------
# Routed Hybrid Retriever wrapper
# ---------------------------------------------------------------------------

class RoutedHybridRetrieverWrapper:
    """
    Route-aware wrapper that adds hierarchical packing to the hybrid retriever.

    Architecture:
      - query complexity classification → route selection
      - hierarchical chunks (raw + parent)
      - BM25 + dense retrieval
      - late reranker (inside the underlying HybridRetriever)
      - adaptive context packing before generation
    """

    def __init__(self, hybrid_retriever, packer: Optional[Any] = None):
        self._inner  = hybrid_retriever
        self._packer = packer or (AdaptiveContextPacker() if HAS_CONTEXT_PACKER else None)
        self.last_trace: Optional[Dict[str, Any]] = None
        self.last_packed_chunks: List[Dict[str, Any]] = []
        self.last_pack_trace: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _chunk_summary(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for d in docs:
            meta = d.get("metadata", {}) or {}
            out.append({
                "chunk_id":     d.get("chunk_id", ""),
                "text":         (d.get("context_text") or d.get("text", ""))[:300],
                "score":        float(d.get("score", d.get("fused_score", d.get("rerank_score", 0.0)))),
                "rerank_score": float(d["rerank_score"]) if "rerank_score" in d else None,
                "fused_score":  float(d["fused_score"])  if "fused_score"  in d else None,
                "page":         meta.get("page_start", 0),
                "source":       meta.get("document_id", ""),
                "node_type":    meta.get("node_type", d.get("node_type", "raw")),
            })
        return out

    def route_for(self, query: str) -> str:
        complexity = _classify_retrieval_complexity(query)
        return f"routed_hybrid/{complexity}"

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

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
            "route":              f"routed_hybrid/{route_complexity}",
            "query_variants":     raw_trace.get("variants", raw_trace.get("query_variants", [query])),
            "query_features":     {
                **raw_trace.get("features", raw_trace.get("query_features", {})),
                "retrieval_complexity": route_complexity,
                "use_summary_nodes":    bool(use_summary_nodes),
                "has_context_packer":   bool(self._packer),
            },
            "embedding_candidates": raw_trace.get("embedding_candidates", []),
            "bm25_candidates":      raw_trace.get("bm25_candidates", []),
            "after_rrf":            raw_trace.get("after_rrf", []),
            "after_rerank":         raw_trace.get("after_rerank", self._chunk_summary(results)),
            "submitted_to_llm":     raw_trace.get("submitted_to_llm", []),
            "candidate_counts":     dict(raw_trace.get("candidate_counts", {})),
            "notes":                list(raw_trace.get("notes", [])),
            "stage_timings_ms":     dict(raw_trace.get("stage_timings_ms", {})),
        }
        self.last_trace["stage_timings_ms"]["total"] = round(t1 - t0, 2)
        self.last_trace["candidate_counts"]["raw"] = len(results)
        self.last_packed_chunks = list(results)
        self.last_pack_trace = {}
        return results

    # ------------------------------------------------------------------
    # Context packing
    # ------------------------------------------------------------------

    def pack_for_llm(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not chunks:
            self.last_packed_chunks = []
            self.last_pack_trace = {"packed": 0, "token_budget": 0, "mode": _classify_retrieval_complexity(query)}
            if self.last_trace is not None:
                self.last_trace["submitted_to_llm"] = []
                self.last_trace["candidate_counts"]["packed"] = 0
            return []

        mode = _classify_retrieval_complexity(query)

        if self._packer is None:
            packed       = chunks[:top_k]
            token_budget = 0
        else:
            base_budget = 1650 if mode != "simple" else 1100
            packed = self._packer.pack(
                query=query,
                chunks=chunks,
                answer_plan={
                    "retrieval_complexity": mode,
                    "mode": mode,
                    "budget": {"max_tokens_for_context": base_budget},
                },
                mode=mode,
            )
            token_budget = self._packer._target_tokens(
                {"retrieval_complexity": mode, "mode": mode, "budget": {"max_tokens_for_context": base_budget}},
                mode,
                chunks=chunks,
            )

        self.last_packed_chunks = list(packed)
        self.last_pack_trace = {"packed": len(packed), "token_budget": token_budget, "mode": mode}

        if self.last_trace is not None:
            self.last_trace["submitted_to_llm"] = self._chunk_summary(packed)
            self.last_trace["candidate_counts"]["packed"] = len(packed)
            notes = self.last_trace.setdefault("notes", [])
            notes.append(
                f"context_packer={'enabled' if self._packer else 'fallback'}; "
                f"mode={mode}; packed={len(packed)}; budget={token_budget}"
            )

        return packed


# ---------------------------------------------------------------------------
# Backend imports
# ---------------------------------------------------------------------------

def _import_stores():
    from vector_store import VectorStore
    from bm25_store import BM25Store
    return VectorStore, BM25Store


def _import_hybrid_retriever():
    from retriever import HybridRetriever
    return HybridRetriever


# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------

def ingest_document(
    pdf_path: str,
    vector_store,
    bm25_store,
) -> List[Dict[str, Any]]:
    from ingestion import DocumentIngestor
    ingestor = DocumentIngestor()
    doc_id   = Path(pdf_path).stem.replace(" ", "_").lower()
    chunks   = ingestor.parse_pdf(pdf_path, doc_id)
    if chunks:
        vector_store.add_chunks(chunks)
        bm25_store.add_chunks(chunks)
    return chunks


# ---------------------------------------------------------------------------
# Core evaluation — single architecture, all questions for one document
# ---------------------------------------------------------------------------

async def evaluate_document(
    doc: EvalDocument,
    retriever: RoutedHybridRetrieverWrapper,
    generator: GeminiAnswerGenerator,
    judge: LLMJudge,
    top_k: int = 5,
) -> List[PerQueryResult]:

    results: List[PerQueryResult] = []

    for q_idx, question in enumerate(doc.questions):
        print(f"    Q{q_idx + 1}/{len(doc.questions)}: {question.query[:70]}...")

        # --- Retrieval ---
        try:
            retrieved = retriever.retrieve(question.query, top_k=top_k)
        except Exception as exc:
            print(f"      [routed_hybrid] Retrieval error: {exc}")
            retrieved = []

        llm_chunks    = _prepare_chunks_for_llm(retriever, question.query, retrieved, top_k=top_k)
        retriever_trace = _trace_to_dict(getattr(retriever, "last_trace", None))
        retrieved_ids   = [_extract_chunk_id(d) for d in retrieved if _extract_chunk_id(d)]

        # Page-level retrieval metrics
        relevant_pages = set(question.relevant_pages)
        retrieved_pages: set          = set()
        retrieved_pages_ordered: list = []

        for d in retrieved:
            meta = d.get("metadata", {})
            page = meta.get("page_start", 0)
            if page == 0:
                cid = _extract_chunk_id(d)
                m   = re.search(r"page_(\d+)", cid)
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
                mm  = re.search(r"page_(\d+)", cid)
                if mm:
                    page = int(mm.group(1))
            if page in relevant_pages:
                page_mrr = 1.0 / i
                break

        r = len(relevant_pages)
        if r == 0:
            page_precision = 1.0
        else:
            top_r_pages = retrieved_pages_ordered[:r]
            page_precision = (
                sum(1 for p in top_r_pages if p in relevant_pages) / len(top_r_pages)
                if top_r_pages else 0.0
            )

        page_recall = (
            len(relevant_pages & retrieved_pages) / len(relevant_pages)
            if relevant_pages else 0.0
        )

        # --- Generation (Gemini 3.1 Flash Lite) ---
        answer = await generator.generate(question.query, llm_chunks)
        await asyncio.sleep(INTER_CALL_DELAY)

        # --- Judging (Gemini 3.1 Flash Lite) ---
        context_str = "\n---\n".join(
            (d.get("context_text") or d.get("text", ""))[:MAX_CONTEXT_CHARS_PER_CHUNK]
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

        results.append(PerQueryResult(
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
            # E-5 FIX: propagate judge failure flag so the output JSON marks
            # which questions had their scores injected as 0.0 vs genuinely judged.
            _judge_failed=(
                correct_detail == "judge_failed_after_retries"
                or complete_detail == "judge_failed_after_retries"
                or faith_detail == "judge_failed_after_retries"
            ),
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
        ))


    return results


# ---------------------------------------------------------------------------
# Pipeline trace display helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int = 200) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:max_len] + ("…" if len(text) > max_len else "")


def _score_bar(score: Optional[float], width: int = 20) -> str:
    if score is None:
        return "[" + " " * width + "] N/A"
    filled = round(score * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.2f}"


def _candidate_preview(c: Dict[str, Any]) -> str:
    page  = c.get("page", "?")
    score = c.get("score", c.get("fused_score", c.get("rerank_score", 0.0)))
    cid   = c.get("chunk_id", "")
    text  = _truncate(c.get("text", ""), 180)
    return f"page={page}  score={float(score):.4f}  id={cid[:48]}  {text}"


def print_detailed_per_question_report(results: List[PerQueryResult]) -> None:
    DIVIDER = "=" * 110
    print("\n" + DIVIDER)
    print("  DETAILED PER-QUESTION PIPELINE TRACE — ROUTED HYBRID (Gemini 3.1 Flash Lite)")
    print(f"  {len(results)} questions")
    print(DIVIDER)

    for q_num, r in enumerate(results, start=1):
        print(f"\n{'━' * 110}")
        print(
            f"  Q{q_num:03d}  [{r.question_id}]  |  doc: {r.document_id}  |  "
            f"type: {r.question_type}  |  difficulty: {r.difficulty}"
        )
        print(f"  QUERY         : {r.query}")
        print(f"  RELEVANT PAGES: {r.relevant_pages}")
        print(f"  GROUND TRUTH  : {_truncate(r.ground_truth, 300)}")
        print(f"{'━' * 110}")

        # Pipeline trace
        feat_str = ", ".join(k for k, v in r.query_features.items() if v and k != "word_count") or "—"
        print(f"  │ route         : {r.route or '?'}")
        print(f"  │ query variants: {' | '.join(r.query_variants) if r.query_variants else r.query}")
        print(f"  │ features      : {feat_str}")
        if r.notes:
            print(f"  │ notes         : {'; '.join(r.notes)}")

        def _print_cands(title: str, cands: List[Dict[str, Any]]):
            print(f"  │ {title}:")
            if not cands:
                print("  │   (none)")
                return
            for ci, c in enumerate(cands[:5], 1):
                print(f"  │   {ci}. {_candidate_preview(c)}")

        _print_cands("embedding candidates", r.embedding_candidates)
        _print_cands("bm25 candidates",      r.bm25_candidates)
        _print_cands("after rrf",            r.after_rrf)
        _print_cands("after rerank",         r.after_rerank)
        _print_cands("submitted to llm",     r.submitted_to_llm)

        print("  │ generated answer:")
        for line in (r.generated_answer or "(no answer)").splitlines()[:10]:
            print(f"  │   {line}")
        if len((r.generated_answer or "").splitlines()) > 10:
            print("  │   …")

        print(
            "  │ scores: "
            f"correctness={_score_bar(r.answer_correctness)}  "
            f"completeness={_score_bar(r.answer_completeness)}  "
            f"faithfulness={_score_bar(r.faithfulness)}"
        )
        if r.correctness_detail:
            print(f"  │ correctness note  : {_truncate(r.correctness_detail, 120)}")
        if r.completeness_detail:
            print(f"  │ completeness note : {_truncate(r.completeness_detail, 120)}")
        if r.faithfulness_detail:
            print(f"  │ faithfulness note : {_truncate(r.faithfulness_detail, 120)}")
        if r.stage_timings_ms:
            timing_parts = [
                f"{k}={v:.1f}ms"
                for k, v in r.stage_timings_ms.items()
                if isinstance(v, (int, float))
            ]
            print(f"  │ timings: {' | '.join(timing_parts)}")

    print(f"\n{DIVIDER}")
    print("  END OF DETAILED PIPELINE TRACE REPORT")
    print(DIVIDER + "\n")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    ("Answer Correctness",    "answer_correctness"),
    ("Answer Completeness",   "answer_completeness"),
    ("Faithfulness",          "faithfulness"),
    ("Hit Rate@5",            "hit_rate"),
    ("MRR@5",                 "mrr"),
    ("Context Precision@5",   "context_precision"),
    ("Context Recall",        "context_recall"),
]


def _fmt_ms(d: Dict[str, float]) -> str:
    if d["mean"] == 0.0 and d["std"] == 0.0:
        return "  N/A"
    return f"{d['mean']:.3f}±{d['std']:.3f}"


def compute_aggregate(results: List[PerQueryResult]) -> Dict[str, Any]:
    # E-5 FIX: include ALL results in the aggregate — judge-failed questions
    # now carry explicit 0.0 scores rather than None, so excluding None values
    # is equivalent to excluding only genuine parse errors (which should be 0).
    # We also report the failure count so the figure is transparently an
    # inclusive mean, not a best-case upper bound.
    judge_failures = sum(1 for r in results if getattr(r, "_judge_failed", False))
    agg = {
        "config": "routed_hybrid",
        "n_questions": len(results),
        "judge_failures": judge_failures,
    }
    for label, key in METRIC_KEYS:
        # Use all non-None values; after E-5 the only Nones left are genuine
        # parse errors on non-LLM metrics (e.g. context_precision edge cases).
        vals = [getattr(r, key) for r in results if getattr(r, key) is not None]
        agg[key] = mean_std(vals)
    return agg



def compute_category_breakdown(results: List[PerQueryResult]) -> Dict[str, Any]:
    buckets: Dict[str, list] = {}
    for r in results:
        buckets.setdefault(r.category, []).append(r)
    return {
        cat: mean_std_n([r.answer_correctness for r in rs if r.answer_correctness is not None])
        for cat, rs in sorted(buckets.items())
    }


def compute_question_type_breakdown(results: List[PerQueryResult]) -> Dict[str, Any]:
    buckets: Dict[str, list] = {}
    for r in results:
        buckets.setdefault(r.question_type, []).append(r)
    return {
        qt: mean_std_n([r.answer_correctness for r in rs if r.answer_correctness is not None])
        for qt, rs in sorted(buckets.items())
    }


def print_report(
    agg: Dict[str, Any],
    cat_breakdown: Dict[str, Any],
    type_breakdown: Dict[str, Any],
    total_docs: int,
    total_questions: int,
) -> None:
    W = 100
    print("\n" + "=" * W)
    print("RAG ANSWER QUALITY REPORT — ROUTED HYBRID  |  Model: Gemini 3.1 Flash Lite")
    print(f"({total_docs} documents, {total_questions} questions)")
    print("=" * W)
    print(f"{'Metric':<26}  {'routed_hybrid'}")
    print("-" * W)
    for label, key in METRIC_KEYS:
        print(f"{label:<26}  {_fmt_ms(agg[key])}")
    print("=" * W)

    print("\nPER-CATEGORY BREAKDOWN (Answer Correctness):")
    print(f"{'Category':<30}  mean±std      n")
    print("-" * W)
    for cat, d in cat_breakdown.items():
        flag = " ⚠ low-n" if d.get("low_sample_size") else ""
        print(f"{cat:<30}  {_fmt_ms(d):<14}  {d['n']}{flag}")
    print("=" * W)

    print("\nPER-QUESTION-TYPE BREAKDOWN (Answer Correctness):")
    print(f"{'Type':<24}  mean±std      n")
    print("-" * W)
    for qt, d in type_breakdown.items():
        flag = " ⚠ low-n" if d.get("low_sample_size") else ""
        print(f"{qt:<24}  {_fmt_ms(d):<14}  {d['n']}{flag}")
    print("=" * W)


# ---------------------------------------------------------------------------
# Golden QA loading
# ---------------------------------------------------------------------------

def load_golden_qa(path: Optional[str] = None) -> List[EvalDocument]:
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
):
    print("=" * 60)
    print("Routed Hybrid RAG Evaluation  —  Gemini 3.1 Flash Lite")
    print("=" * 60)

    if not skip_download:
        print("\n[Step 0] Downloading evaluation corpus...")
        try:
            from download_eval_corpus import main as download_all
            download_all()
        except Exception as exc:
            print(f"  Warning: Download failed ({exc}). Continuing with existing files.")

    print("\n[Step 1] Loading golden QA dataset...")
    documents = load_golden_qa(golden_qa_path)
    if subset:
        documents = documents[:subset]
    total_questions = sum(len(d.questions) for d in documents)
    print(f"  Loaded {len(documents)} documents, {total_questions} questions")

    print("\n[Step 2] Initialising Gemini 3.1 Flash Lite client (key pool)...")
    gemini_client = GeminiClient()
    if not gemini_client.enabled:
        print(f"  FATAL: {gemini_client._init_error}")
        print(
            "  Set GEMINI_API_KEY (and optionally GEMINI_API_KEY_2 / "
            "GEMINI_API_KEY_3) and ensure google-generativeai is installed."
        )
        return {}

    generator = GeminiAnswerGenerator(gemini_client)
    judge     = LLMJudge(gemini_client)

    print("\n[Step 3] Importing RAG backend components...")
    VectorStore, BM25Store   = _import_stores()
    HybridRetriever          = _import_hybrid_retriever()

    all_results: List[PerQueryResult] = []

    for doc_idx, doc in enumerate(documents):
        print(f"\n[Step 4.{doc_idx + 1}] Processing: {doc.filename} ({doc.category}, ~{doc.pages} pages)")

        pdf_path = EVAL_DIR / doc.filename
        if not pdf_path.exists():
            print(f"  SKIP: PDF not found at {pdf_path}")
            continue

        iso_dir = ISOLATED_DATA_DIR / doc.id
        if iso_dir.exists():
            shutil.rmtree(iso_dir)
        iso_dir.mkdir(parents=True, exist_ok=True)

        chroma_dir = str(iso_dir / "chroma")
        bm25_dir   = str(iso_dir / "bm25")

        try:
            vector_store = VectorStore(persist_dir=chroma_dir)
            bm25_store   = BM25Store(persist_dir=bm25_dir)

            print(f"  Ingesting {doc.filename}...")
            chunks = ingest_document(str(pdf_path), vector_store, bm25_store)
            print(f"  Ingested {len(chunks)} chunks")
            if not chunks:
                print(f"  SKIP: No chunks extracted from {doc.filename}")
                continue

            hybrid_ret = HybridRetriever(vector_store, bm25_store)
            retriever  = RoutedHybridRetrieverWrapper(hybrid_ret)

            print(f"  Evaluating {len(doc.questions)} questions  [architecture: routed_hybrid]")
            doc_results = await evaluate_document(doc, retriever, generator, judge, top_k=top_k)
            all_results.extend(doc_results)

            # Log rate-limit usage after each document so operators can spot
            # approaching RPD exhaustion early (500 req/day per key).
            for ks in gemini_client._pool.pool_status():
                exhausted_flag = " [EXHAUSTED — will rotate]" if ks["rpd_used"] >= ks["rpd_limit"] else ""
                print(
                    f"  [RATE/{ks['label']}] After doc {doc_idx + 1}: "
                    f"RPM {ks['rpm_used']}/{ks['rpm_limit']}  "
                    f"TPM {ks['tpm_used']:,}/{ks['tpm_limit']:,}  "
                    f"RPD {ks['rpd_used']}/{ks['rpd_limit']}"
                    f"{exhausted_flag}"
                )

        except Exception as exc:
            print(f"  ERROR processing {doc.filename}: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                if iso_dir.exists():
                    shutil.rmtree(iso_dir)
            except Exception:
                pass

    print("\n[Step 5] Computing metrics...")
    aggregate      = compute_aggregate(all_results)
    cat_breakdown  = compute_category_breakdown(all_results)
    type_breakdown = compute_question_type_breakdown(all_results)

    questions_per_document: Dict[str, int] = {}
    for r in all_results:
        questions_per_document[r.document_id] = questions_per_document.get(r.document_id, 0) + 1

    print_report(aggregate, cat_breakdown, type_breakdown, len(documents), total_questions)
    print_detailed_per_question_report(all_results)

    if output_path:
        output = {
            "metadata": {
                "architecture":           "routed_hybrid",
                "llm_model":              GEMINI_MODEL,
                "rate_limits": {
                    "rpm":  RATE_LIMIT_RPM,
                    "tpm":  RATE_LIMIT_TPM,
                    "rpd":  RATE_LIMIT_RPD,
                },
                "rate_limit_usage":       gemini_client._pool.pool_status(),
                "total_documents":        len(documents),
                "total_questions":        total_questions,
                "timestamp":              time.strftime("%Y-%m-%dT%H:%M:%S"),
                "questions_per_document": questions_per_document,
                "judge_parse_stats":      judge.parse_stats,
            },
            "aggregate":              aggregate,
            "category_breakdown":     cat_breakdown,
            "question_type_breakdown": type_breakdown,
            "per_query": [asdict(r) for r in all_results],
        }
        Path(output_path).write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote results to {output_path}")

    return aggregate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Routed Hybrid RAG evaluation — Gemini 3.1 Flash Lite"
    )
    parser.add_argument(
        "--output", type=str, default="eval_results_routed_hybrid_gemini.json",
        help="Output JSON path",
    )
    parser.add_argument("--top-k",        type=int,  default=5,    help="top_k for retrieval")
    parser.add_argument("--subset",       type=int,  default=None, help="Only evaluate first N documents")
    parser.add_argument("--skip-download", action="store_true",    help="Skip PDF download step")
    parser.add_argument("--golden-qa",    type=str,  default=None, help="Path to golden QA JSON")
    args = parser.parse_args()

    asyncio.run(main(
        output_path=args.output,
        top_k=args.top_k,
        subset=args.subset,
        skip_download=args.skip_download,
        golden_qa_path=args.golden_qa,
    ))