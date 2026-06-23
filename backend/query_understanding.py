import os
import json
import re
from typing import Dict, Any
from groq import AsyncGroq
from logger import logger
from display_agent import DisplayFormattingAgent

# ---------------------------------------------------------------------------
# Three-way retrieval complexity classification
# ---------------------------------------------------------------------------
# This is consumed specifically by the doc-RAG retriever:
#
#   "simple"     -> a direct fact lookup that can be answered from a small,
#                   local evidence window.
#   "multi_hop"  -> asks for comparisons, exceptions, negation, legal
#                   interpretation, or connecting multiple pieces of evidence.
#   "global"     -> asks about the document/corpus as a whole.
#
# The previous revision accidentally injected literal backspace characters in
# these regexes, which broke the word-boundary checks. This version restores
# proper boundaries and adds a few conservative legal/clause heuristics so
# short but semantically hard constitutional questions no longer fall into the
# fragile fast path.

_GLOBAL_PATTERNS = re.compile(
    r"\b(summari[sz]e|overview|main (themes?|points?|ideas?|topics?)|"
    r"what is this (document|file|pdf|text) about|"
    r"key takeaways?|tl;?dr|in general|overall|"
    r"what does (this|the) (document|file|pdf|report) (say|cover|discuss))\b",
    re.IGNORECASE,
)

_MULTI_HOP_PATTERNS = re.compile(
    r"\b(compare|comparison|difference between|relationship between|"
    r"how does .+ (affect|impact|relate to|compare to)|"
    r"both .+ and|across (multiple|all|several)|"
    r"first .+ then|before .+ after)\b",
    re.IGNORECASE,
)

# FIX (#1): _NEGATION_PATTERNS, _LEGAL_PATTERNS, _LEGAL_AMBIGUOUS_TERMS, and
# _TECHNICAL_DOMAIN_PATTERNS used to be defined here AND independently in
# retriever.py as byte-for-byte duplicates (only the variable names
# differed), with nothing keeping the two copies in sync. Now imported from
# a single shared module -- see domain_patterns.py for the full rationale
# and the rfc_q1 regression this guard exists to prevent.
from domain_patterns import (
    NEGATION_RE as _NEGATION_PATTERNS,
    LEGAL_OR_POLARITY_RE as _LEGAL_PATTERNS,
    LEGAL_AMBIGUOUS_RE as _LEGAL_AMBIGUOUS_TERMS,
    TECHNICAL_DOMAIN_RE as _TECHNICAL_DOMAIN_PATTERNS,
    has_negation_signal as _has_negation_signal,
)


class QueryUnderstandingEngine:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"

    def _normalize_query(self, query: str) -> str:
        q = query.strip().lower()
        q = re.sub(r"\s+", " ", q)
        return q

    def _is_legal_query(self, q: str) -> bool:
        """True if this query is asking about constitutional/legal clause
        content specifically (not a technical-spec lookup that happens to
        share vocabulary like "section" or "court" with legal text)."""
        if _LEGAL_PATTERNS.search(q):
            return True
        if _LEGAL_AMBIGUOUS_TERMS.search(q):
            # "section", "court", "treaty", etc. are ambiguous on their own.
            # Only treat as legal if there's no competing technical signal.
            return not _TECHNICAL_DOMAIN_PATTERNS.search(q)
        return False

    def classify_retrieval_complexity(self, query_lower: str) -> str:
        """Fast heuristic 3-way classification used to gate doc-RAG retrieval."""
        q = query_lower.strip().lower()

        if _GLOBAL_PATTERNS.search(q):
            return "global"
        if _MULTI_HOP_PATTERNS.search(q):
            return "multi_hop"

        # Negation and legal interpretation are high-risk for the fast path.
        if _has_negation_signal(q) or self._is_legal_query(q):
            return "multi_hop"

        # Genuine multi-clause structure: the query asks two distinct things
        # ("...and what...", "...and how...") or contains multiple question
        # marks. A single incidental "and"/"or" inside an otherwise
        # single-fact question does not by itself indicate multi-hop
        # reasoning, and sheer length is not a complexity signal either —
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

        if self._is_simple_query(q):
            return "simple"

        # Nothing above flagged this as global, negated, legal/ambiguous, or
        # genuinely multi-clause — it's a single, direct lookup regardless
        # of how many words it took to phrase precisely.
        if not self._is_legal_query(q):
            return "simple"
        return "multi_hop"

    def _is_simple_query(self, query_lower: str) -> bool:
        """
        Simple lookups are only the direct, unambiguous ones.

        Examples:
        - "what is photosynthesis"
        - "who is alan turing"
        - "define entropy"
        """
        q = query_lower.strip()
        if not q:
            return False

        # Anything that looks like a clause/exceptions/legal-style question is not simple.
        if _has_negation_signal(q) or self._is_legal_query(q):
            return False

        if q.startswith(("how long", "how many", "which branch", "which amendment", "which article", "what power", "what right")):
            return False

        words = q.split()
        if len(words) > 8:
            return False

        return any(
            q.startswith(prefix)
            for prefix in ("what is ", "who is ", "when is ", "where is ", "define ", "what does ")
        )

    def _deterministic_understand(self, original_query: str, normalized_query: str) -> Dict[str, Any]:
        logger.info("Using Deterministic Understanding.")

        concept = normalized_query
        for prefix in ["what is a", "what is an", "what is", "who is", "when is", "where is", "define ", "what does"]:
            if normalized_query.startswith(prefix):
                concept = normalized_query[len(prefix):].strip()
                break

        intent = "definition" if ("what is" in normalized_query or "define" in normalized_query) else "simple_fact"

        return {
            "original_query": original_query,
            "normalized_query": normalized_query,
            "intent": intent,
            "concepts": [concept.strip("?")],
            "required_sections": [intent],
            "mode": "direct_web",
            "is_complex": False,
            "needs_clarification": False,
        }

    async def _llm_understand(self, original_query: str, normalized_query: str) -> Dict[str, Any]:
        logger.info("Using LLM Planner for Query Understanding.")
        system_prompt = f"""You are a query analysis engine. Your job is to decompose the user's query into a structured plan for search and answer generation.

Rules:
1. Classify the intent: 'comparison', 'research', 'multi_hop', 'troubleshooting', 'ambiguous'.
2. Identify the core concepts (entities/topics).
3. Determine required sections for a complete answer.
4. If the query is highly ambiguous and requires clarification from the user, set 'needs_clarification' to true.
5. Return ONLY a JSON object matching this schema:
{{
    "intent": "string",
    "concepts": ["string"],
    "required_sections": ["string"],
    "needs_clarification": boolean
}}"""
        try:
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": original_query},
                ],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            parsed = json.loads(response.choices[0].message.content)
            return {
                "original_query": original_query,
                "normalized_query": normalized_query,
                "intent": parsed.get("intent", "research"),
                "concepts": parsed.get("concepts", []),
                "required_sections": parsed.get("required_sections", []),
                "mode": "clarify" if parsed.get("needs_clarification") else "web_rag",
                "is_complex": True,
                "needs_clarification": parsed.get("needs_clarification", False),
            }
        except Exception as exc:
            logger.error(f"Error in LLM understanding: {exc}")
            return self._deterministic_understand(original_query, normalized_query)

    async def understand(self, query: str, norm_q: str, has_documents: bool = False) -> Dict[str, Any]:
        """Main entry point."""
        explicit_doc = bool(re.search(r'\b(document|file|text|pdf|upload|this|it)\b', norm_q, flags=re.IGNORECASE))
        if has_documents and explicit_doc:
            result = self._deterministic_understand(query, norm_q)
            result["mode"] = "doc_rag"
            result["intent"] = "document_lookup"
            result["retrieval_complexity"] = self.classify_retrieval_complexity(norm_q)
            return result

        if self._is_simple_query(norm_q):
            result = self._deterministic_understand(query, norm_q)
        else:
            result = await self._llm_understand(query, norm_q)

        result["retrieval_complexity"] = self.classify_retrieval_complexity(norm_q)
        return result
