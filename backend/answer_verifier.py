import os
import re
from typing import Dict, Any, List, Optional
from groq import AsyncGroq
from logger import logger


# ---------------------------------------------------------------------------
# Self-RAG faithfulness scoring constants
# ---------------------------------------------------------------------------

# Score below which we trigger a second retrieval pass
FAITHFULNESS_RETRY_THRESHOLD = 0.45

# Maximum number of re-retrieval attempts before accepting whatever we have
MAX_SELF_RAG_RETRIES = 1

# Minimum answer length (words) before we bother scoring faithfulness
# (very short answers like "I don't know" aren't worth re-retrieving for)
MIN_ANSWER_WORDS_FOR_SELF_RAG = 10


class AnswerVerifier:
    """
    Post-generation verification that combines:

    1. Deterministic heuristic checks (coverage, citation presence,
       completeness, concept relevance) — unchanged from v1.

    2. Self-RAG faithfulness loop (new in v2): before accepting an answer,
       we score how well it is grounded in the *retrieved context* using a
       fast LLM judge call. If the score is below FAITHFULNESS_RETRY_THRESHOLD
       we call the optional `retriever` and `generator` callbacks to fetch
       fresh context with an expanded/rewritten query, regenerate, and score
       again. This closes the loop that was open in v1 (AnswerVerifier had no
       re-retrieval path).

    The verifier is backward-compatible: if constructed without `retriever` /
    `generator` callbacks (or if `retrieved_chunks` is not supplied to
    `verify()`), it behaves exactly like v1 — heuristic checks only, no retry.
    """

    def __init__(
        self,
        retriever=None,
        generator=None,
    ):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"

        # Optional callbacks for the Self-RAG loop.
        # retriever: callable(query: str, top_k: int) -> List[Dict]
        # generator: callable(query: str, chunks: List[Dict], mode: str)
        #            must be an async generator yielding {"event": ..., "data": ...}
        self._retriever = retriever
        self._generator = generator

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def verify(
        self,
        answer_obj: Dict[str, Any],
        answer_plan: Dict[str, Any],
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Verify answer_obj and (if context/callbacks available) run the
        Self-RAG faithfulness loop.

        Parameters
        ----------
        answer_obj      : dict returned by the generator (must have "answer" key)
        answer_plan     : planning metadata (original_query, required_sections, …)
        retrieved_chunks: the chunks that were passed to the generator; needed
                          for faithfulness scoring. If omitted, faithfulness
                          scoring is skipped.
        """
        original_query = answer_plan.get("original_query", "")

        # --- Self-RAG loop (only for doc_rag and only when we have context) ---
        if (
            answer_plan.get("mode") == "doc_rag"
            and retrieved_chunks
            and self._retriever is not None
            and self._generator is not None
        ):
            answer_obj = await self._self_rag_loop(
                answer_obj, original_query, retrieved_chunks
            )

        # --- Heuristic verification (unchanged from v1) ---
        answer_obj = await self._heuristic_verify(answer_obj, answer_plan)
        return answer_obj

    # ------------------------------------------------------------------
    # Self-RAG faithfulness loop
    # ------------------------------------------------------------------

    async def _self_rag_loop(
        self,
        answer_obj: Dict[str, Any],
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Score answer faithfulness against retrieved chunks.
        If below threshold, rewrite query, re-retrieve, regenerate, and score again.
        """
        answer_text = answer_obj.get("answer", "")
        if len(answer_text.split()) < MIN_ANSWER_WORDS_FOR_SELF_RAG:
            logger.info("[Self-RAG] Answer too short to score faithfulness — skipping loop.")
            return answer_obj

        for attempt in range(MAX_SELF_RAG_RETRIES + 1):
            score, rationale = await self._score_faithfulness(query, answer_text, chunks)
            logger.info(
                f"[Self-RAG] Faithfulness score={score:.2f} (attempt {attempt + 1}/"
                f"{MAX_SELF_RAG_RETRIES + 1}). Threshold={FAITHFULNESS_RETRY_THRESHOLD}"
            )
            answer_obj.setdefault("self_rag", {})
            answer_obj["self_rag"][f"faithfulness_attempt_{attempt + 1}"] = {
                "score": score,
                "rationale": rationale,
            }

            if score >= FAITHFULNESS_RETRY_THRESHOLD:
                answer_obj["self_rag"]["accepted_at_attempt"] = attempt + 1
                answer_obj["faithfulness"] = score
                break

            if attempt >= MAX_SELF_RAG_RETRIES:
                # Accept the current answer even though it's below threshold
                logger.info(
                    "[Self-RAG] Max retries reached. Accepting best available answer."
                )
                answer_obj["self_rag"]["accepted_at_attempt"] = attempt + 1
                answer_obj["faithfulness"] = score
                answer_obj["self_rag"]["low_faithfulness_flag"] = True
                break

            # --- Re-retrieve with an expanded query ---
            rewritten = self._rewrite_query(query, rationale)
            logger.info(f"[Self-RAG] Re-retrieving with rewritten query: {rewritten!r}")
            try:
                fresh_chunks = self._retriever(rewritten, top_k=7)
                if not fresh_chunks:
                    logger.warning("[Self-RAG] Re-retrieval returned no chunks. Aborting loop.")
                    answer_obj["faithfulness"] = score
                    break

                # Regenerate from fresh context
                new_answer_text = ""
                async for event in self._generator(query, fresh_chunks, mode="doc_rag"):
                    if event.get("event") == "final":
                        try:
                            import json as _json
                            data = _json.loads(event["data"])
                            new_answer_text = data.get("answer", "")
                            # Carry over sources etc. from the new generation
                            answer_obj.update({
                                k: v for k, v in data.items()
                                if k not in ("answer", "self_rag", "verification")
                            })
                        except Exception:
                            pass

                if new_answer_text:
                    answer_text = new_answer_text
                    answer_obj["answer"] = answer_text
                    chunks = fresh_chunks
                    answer_obj["self_rag"]["retrieval_pass"] = attempt + 2
                else:
                    logger.warning("[Self-RAG] Regeneration produced empty answer. Aborting loop.")
                    answer_obj["faithfulness"] = score
                    break

            except Exception as exc:
                logger.error(f"[Self-RAG] Re-retrieval/regeneration error: {exc}")
                answer_obj["faithfulness"] = score
                break

        return answer_obj

    async def _score_faithfulness(
        self,
        query: str,
        answer: str,
        chunks: List[Dict[str, Any]],
    ) -> tuple:
        """
        Ask the LLM judge: is this answer grounded in the retrieved context?
        Returns (score: float 0–1, rationale: str).
        """
        # Build a compact context block (cap at ~2000 chars to stay within tokens)
        context_parts = []
        total_chars = 0
        for c in chunks:
            text = c.get("context_text") or c.get("text", "")
            if not text:
                continue
            snippet = text[:400]
            total_chars += len(snippet)
            context_parts.append(snippet)
            if total_chars > 2000:
                break

        context_block = "\n---\n".join(context_parts) if context_parts else "(no context)"

        prompt = f"""You are a strict RAG faithfulness judge.

Query: {query}

Retrieved context (source documents):
{context_block}

Generated answer:
{answer}

Task: Score how well the answer is grounded in the retrieved context.
A score of 1.0 means every claim in the answer is directly supported by the context.
A score of 0.0 means the answer contains claims with no basis in the context.

Respond in JSON only, no preamble:
{{"score": <float 0.0-1.0>, "rationale": "<one sentence>"}}"""

        try:
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a faithfulness scoring assistant. Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=120,
            )
            import json as _json
            raw = response.choices[0].message.content.strip()
            parsed = _json.loads(raw)
            score = float(parsed.get("score", 0.5))
            rationale = parsed.get("rationale", "")
            return max(0.0, min(1.0, score)), rationale
        except Exception as e:
            logger.error(f"[Self-RAG] Faithfulness scoring error: {e}")
            return 0.5, "scoring unavailable"

    @staticmethod
    def _rewrite_query(original_query: str, low_faithfulness_rationale: str) -> str:
        """
        Produce a richer search query for the second retrieval pass.
        Strategy: append the rationale's key noun phrases to widen coverage,
        without adding LLM latency (deterministic rewrite).
        """
        # Extract capitalized noun chunks from the rationale as expansion terms
        extra_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', low_faithfulness_rationale)
        # Deduplicate and limit
        seen = set()
        unique = []
        for t in extra_terms:
            tl = t.lower()
            if tl not in seen and tl not in original_query.lower():
                seen.add(tl)
                unique.append(t)
                if len(unique) >= 3:
                    break

        if unique:
            return f"{original_query} {' '.join(unique)}"
        # Fallback: strip question words and broaden
        broadened = re.sub(r'^(what|who|when|where|why|how|is|are|does|do)\s+', '', original_query, flags=re.IGNORECASE)
        return broadened.strip() or original_query

    # ------------------------------------------------------------------
    # Heuristic verification (v1, preserved exactly)
    # ------------------------------------------------------------------

    async def _heuristic_verify(
        self, answer_obj: Dict[str, Any], answer_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        answer_text = answer_obj.get("answer", "").lower()

        # 1. Coverage Check
        sections = answer_plan.get("required_sections", [])
        missing_sections = []
        for section in sections:
            keywords = section.lower().split()
            if not any(kw in answer_text for kw in keywords if len(kw) > 3):
                missing_sections.append(section)

        # 2. Citation Coverage
        sources = answer_obj.get("sources", [])
        has_citations = False
        if answer_plan.get("mode") == "direct_web":
            has_citations = True
        elif not sources:
            has_citations = True
        else:
            if re.search(r'\[\d+\]', answer_text):
                has_citations = True
            else:
                for src in sources:
                    title = src.get("title", "").lower()
                    url = src.get("url", "").lower()
                    if title and len(title) > 5 and title in answer_text:
                        has_citations = True
                        break
                    domain = (
                        url.replace("https://", "").replace("http://", "").split("/")[0]
                    )
                    if domain and len(domain) > 3 and domain in answer_text:
                        has_citations = True
                        break

        # 3. Completeness
        is_complete = True
        if len(answer_text.split()) < 20 and len(sections) > 2:
            is_complete = False

        # 4. Relevance
        original_query = answer_plan.get("original_query", "").lower()
        concept = original_query
        for prefix in ["what is a ", "what is an ", "what is ", "who is ", "define "]:
            if original_query.startswith(prefix):
                concept = original_query[len(prefix):].strip("? ")
                break

        is_relevant = True
        concept_stem = concept if concept else ""
        if concept_stem:
            if concept_stem.endswith("es") and not concept_stem.endswith("ses"):
                concept_stem = concept_stem[:-2]
            elif concept_stem.endswith("s") and not concept_stem.endswith("ss"):
                concept_stem = concept_stem[:-1]
        if concept_stem and len(concept_stem) >= 3 and concept_stem not in answer_text:
            is_relevant = False

        verification_result = {
            "passed": (
                len(missing_sections) == 0
                and (has_citations or len(sources) == 0)
                and is_complete
                and is_relevant
            ),
            "missing_sections": missing_sections,
            "has_citations": has_citations,
            "is_complete": is_complete,
            "is_relevant": is_relevant,
        }

        # LLM fallback if deterministic failed
        if not verification_result["passed"] and answer_text:
            logger.info("Deterministic verification failed, falling back to LLM verifier pass.")
            try:
                response = await self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an answer verification assistant. Answer strictly with YES or NO.",
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Does the following answer provide a reasonable and relevant response "
                                f"to the query: '{original_query}'?\n\nAnswer: {answer_text}"
                            ),
                        },
                    ],
                    model=self.model,
                    temperature=0.0,
                    max_tokens=5,
                )
                if "yes" in response.choices[0].message.content.lower():
                    logger.info("LLM verifier approved the answer.")
                    verification_result["passed"] = True
                    verification_result["llm_override"] = True
            except Exception as e:
                logger.error(f"Error in LLM verification pass: {e}")

        answer_obj["verification"] = verification_result
        return answer_obj