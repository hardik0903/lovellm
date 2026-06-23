"""
DocumentAgent — Observe → Retrieve → Answer → Verify → Clarify loop.

Old:  retrieve chunks → one-shot Groq call → return answer.

New loop:
  Step 1 (Observe)   — detect intent (qa vs summarise vs compare) and
                       identify which specific section/page the user is asking about
  Step 2 (Retrieve)  — adaptive retrieval: start with top-5; if confidence is
                       low after step 3, retrieve 5 more chunks and try again
  Step 3 (Answer)    — produce a grounded answer with evidence quotes
  Step 4 (Reflect)   — self-verify: does the answer actually address the query?
                       does it contradict the retrieved evidence?
                       if verification fails → expand retrieval and retry (once)
"""

import os
import json
from typing import Dict, Any, AsyncGenerator, List
from groq import AsyncGroq
from logger import logger
from agent_base import BaseAgent, AgentMemory, AgentStep
from document_detector import DocumentDetector
from document_classifier import DocumentClassifier


class DocumentAgent(BaseAgent):
    def __init__(self):
        self.detector = DocumentDetector()
        self.classifier = DocumentClassifier()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.fast_model = "llama-3.1-8b-instant"
        self.strong_model = "llama-3.3-70b-versatile"

    def detect(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.detector.detect(query, context)

    async def classify(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return await self.classifier.classify(query)

    # ------------------------------------------------------------------ #
    # Internal tools                                                      #
    # ------------------------------------------------------------------ #

    def _format_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        ctx = ""
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            page = meta.get("page_start", "?")
            text = chunk.get("context_text", chunk.get("text", ""))
            ctx += f"\n--- [Page {page}] ---\n{text}\n"
        return ctx

    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen, sources = set(), []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            src = str(meta.get("source_file") or chunk.get("document_id", "unknown"))
            if src not in seen:
                seen.add(src)
                sources.append({"title": src, "url": src, "type": "document"})
        return sources

    async def _answer(
        self, query: str, intent: str, classification: Dict[str, Any],
        context_str: str, memory: AgentMemory
    ) -> Dict[str, Any]:
        """Act: produce grounded answer with evidence."""
        prior = memory.get_context_summary()

        if intent == "qa":
            schema = {
                "intent": "qa",
                "answer": "direct answer grounded in the document",
                "evidence": [
                    {"quote": "exact passage", "page": 1, "relevance_score": 0.9}
                ],
                "confidence": 0.0,
                "answer_location": "Section X, Page Y",
                "caveat": "note if the document doesn't fully answer the question",
                "unanswered_aspect": "any part of the query NOT answered",
            }
        else:
            schema = {
                "intent": "summarise",
                "summary": "the summary text",
                "key_points": ["point 1"],
                "document_type": "type of document",
                "sections_covered": ["section 1"],
                "compression_ratio": "original / summary word count",
            }

        prompt = f"""You are a precise Document Analyst. Answer ONLY from the provided context.
Prior reasoning:
{prior}

Task: {intent}
Query: "{query}"

Document context:
{context_str}

Return ONLY valid JSON matching:
{json.dumps(schema, indent=2)}

If the answer is not in the context, set confidence to 0.0 and explain in 'caveat'.
"""
        resp = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.strong_model,
            response_format={"type": "json_object"},
            stream=True,
            temperature=0.0,
        )
        buf = ""
        async for chunk in resp:
            c = chunk.choices[0].delta.content
            if c:
                buf += c
        return json.loads(buf)

    async def _verify_answer(
        self, query: str, answer_obj: Dict[str, Any],
        context_str: str, memory: AgentMemory
    ) -> Dict[str, Any]:
        """
        Reflect: independent fast-model pass to check the answer is grounded
        and does not contradict the evidence.
        Returns {ok: bool, issue: str or None, confidence_adjustment: float}
        """
        prior = memory.get_context_summary()
        answer_text = answer_obj.get("answer") or answer_obj.get("summary") or ""
        evidence = json.dumps(answer_obj.get("evidence", []))

        prompt = f"""You are an answer verifier. Check whether the answer is supported by the evidence.
Prior reasoning:
{prior}

Query: "{query}"
Answer: "{answer_text}"
Evidence cited: {evidence}

Context (ground truth):
{context_str[:3000]}

Return JSON:
{{
  "ok": true/false,
  "grounded": true/false,
  "contradicts_evidence": true/false,
  "issue": "describe the problem if ok is false, else null",
  "confidence_adjustment": 0.0   // +0.1 if more confident, -0.2 if less, etc.
}}
"""
        resp = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.fast_model,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)

    # ------------------------------------------------------------------ #
    # Main agent loop                                                     #
    # ------------------------------------------------------------------ #

    async def solve(self, query: str, context: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        memory = AgentMemory(query)
        agent_name = "document"

        try:
            # ================================================================
            # OBSERVE
            # ================================================================
            yield self._thinking_event(agent_name, "Observing document query…", step=0)
            local_retriever = (context or {}).get("local_retriever")
            classification = await self.classify(query)
            intent = classification.get("intent", "qa")

            # ================================================================
            # STEP 1 — RETRIEVE (initial pass: top-5)
            # ================================================================
            yield self._thinking_event(agent_name, "Retrieving relevant document chunks…", step=1)

            intent_norm = (intent or "").lower()
            retrieval_complexity = "simple" if intent_norm == "qa" else "global" if intent_norm in {"summarize", "summarise", "outline"} else "multi_hop"
            if local_retriever:
                chunks = local_retriever.retrieve(
                    query,
                    top_k=5,
                    complexity=retrieval_complexity,
                    use_summary_nodes=(intent != "qa"),
                )
            else:
                chunks = (context or {}).get("context_chunks", [])[:5]

            context_str = self._format_chunks(chunks)
            sources = self._extract_sources(chunks)

            step1 = AgentStep(
                step_num=1,
                thought="Retrieve the most relevant chunks for this query.",
                action="retrieve",
                observation=f"Retrieved {len(chunks)} chunks covering {len(sources)} source(s).",
                result={"chunks": len(chunks), "sources": len(sources)},
            )
            memory.add_step(step1)
            yield self._step_event(agent_name, step1)

            if not chunks:
                yield self._final_event(
                    mode=agent_name,
                    answer="No relevant content found in the uploaded document for this query.",
                    sources=[],
                    confidence="low",
                    memory=memory,
                )
                return

            # ================================================================
            # STEP 2 — ACT: Answer
            # ================================================================
            yield self._thinking_event(agent_name, f"Generating grounded {intent} from document…", step=2)
            answer_obj = await self._answer(query, intent, classification, context_str, memory)

            raw_answer = answer_obj.get("answer") or answer_obj.get("summary") or ""
            answer_confidence = float(answer_obj.get("confidence", 0.5))

            step2 = AgentStep(
                step_num=2,
                thought="Generate a grounded answer using only the retrieved document evidence.",
                action="answer",
                observation=f"Answer produced. Confidence: {answer_confidence:.2f}. "
                            f"Evidence cited: {len(answer_obj.get('evidence', []))}.",
                result={"confidence": answer_confidence, "has_caveat": bool(answer_obj.get("caveat"))},
            )
            memory.add_step(step2)
            yield self._step_event(agent_name, step2)

            # ================================================================
            # STEP 3 — REFLECT: Verify answer
            # ================================================================
            yield self._thinking_event(agent_name, "Verifying answer is grounded in evidence…", step=3)
            verification = await self._verify_answer(query, answer_obj, context_str, memory)

            step3 = AgentStep(
                step_num=3,
                thought="Independent check: does the answer accurately reflect the document?",
                action="verify",
                observation=(
                    "Verification passed." if verification.get("ok")
                    else f"Verification failed: {verification.get('issue')}. Expanding retrieval."
                ),
                result=verification,
            )
            memory.add_step(step3)
            yield self._step_event(agent_name, step3)
            yield self._reflection_event(
                agent_name,
                "stop" if verification.get("ok") else "expand_retrieval",
                step3.observation,
            )

            # ================================================================
            # STEP 4 (conditional) — Expand retrieval and retry once
            # ================================================================
            if not verification.get("ok") and local_retriever:
                yield self._thinking_event(agent_name, "Expanding retrieval to top-10 and retrying…", step=4)
                chunks = local_retriever.retrieve(query, top_k=10)
                context_str = self._format_chunks(chunks)
                sources = self._extract_sources(chunks)
                answer_obj = await self._answer(query, intent, classification, context_str, memory)
                raw_answer = answer_obj.get("answer") or answer_obj.get("summary") or ""

                step4 = AgentStep(
                    step_num=4,
                    thought="Expand retrieval to top-10 chunks and retry the answer generation.",
                    action="expand_and_retry",
                    observation=f"Retry with {len(chunks)} chunks. New answer length: {len(raw_answer)} chars.",
                    result={"chunks_expanded": len(chunks)},
                )
                memory.add_step(step4)
                yield self._step_event(agent_name, step4)

            # ================================================================
            # Final answer
            # ================================================================
            if raw_answer and raw_answer not in ("Direct answer to the question", "The summary text"):
                answer_text = f"Based on the document: {raw_answer}"
            else:
                evidence = answer_obj.get("evidence", [])
                if evidence and evidence[0].get("quote"):
                    answer_text = f"Based on the document: {evidence[0]['quote']}"
                else:
                    answer_text = "Based on the document: No relevant information found for your query."

            yield {"event": "delta", "data": json.dumps({"text": answer_text})}
            yield self._final_event(
                mode=agent_name,
                answer=answer_text,
                display=answer_obj,
                sources=sources or [{"title": "uploaded document", "url": "uploaded document", "type": "document"}],
                memory=memory,
            )

        except Exception as e:
            logger.error(f"[DocumentAgent] Unhandled error: {e}")
            yield self._error_event("document", str(e))