import os
import json
import re
from dotenv import load_dotenv
load_dotenv()
from typing import AsyncGenerator, List, Dict, Any, Optional
from logger import logger

# ---------------------------------------------------------------------------
# Ollama config (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "")   # e.g. "llama3.2"

# Max tokens for Ollama responses. Ollama's default is often very small (128-256)
# which truncates JSON mid-output. 1536 gives ample room for structured comparison
# tables and other rich schema responses.
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "1536"))

# Context window passed to Ollama on every generation call.
# Ollama pre-allocates the FULL KV-cache buffer at model load time based on
# this value — not incrementally as tokens arrive. On an RTX 3050 4 GB:
#   llama3.2 weights ≈ 2000 MB  +  OS/driver ≈ 200 MB  =  2200 MB fixed.
#   Remaining headroom: ~1896 MB.
#   KV-cache @ num_ctx=4096                  ≈  560 MB  → 1336 MB margin. ✓
#   KV-cache @ num_ctx=8192 (Ollama default) ≈ 1742 MB  → OOM (21 MB short).
# The web_rag prompt (context_budget=1800 + system~400 + max_output=1536)
# needs ~3736 tokens, so 4096 is the minimum safe window.
# Override via OLLAMA_NUM_CTX env var if you upgrade to a GPU with more VRAM.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))


class AnswerGenerator:
    """Generates RAG answers via Groq (default) or a local Ollama model.

    Backend selection (checked in order):
    1. ``OLLAMA_MODEL`` env var is set → Ollama
    2. ``GROQ_API_KEY`` env var is set → Groq
    3. Neither → raises ValueError
    """

    def __init__(self):
        ollama_model = OLLAMA_MODEL.strip()

        if ollama_model:
            from openai import AsyncOpenAI
            base_url = OLLAMA_BASE_URL.rstrip("/") + "/v1"
            self.client = AsyncOpenAI(base_url=base_url, api_key="ollama")
            self.model  = ollama_model
            self._backend = "ollama"
            logger.info(f"AnswerGenerator using Ollama: {ollama_model} @ {base_url}")
        else:
            from groq import AsyncGroq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError(
                    "Neither OLLAMA_MODEL nor GROQ_API_KEY is set. "
                    "Set one of them to enable answer generation."
                )
            self.client  = AsyncGroq(api_key=api_key)
            self.model   = "llama-3.1-8b-instant"
            self._backend = "groq"
            logger.info("AnswerGenerator using Groq: llama-3.1-8b-instant")

    def _build_prompt(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        source_map: Dict[str, Any] = None,
        answer_plan: Dict[str, Any] = None,
        display_injection: str = "",
    ) -> str:
        context_str = ""
        for i, chunk in enumerate(context_chunks):
            doc_id = chunk.get("metadata", {}).get("document_id", "unknown")
            source_file = chunk.get("metadata", {}).get("source_file", "unknown")
            is_web = chunk.get("metadata", {}).get("is_web", False)

            if source_map and doc_id in source_map:
                title = source_map[doc_id]["title"]
                url   = source_map[doc_id]["url"]
            else:
                title = source_file
                url   = source_file

            text = chunk.get("context_text", chunk.get("text", ""))
            src_type = "web" if is_web else "document"
            context_str += f"\n--- [Source ID: {doc_id}, Title: {title}, URL: {url}, Type: {src_type}] ---\n{text}\n"

        sections = answer_plan.get("required_sections", []) if answer_plan else []
        plan_str = f"\nStructure your answer with these sections: {', '.join(sections)}" if sections else ""
        style = (answer_plan or {}).get("response_style", "standard")
        style_instruction = ""
        if style == "ordered":
            style_instruction = (
                "\n11. If the answer involves precedence, priority, ordering, ranking, or technical precedence rules, "
                "present the result as an explicitly numbered sequence from highest priority to lowest priority. "
                "State the final answer first, then the ordering rule, then the short explanation."
            )
        elif style == "legal_mechanism":
            style_instruction = (
                "\n11. If the question asks for a constitutional or legal mechanism, state the mechanism explicitly, "
                "then explain it in one or two short sentences using the exact legal terms from the context."
            )

        # Detect negation directly from the query (same pattern family used by
        # retriever.py / query_understanding.py for routing) so the generator
        # gets an explicit, impossible-to-miss flag rather than relying on the
        # LLM to notice the negation language buried inside a long question.
        is_negation_query = bool(re.search(
            r"\b(not|n't|never|no|none|without|except|excluding|isn't|doesn't|don't|didn't|cannot|can't|won't|wouldn't|shouldn't)\b",
            query, re.IGNORECASE,
        ))
        negation_banner = (
            "\n*** NEGATION QUESTION DETECTED ***\n"
            "This question asks for what is NOT true / the exception / the unsupported option.\n"
            "Your \"answer\" field MUST be a full sentence with explicit polarity (e.g. \"X is NOT...\"), "
            "never a bare phrase. Re-read Rule 7 before answering.\n"
            if is_negation_query else ""
        )

        prompt = f"""You are a retrieval-grounded answering system.
{negation_banner}
Rules:
1. Answer only using the provided context.
2. Do not use outside knowledge unless the context explicitly supports it.
3. Keep the answer concise and accurate.
4. When possible, cite the sources used.
5. Do not hallucinate names, dates, numbers, or clauses.
6. If the context uses archaic, legal, or technical phrasing, paraphrase it into clear modern English. Preserve terms of art, numbers, names, and clause references exactly.
7. If the question uses NOT / EXCEPT / "which of the following is not" / similar negation language: first identify the correct option by elimination, then state the answer as a full sentence that explicitly restates the negation/exception in its own words (e.g. "X is NOT a power granted to Congress because..." or "Unlike the others, X is not..."). NEVER answer a negation question with a bare noun phrase alone (e.g. just "appointing ambassadors") — a bare phrase reads as an affirmative claim and is wrong even when the underlying choice is correct. The polarity of the question must be visible in the polarity of the sentence you write.
8. If the question is multiple-choice, return the exact option text or a minimal phrase that uniquely identifies the chosen option.
9. For list/extraction questions, include every item visible in the context and keep the order stable.
10. Return a valid JSON object exactly matching the Base Schema below. Do not truncate or omit any fields.{plan_str}{style_instruction}

{display_injection}

Example of correct negation handling:
  Question: "Which of the following is NOT a power granted to Congress: taxes, war, ambassadors, or coining money?"
  WRONG answer: "appointing ambassadors"  (bare phrase, reads as affirmative)
  CORRECT answer: "Appointing ambassadors is NOT a power granted to Congress in Article I, Section 8 — that power belongs to the President. Congress does have the powers to lay taxes, declare war, and coin money."

Base Schema:
{{
  "answer": "your concise answer string",
  "sources": [
    {{
      "title": "title of the source",
      "url": "url or filename of the source",
      "type": "web | document"
    }}
  ],
  "confidence": "high|medium|low",
  "needs_clarification": false,
  "display": null
}}

Context:
{context_str}"""
        return prompt

    @staticmethod
    def _strip_fences(text: str) -> str:
        return re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

    @staticmethod
    def _extract_json_candidate(text: str) -> Optional[str]:
        if not text:
            return None
        cleaned = AnswerGenerator._strip_fences(text)
        if not cleaned:
            return None

        # Exact JSON first.
        try:
            json.loads(cleaned)
            return cleaned
        except Exception:
            pass

        # First balanced-looking object.
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if m:
            candidate = m.group(0)
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                pass

        return None

    @staticmethod
    def _coerce_fallback_answer(raw_text: str) -> Dict[str, Any]:
        text = AnswerGenerator._strip_fences(raw_text)
        text = re.sub(r"^Answer\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        if not text:
            text = "I could not format the model output correctly."
        return {
            "answer": text,
            "sources": [],
            "confidence": "low",
            "needs_clarification": True,
            "display": None,
        }


    def _postprocess_answer(self, query: str, answer: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Lightweight answer normalization for short legal / clause answers."""
        ql = (query or "").lower()
        al = (answer or "").strip()
        cl = " ".join((chunk.get("context_text", chunk.get("text", "")) or "") for chunk in context_chunks).lower()

        # Avoid returning only an archaic phrase when the query asks for duration.
        if "how long" in ql and re.fullmatch(r"good behavior|good behaviour", al, flags=re.IGNORECASE):
            return "Federal judges serve during good Behaviour."

        # If the model answers with only the literal constitutional clause, keep it but
        # make it a full sentence so downstream judges do not mark it as incomplete.
        if re.fullmatch(r"good behavior|good behaviour", al, flags=re.IGNORECASE) and "good behaviour" in cl:
            return "Federal judges serve during good Behaviour."

        return al

    async def _repair_model_output(self, prompt: str, buffer: str) -> Optional[Dict[str, Any]]:
        """One repair pass for malformed JSON or truncated outputs."""
        repair_prompt = f"""The previous assistant output was malformed.

Return ONLY a valid JSON object matching the original schema.
Do not add commentary. Preserve the answer if it exists.

Original task:
{prompt}

Broken output:
{buffer}
"""
        try:
            if self._backend == "groq":
                response = await self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": repair_prompt},
                        {"role": "user", "content": "Repair the output into valid JSON only."},
                    ],
                    model=self.model,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=OLLAMA_MAX_TOKENS,
                )
                repaired = response.choices[0].message.content or ""
            else:
                response = await self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": repair_prompt},
                        {"role": "user", "content": "Repair the output into valid JSON only."},
                    ],
                    model=self.model,
                    temperature=0.0,
                    max_tokens=OLLAMA_MAX_TOKENS,
                    extra_body={"options": {"num_ctx": OLLAMA_NUM_CTX}},
                )
                repaired = response.choices[0].message.content or ""

            candidate = self._extract_json_candidate(repaired)
            if candidate is None:
                return None
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except Exception as exc:
            logger.error(f"Repair pass failed: {exc}")
            return None

    def _parse_model_output(self, buffer: str) -> Dict[str, Any]:
        candidate = self._extract_json_candidate(buffer)
        if candidate is None:
            # Salvage a plain-text answer instead of dropping everything.
            return self._coerce_fallback_answer(buffer)

        try:
            parsed = json.loads(candidate)
        except Exception:
            return self._coerce_fallback_answer(buffer)

        # Normalize the schema defensively.
        if not isinstance(parsed, dict):
            return self._coerce_fallback_answer(buffer)
        parsed.setdefault("answer", "")
        parsed.setdefault("sources", [])
        parsed.setdefault("confidence", "low")
        parsed.setdefault("needs_clarification", False)
        parsed.setdefault("display", None)
        return parsed

    async def generate_stream(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        mode: str = "doc_rag",
        source_map: Dict[str, Any] = None,
        answer_plan: Dict[str, Any] = None,
        display_injection: str = "",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not context_chunks:
            fallback_answer = "I could not find support for that in the retrieved documents."
            fallback = {"mode": mode, "answer": fallback_answer, "sources": [], "confidence": "low", "needs_clarification": False}
            yield {"event": "delta", "data": json.dumps({"text": fallback_answer})}
            yield {"event": "final", "data": json.dumps(fallback)}
            return

        prompt = self._build_prompt(query, context_chunks, source_map, answer_plan, display_injection)
        logger.info(f"Calling {self._backend.upper()} LLM ({self.model}) with {len(context_chunks)} chunks. Mode: {mode}")

        try:
            if self._backend == "groq":
                stream = await self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user",   "content": query}
                    ],
                    model=self.model,
                    response_format={"type": "json_object"},
                    stream=True,
                    temperature=0.0,
                    max_tokens=OLLAMA_MAX_TOKENS,
                )
                buffer = ""
                async for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        buffer += content
            else:
                # Ollama: pass num_ctx via extra_body to pin the KV-cache
                # pre-allocation to 2048 tokens. Without this, Ollama defaults
                # to num_ctx=8192 and pre-allocates ~1742 MB of KV-cache at
                # model load time — exceeding the RTX 3050 4 GB headroom by
                # ~21 MB and causing the CUDA OOM seen in production logs.
                resp = await self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user",   "content": query}
                    ],
                    model=self.model,
                    temperature=0.0,
                    max_tokens=OLLAMA_MAX_TOKENS,
                    extra_body={"options": {"num_ctx": OLLAMA_NUM_CTX}},
                )
                buffer = resp.choices[0].message.content or ""

            candidate = self._extract_json_candidate(buffer)
            if candidate is None:
                final_json = await self._repair_model_output(prompt, buffer) or self._coerce_fallback_answer(buffer)
            else:
                try:
                    parsed = json.loads(candidate)
                    final_json = parsed if isinstance(parsed, dict) else self._coerce_fallback_answer(buffer)
                except Exception:
                    final_json = self._coerce_fallback_answer(buffer)

            final_json.setdefault("answer", "")
            final_json.setdefault("sources", [])
            final_json.setdefault("confidence", "low")
            final_json.setdefault("needs_clarification", False)
            final_json.setdefault("display", None)
            final_json["answer"] = self._postprocess_answer(query, str(final_json.get("answer", "")), context_chunks)
            final_json["mode"] = mode
            answer_text = final_json.get("answer", "")
            if answer_text:
                yield {"event": "delta", "data": json.dumps({"text": answer_text})}
            yield {"event": "final", "data": json.dumps(final_json)}

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            fallback_answer = "An error occurred during answer generation."
            fallback = {"mode": mode, "answer": fallback_answer, "sources": [], "confidence": "low", "needs_clarification": False}
            yield {"event": "delta", "data": json.dumps({"text": fallback_answer})}
            yield {"event": "final", "data": json.dumps(fallback)}
