import os
import json
import asyncio
from dotenv import load_dotenv
load_dotenv()
from typing import AsyncGenerator, List, Dict, Any
from groq import AsyncGroq
from logger import logger

class AnswerGenerator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"

    def _build_prompt(self, query: str, context_chunks: List[Dict[str, Any]], source_map: Dict[str, Any] = None, answer_plan: Dict[str, Any] = None, display_injection: str = "") -> str:
        context_str = ""
        for i, chunk in enumerate(context_chunks):
            doc_id = chunk.get("metadata", {}).get("document_id", "unknown")
            page = chunk.get("metadata", {}).get("page_start", "?")
            source_file = chunk.get("metadata", {}).get("source_file", "unknown")
            is_web = chunk.get("metadata", {}).get("is_web", False)
            
            # If we have a source map (for web RAG), inject the real title/url
            if source_map and doc_id in source_map:
                title = source_map[doc_id]["title"]
                url = source_map[doc_id]["url"]
            else:
                title = source_file
                url = source_file

            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            text = chunk.get("context_text", chunk.get("text", ""))
            
            src_type = "web" if is_web else "document"
            context_str += f"\n--- [Source ID: {doc_id}, Title: {title}, URL: {url}, Type: {src_type}] ---\n{text}\n"

        sections = answer_plan.get("required_sections", []) if answer_plan else []
        plan_str = f"\nStructure your answer with these sections: {', '.join(sections)}" if sections else ""

        prompt = f"""You are a retrieval-grounded answering system.

Rules:
1. Answer only using the provided context.
2. Do not use outside knowledge unless the context explicitly supports it.
3. Keep the answer concise and accurate.
4. When possible, cite the sources used.
5. Do not hallucinate names, dates, numbers, or clauses.{plan_str}
6. Return a valid JSON object exactly matching the Base Schema below.

{display_injection}

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

    async def generate_stream(self, query: str, context_chunks: List[Dict[str, Any]], mode: str = "doc_rag", source_map: Dict[str, Any] = None, answer_plan: Dict[str, Any] = None, display_injection: str = "") -> AsyncGenerator[Dict[str, Any], None]:
        if not context_chunks:
            # Fallback for no context
            fallback_answer = "I could not find support for that in the retrieved documents."
            fallback = {
                "mode": mode,
                "answer": fallback_answer,
                "sources": [],
                "confidence": "low",
                "needs_clarification": False
            }
            yield {
                "event": "delta",
                "data": json.dumps({"text": fallback_answer})
            }
            yield {
                "event": "final",
                "data": json.dumps(fallback)
            }
            return

        prompt = self._build_prompt(query, context_chunks, source_map, answer_plan, display_injection)
        
        logger.info(f"Calling Groq LLM with {len(context_chunks)} chunks for context. Mode: {mode}")
        
        try:
            stream = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query}
                ],
                model=self.model,
                response_format={"type": "json_object"},
                stream=True,
                temperature=0.0
            )

            buffer = ""
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    buffer += content
            
            # Now parse the full buffer as JSON
            try:
                final_json = json.loads(buffer)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse Groq output as JSON. Raw: {buffer[:200]}")
                final_json = {
                    "answer": "Error: Failed to generate valid structured output.",
                    "sources": [],
                    "confidence": "low",
                    "needs_clarification": True
                }

            final_json["mode"] = mode

            # Emit a single delta with the full answer text so the frontend
            # still gets a streaming-style event
            answer_text = final_json.get("answer", "")
            if answer_text:
                yield {
                    "event": "delta",
                    "data": json.dumps({'text': answer_text})
                }
                
            yield {
                "event": "final",
                "data": json.dumps(final_json)
            }

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            fallback_answer = "An error occurred during answer generation."
            fallback = {
                "mode": mode,
                "answer": fallback_answer,
                "sources": [],
                "confidence": "low",
                "needs_clarification": False
            }
            yield {
                "event": "delta",
                "data": json.dumps({"text": fallback_answer})
            }
            yield {
                "event": "final",
                "data": json.dumps(fallback)
            }