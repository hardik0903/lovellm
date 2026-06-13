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
        # Requires GROQ_API_KEY environment variable
        self.client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY", "dummy_key"))
        self.model = "llama-3.1-8b-instant"

    def _build_prompt(self, query: str, context_chunks: List[Dict[str, Any]], source_map: Dict[str, Any] = None) -> str:
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

        prompt = f"""You are a retrieval-grounded answering system.

Rules:
1. Answer only using the provided context.
2. Do not use outside knowledge unless the context explicitly supports it.
3. Keep the answer concise and accurate.
4. When possible, cite the sources used.
5. Do not hallucinate names, dates, numbers, or clauses.
6. Return a valid JSON object exactly matching this schema:
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
  "needs_clarification": false
}}
Context:
{context_str}

Question:
{query}
"""
        return prompt

    async def generate_stream(self, query: str, context_chunks: List[Dict[str, Any]], mode: str = "doc_rag", source_map: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        if not context_chunks:
            # Fallback for no context
            fallback = {
                "mode": mode,
                "answer": "I could not find support for that in the retrieved documents.",
                "sources": [],
                "confidence": "low",
                "needs_clarification": False
            }
            yield {
                "event": "final",
                "data": json.dumps(fallback)
            }
            return

        prompt = self._build_prompt(query, context_chunks, source_map)
        
        logger.info(f"Calling Groq LLM with {len(context_chunks)} chunks for context. Mode: {mode}")
        
        try:
            stream = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt}
                ],
                model=self.model,
                response_format={"type": "json_object"},
                stream=True,
                temperature=0.0
            )

            buffer = ""
            in_answer = False
            answer_started = False
            
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if not content:
                    continue
                    
                buffer += content
                
                # Simple state machine to extract the "answer" field for delta events
                if not answer_started:
                    # Look for "answer": "
                    if '"answer"' in buffer:
                        start_idx = buffer.find('"answer"')
                        # Find the first quote after "answer"
                        val_start = buffer.find('"', start_idx + 8)
                        if val_start != -1:
                            val_start += 1 # move past quote
                            answer_started = True
                            in_answer = True
                            # Extract what we have so far
                            text_so_far = buffer[val_start:]
                            # check if it ended immediately
                            if '"' in text_so_far and not text_so_far.endswith('\\"'):
                                end_idx = text_so_far.find('"')
                                text_so_far = text_so_far[:end_idx]
                                in_answer = False
                            if text_so_far:
                                yield {
                                    "event": "delta",
                                    "data": json.dumps({'text': text_so_far})
                                }
                elif in_answer:
                    # Check if we hit the closing quote of the answer field
                    # This is a naive check (assumes no escaped quotes in content, or handles them simply)
                    # For production, a proper incremental JSON parser is better.
                    if content.find('"') != -1:
                        # Might be closing quote
                        parts = content.split('"', 1)
                        if parts[0]:
                            yield {
                                "event": "delta",
                                "data": json.dumps({'text': parts[0]})
                            }
                        in_answer = False
                    else:
                        # Just yield the delta text
                        yield {
                            "event": "delta",
                            "data": json.dumps({'text': content})
                        }
            
            # Now parse the full buffer as JSON to send the final event
            try:
                final_json = json.loads(buffer)
                final_json["mode"] = mode
            except json.JSONDecodeError:
                logger.error("Failed to parse Groq output as JSON.")
                # Attempt to recover or use fallback
                final_json = {
                    "mode": mode,
                    "answer": "Error: Failed to generate valid structured output.",
                    "sources": [],
                    "confidence": "low",
                    "needs_clarification": True
                }
                
            yield {
                "event": "final",
                "data": json.dumps(final_json)
            }

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            fallback = {
                "mode": mode,
                "answer": "An error occurred during answer generation.",
                "sources": [],
                "confidence": "low",
                "needs_clarification": False
            }
            yield {
                "event": "final",
                "data": json.dumps(fallback)
            }
