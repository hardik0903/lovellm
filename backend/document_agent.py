import os
import json
from typing import Dict, Any, AsyncGenerator
from groq import AsyncGroq
from logger import logger
from agent_base import BaseAgent
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
        self.model = "llama-3.3-70b-versatile"

    def detect(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.detector.detect(query, context)

    async def classify(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return await self.classifier.classify(query)

    async def solve(self, query: str, context: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        classification = await self.classify(query)
        intent = classification.get("intent", "qa")
        
        yield {
            "event": "document_thinking",
            "data": json.dumps({"status": f"Analyzing document for {intent} task..."})
        }
        
        # Pull context_chunks and local_retriever from context
        local_retriever = context.get("local_retriever") if context else None
        
        if local_retriever:
            # We fetch more chunks for document agents
            context_chunks = local_retriever.retrieve(query, top_k=10)
        else:
            context_chunks = context.get("context_chunks", []) if context else []
            
        context_str = ""
        for i, chunk in enumerate(context_chunks):
            page = chunk.get("metadata", {}).get("page_start", "?")
            text = chunk.get("context_text", chunk.get("text", ""))
            context_str += f"\n--- [Page {page}] ---\n{text}\n"
            
        if intent == "qa":
            schema = {
                "intent": "qa",
                "answer": "Direct answer to the question",
                "evidence": [
                    {
                        "quote": "Relevant passage from document",
                        "section": "Section name if known",
                        "page": 7,
                        "relevance": 0.94
                    }
                ],
                "confidence": 0.91,
                "answer_location": "Section 3.2, Page 7",
                "caveat": "Note if the document doesn't fully address the question"
            }
        else:
            schema = {
                "intent": "summarize",
                "length": classification.get("length", "standard"),
                "summary": "The summary text",
                "key_points": ["Point 1", "Point 2"],
                "document_type": "Document Type",
                "sections_covered": ["Introduction"],
                "word_count_original": 0,
                "word_count_summary": 0
            }

        prompt = f"""You are an expert Document Analyst.
Task: {intent}
Query: {query}

Provide a structured response that perfectly matches the following JSON schema:
{json.dumps(schema, indent=2)}

Use the following document chunks as context to answer the user's query:
{context_str}
"""
        try:
            stream = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                response_format={"type": "json_object"},
                stream=True,
                temperature=0.1
            )
            
            buffer = ""
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    buffer += content
            
            final_json = json.loads(buffer)
            yield {
                "event": "final",
                "data": json.dumps({
                    "mode": "document",
                    "display": final_json
                })
            }
        except Exception as e:
            logger.error(f"[DocumentAgent] Error: {e}")
            yield {
                "event": "final",
                "data": json.dumps({
                    "mode": "document",
                    "answer": "An error occurred while analyzing the document.",
                    "display": None
                })
            }
