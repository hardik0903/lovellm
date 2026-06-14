import os
import json
from typing import Dict, Any, AsyncGenerator
from groq import AsyncGroq
from logger import logger
from agent_base import BaseAgent
from writing_detector import WritingDetector
from writing_classifier import WritingClassifier

class WritingAgent(BaseAgent):
    def __init__(self):
        self.detector = WritingDetector()
        self.classifier = WritingClassifier()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    def detect(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.detector.detect(query)

    async def classify(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return await self.classifier.classify(query)

    async def solve(self, query: str, context: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        classification = await self.classify(query)
        intent = classification.get("intent", "draft")
        doc_type = classification.get("document_type", "general")
        
        yield {
            "event": "writing_thinking",
            "data": json.dumps({"status": f"Preparing to {intent} a {doc_type}..."})
        }
        
        schema = {
            "intent": intent,
            "document_type": doc_type,
            "tone": "detected or requested tone",
            "original": "Original text if editing/rewriting",
            "result": "Drafted or improved text",
            "changes": [
                {
                    "type": "grammar / clarity / tone / structure / conciseness",
                    "original_phrase": "old phrase",
                    "revised_phrase": "new phrase",
                    "reason": "Why it was changed"
                }
            ],
            "readability_before": "Grade level before (if editing)",
            "readability_after": "Grade level after",
            "word_count_before": 0,
            "word_count_after": 0,
            "summary_of_changes": "Short summary of edits made"
        }

        prompt = f"""You are an expert Writing Assistant.
Task: {intent}
Document Type: {doc_type}
Query: {query}

Provide a structured response that perfectly matches the following JSON schema:
{json.dumps(schema, indent=2)}

If the task is to draft something new from scratch, the 'original' field should be empty and 'changes' can be empty.
If the task is to edit existing text, fill out the changes array with specific edits you made.
"""
        try:
            stream = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                response_format={"type": "json_object"},
                stream=True,
                temperature=0.5
            )
            
            buffer = ""
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    buffer += content
            
            final_json = json.loads(buffer)

            # Extract a human-readable answer for consumers that expect the 'answer' field
            answer_text = (
                final_json.get("result") or
                final_json.get("summary_of_changes") or
                ""
            )
            if not answer_text:
                answer_text = "Here is the writing result for your query."

            yield {
                "event": "delta",
                "data": json.dumps({"text": answer_text})
            }
            yield {
                "event": "final",
                "data": json.dumps({
                    "mode": "writing",
                    "answer": answer_text,
                    "sources": [],
                    "confidence": "high",
                    "needs_clarification": False,
                    "display": final_json
                })
            }
        except Exception as e:
            logger.error(f"[WritingAgent] Error: {e}")
            yield {
                "event": "final",
                "data": json.dumps({
                    "mode": "writing",
                    "answer": "An error occurred during the writing task.",
                    "sources": [],
                    "confidence": "low",
                    "needs_clarification": False,
                    "display": None
                })
            }