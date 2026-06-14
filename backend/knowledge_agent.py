import os
import json
from typing import Dict, Any, AsyncGenerator
from groq import AsyncGroq
from logger import logger
from agent_base import BaseAgent
from knowledge_detector import KnowledgeDetector
from knowledge_classifier import KnowledgeClassifier

class KnowledgeAgent(BaseAgent):
    def __init__(self):
        self.detector = KnowledgeDetector()
        self.classifier = KnowledgeClassifier()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile" # Larger model for synthesis

    def detect(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.detector.detect(query)

    async def classify(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return await self.classifier.classify(query)

    async def solve(self, query: str, context: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        classification = await self.classify(query)
        domain = classification.get("domain", "General")
        expertise = classification.get("expertise_level", "intermediate")
        
        # Stream a thinking event
        yield {
            "event": "knowledge_thinking",
            "data": json.dumps({"status": f"Consulting knowledge base for {domain} at {expertise} level..."})
        }
        
        schema = {
            "concept": "Name of the concept",
            "domain": domain,
            "expertise_level": expertise,
            "definition": "One sentence definition",
            "explanation": "Core explanation calibrated to expertise level",
            "analogy": "An analogy that makes it intuitive",
            "components": [
                {
                    "name": "Component name",
                    "role": "What it does in the system",
                    "simple_description": "Plain English version"
                }
            ],
            "examples": ["Example 1", "Example 2"],
            "common_misconceptions": ["Misconception 1"],
            "related_concepts": ["Concept 1", "Concept 2"],
            "further_reading": ["Topic to explore next"]
        }
        
        prompt = f"""You are an expert Knowledge Agent.
Explain the following concept to a user with an expertise level of: {expertise}.
If the user is a beginner, use simple terms and focus on intuition.
If the user is an expert, use precise technical language.

Return ONLY a valid JSON object matching this schema:
{json.dumps(schema, indent=2)}

Query: "{query}"
"""

        try:
            stream = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                response_format={"type": "json_object"},
                stream=True,
                temperature=0.3
            )
            
            buffer = ""
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    buffer += content
            
            final_json = json.loads(buffer)

            # Build a human-readable answer string from the structured knowledge response
            answer_text = final_json.get("definition", "") or final_json.get("explanation", "") or ""
            if not answer_text:
                answer_text = "Here is a knowledge-based explanation for your query."

            # Emit delta so streaming consumers (and tests) receive incremental text
            yield {
                "event": "delta",
                "data": json.dumps({"text": answer_text})
            }

            # Send final event — include both 'answer' (for test/consumer compatibility)
            # and 'display' (for rich UI rendering)
            yield {
                "event": "final",
                "data": json.dumps({
                    "mode": "knowledge",
                    "answer": answer_text,
                    "sources": [],
                    "confidence": "high",
                    "needs_clarification": False,
                    "display": final_json
                })
            }
            
        except Exception as e:
            logger.error(f"[KnowledgeAgent] Error: {e}")
            yield {
                "event": "final",
                "data": json.dumps({
                    "mode": "knowledge",
                    "answer": "Failed to generate knowledge explanation.",
                    "sources": [],
                    "confidence": "low",
                    "needs_clarification": False,
                    "display": None
                })
            }