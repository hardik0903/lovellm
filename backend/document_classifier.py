import os
import json
from typing import Dict, Any
from groq import AsyncGroq
from logger import logger

class DocumentClassifier:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"

    async def classify(self, query: str) -> Dict[str, Any]:
        prompt = f"""Analyze this document-related query.
Return a JSON object with:
"intent": string (one of "summarize", "qa", "extract", "compare", "outline", "find")
"length": string (optional, for summarize intent: "brief", "standard", "detailed")

Query: "{query}"
"""
        try:
            response = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"[DocumentClassifier] Error: {e}")
            return {"intent": "qa", "length": "standard"}
