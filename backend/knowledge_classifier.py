import os
import json
from groq import AsyncGroq
from typing import Dict, Any
from logger import logger

class KnowledgeClassifier:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.model = "llama-3.1-8b-instant" # Smaller model for classification

    async def classify(self, query: str) -> Dict[str, Any]:
        """
        Classifies the knowledge domain and estimates user expertise level.
        """
        prompt = f"""Analyze the following knowledge query.
Return a JSON object with:
"domain": string (e.g. "Science", "History", "Technology", "Philosophy", "General")
"expertise_level": string ("beginner", "intermediate", "expert")

Determine the expertise based on the query phrasing (e.g. "Explain like I'm 5" -> beginner, "difference between L2 and L3 cache" -> expert).

Query: "{query}"
"""
        try:
            response = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error(f"[KnowledgeClassifier] Error: {e}")
            return {"domain": "General", "expertise_level": "intermediate"}
