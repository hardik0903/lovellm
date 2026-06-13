import os
import json
from typing import Dict, Any
from groq import AsyncGroq
from logger import logger

class ResearchPlanner:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"

    async def plan(self, query: str) -> Dict[str, Any]:
        prompt = f"""You are a research planner. Analyze this query and create a research plan.
Return a JSON object with:
"subtopics": array of strings (aspects to cover)
"search_queries": array of strings (2-4 search queries to run)
"min_sources": integer (usually 3-5)
"credibility_threshold": float (0.0 to 1.0, usually 0.7)

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
            logger.error(f"[ResearchPlanner] Error: {e}")
            return {
                "subtopics": ["General overview"],
                "search_queries": [query],
                "min_sources": 3,
                "credibility_threshold": 0.5
            }
