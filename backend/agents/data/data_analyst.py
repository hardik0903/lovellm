import os
import json
from typing import Dict, Any
from groq import AsyncGroq
from logger import logger

class DataAnalyst:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    async def analyze(self, query: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""You are a Data Analyst. Answer the analytical question based ONLY on the data profile below.
Do not hallucinate data that isn't in the preview or implied by the column names/stats.

Data Profile:
{json.dumps(profile, indent=2)}

Return a JSON object with:
"answer": "Clear analytical answer",
"insight_type": "trend/distribution/correlation/comparison/summary",
"statistics": {{"key": "value" (e.g. min, max, mean, growth_rate based on the preview)}},
"follow_up_questions": ["question 1", "question 2"]

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
            logger.error(f"[DataAnalyst] Error: {e}")
            return {"answer": "Failed to analyze data.", "insight_type": "summary"}
