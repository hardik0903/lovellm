import os
import json
from typing import Dict, Any
from groq import AsyncGroq
from logger import logger

class ChartGenerator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"

    async def generate_chart(self, query: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""You are a Chart Config Generator. Based on the data profile and user query, generate a chart configuration.
If there's not enough data in the preview to make a real chart, make a representative mock chart based on the columns.

Return a JSON object with:
"type": "bar" | "line" | "scatter" | "pie"
"title": "Chart Title"
"x_axis": {{"label": "X axis name", "values": ["a", "b", "c"]}}
"y_axis": {{"label": "Y axis name", "values": [1, 2, 3]}}
"annotations": [{{"x": "b", "label": "Notable point"}}]

Data Profile:
{json.dumps(profile, indent=2)}

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
            logger.error(f"[ChartGenerator] Error: {e}")
            return None
