import os
import json
from typing import Dict, Any
from groq import AsyncGroq
from logger import logger

class MathClassifier:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"

    async def classify(self, query: str) -> Dict[str, Any]:
        system_prompt = """You are a mathematical query classifier.
Analyze the user's math problem and classify it.
Categories: arithmetic, algebra, quadratic, calculus_diff, calculus_int, geometry, trigonometry, statistics, matrix, number_theory, word_problem.

Return ONLY JSON:
{
  "category": "algebra",
  "subcategory": "linear_equation",
  "difficulty": "beginner|intermediate|advanced",
  "requires_graph": true|false,
  "estimated_steps": 4
}
Set "requires_graph" to true ONLY if graphing is essential (functions, inequalities, intersections, regions).
"""
        try:
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error in math classification: {e}")
            return {
                "category": "algebra",
                "subcategory": "unknown",
                "difficulty": "intermediate",
                "requires_graph": False,
                "estimated_steps": 3
            }
