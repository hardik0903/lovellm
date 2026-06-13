import os
import json
from typing import AsyncGenerator, Dict, Any
from groq import AsyncGroq
from logger import logger
from math_prompt_templates import get_prompt_for_category

class MathSolver:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"

    async def solve(self, query: str, classification: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        category = classification.get("category", "algebra")
        system_prompt = get_prompt_for_category(category)
        
        logger.info(f"MathSolver starting for category: {category}")
        
        try:
            stream = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                model=self.model,
                response_format={"type": "json_object"},
                stream=True,
                temperature=0.0
            )

            buffer = ""
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    buffer += content
                    yield {
                        "event": "math_thinking",
                        "data": json.dumps({"delta": content})
                    }
                    
            try:
                final_json = json.loads(buffer)
            except json.JSONDecodeError:
                logger.error("MathSolver failed to output valid JSON.")
                yield {
                    "event": "math_final",
                    "data": json.dumps({"error": "Failed to generate structured math response."})
                }
                return

            for step in final_json.get("steps", []):
                yield {
                    "event": "math_step",
                    "data": json.dumps(step)
                }

            yield {
                "event": "math_final",
                "data": json.dumps(final_json)
            }

        except Exception as e:
            logger.error(f"Error in MathSolver: {e}")
            yield {
                "event": "math_final",
                "data": json.dumps({"error": str(e)})
            }
