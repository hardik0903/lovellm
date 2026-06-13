import os
import json
from typing import Dict, Any, AsyncGenerator
from groq import AsyncGroq
from logger import logger

class CodeSolver:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    async def solve(self, query: str, classification: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        intent = classification.get("intent", "write")
        language = classification.get("language", "unknown")
        
        yield {
            "event": "code_thinking",
            "data": json.dumps({"status": f"Analyzing {language} code for {intent} task..."})
        }
        
        schema = {
            "intent": intent,
            "language": language,
            "problem_summary": "What's wrong in one sentence (if debugging) or what is being built",
            "code_before": "Original broken code (if applicable)",
            "code_after": "Fixed or generated code",
            "diff": [
                {
                    "line": 5,
                    "before": "old code line",
                    "after": "new code line",
                    "reason": "Why it was changed"
                }
            ],
            "explanation": "Step by step explanation",
            "time_complexity": "O(n) if applicable",
            "space_complexity": "O(1) if applicable",
            "alternative_approaches": ["approach A", "approach B"],
            "common_mistakes": ["mistake 1"]
        }

        prompt = f"""You are an expert {language} developer.
Task: {intent}
Query: {query}

Provide a structured response that perfectly matches the following JSON schema:
{json.dumps(schema, indent=2)}

Ensure your code follows idiomatic {language} conventions and best practices.
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
                    "mode": "code",
                    "display": final_json
                })
            }
        except Exception as e:
            logger.error(f"[CodeSolver] Error: {e}")
            yield {
                "event": "final",
                "data": json.dumps({
                    "mode": "code",
                    "answer": "An error occurred while solving the code task.",
                    "display": None
                })
            }
