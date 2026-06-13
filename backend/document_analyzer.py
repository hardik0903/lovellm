import os
import json
from groq import AsyncGroq
from logger import logger

class DocumentAnalyzer:
    """
    Runs on document upload to build a structural map of the document.
    """
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"

    async def analyze(self, text: str) -> dict:
        """
        Analyzes the first chunk of a document to determine its structural map.
        """
        # We only pass the first ~3000 chars to avoid massive context
        sample_text = text[:3000]
        
        prompt = f"""Analyze the following document sample and build a structural map.
Return a JSON object with:
"document_type": string ("academic paper", "contract", "report", "manual", "general")
"key_entities": array of strings
"sections_detected": array of strings (guess based on the sample or standard structure for this type)

Sample Text:
{sample_text}
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
            logger.error(f"[DocumentAnalyzer] Error: {e}")
            return {
                "document_type": "general",
                "key_entities": [],
                "sections_detected": []
            }
