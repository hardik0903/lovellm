import os
import json
from typing import Dict, Any, List
from groq import AsyncGroq
from logger import logger

class ResearchSynthesizer:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    async def synthesize(self, plan: Dict[str, Any], sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        # Prepare context string
        context_str = ""
        for i, src in enumerate(sources):
            title = src.get("title", "Unknown")
            url = src.get("url", "Unknown")
            text = src.get("text", "")[:3000] # Limit length per source
            context_str += f"\n--- [Source {i}: {title} | URL: {url}] ---\n{text}\n"

        schema = {
            "topic": "The main topic",
            "confidence": 0.87,
            "summary": "3-4 sentence executive summary",
            "sections": [
                {
                    "title": "Section title based on plan subtopics",
                    "content": "Synthesized paragraph",
                    "sources": [0, 1],
                    "consensus_level": "high / medium / contested"
                }
            ],
            "sources": [
                {
                    "index": 0,
                    "title": "Source title",
                    "url": "https://...",
                    "credibility_score": 0.9,
                    "recency": "YYYY-MM if known"
                }
            ],
            "conflicting_claims": [
                {
                    "claim": "X vs Y",
                    "supporters": [0],
                    "opponents": [1],
                    "verdict": "Contested"
                }
            ],
            "knowledge_gaps": ["What we don't know yet"],
            "last_updated": "2026-06"
        }

        prompt = f"""You are an expert Research Synthesizer.
Based on the provided sources, synthesize a comprehensive research report on the topic.
Address the following subtopics if possible: {', '.join(plan.get('subtopics', []))}.

Return ONLY a valid JSON object matching this schema:
{json.dumps(schema, indent=2)}

Sources:
{context_str}
"""
        try:
            response = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.2
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"[ResearchSynthesizer] Error: {e}")
            return {"error": "Failed to synthesize research"}
