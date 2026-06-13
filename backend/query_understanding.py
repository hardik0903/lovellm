import os
import json
import re
from typing import Dict, Any
from groq import AsyncGroq
from logger import logger

class QueryUnderstandingEngine:
    def __init__(self):
        self.client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY", "dummy_key"))
        self.model = "llama-3.1-8b-instant"

    def _normalize_query(self, query: str) -> str:
        # Fast deterministic normalization
        q = query.strip().lower()
        q = re.sub(r'\s+', ' ', q)
        return q

    def _is_simple_query(self, query_lower: str) -> bool:
        """
        Determines if a query can be handled deterministically.
        Simple lookups: "what is X", "who is Y", "when did Z", "where is W"
        """
        words = query_lower.split()
        if len(words) < 8:
            if any(query_lower.startswith(prefix) for prefix in ["what is", "who is", "when is", "where is", "define "]):
                return True
        return False

    def _deterministic_understand(self, original_query: str, normalized_query: str) -> Dict[str, Any]:
        """
        Fast path for simple queries.
        """
        logger.info("Using Deterministic Understanding.")
        
        # Extract concept heuristically
        concept = normalized_query
        for prefix in ["what is a", "what is an", "what is", "who is", "when is", "where is", "define "]:
            if normalized_query.startswith(prefix):
                concept = normalized_query[len(prefix):].strip()
                break
                
        # Basic intent
        intent = "definition" if "what is" in normalized_query or "define" in normalized_query else "fact_lookup"

        return {
            "original_query": original_query,
            "normalized_query": normalized_query,
            "intent": intent,
            "concepts": [concept.strip("?")],
            "required_sections": [intent],
            "mode": "direct_web",
            "is_complex": False,
            "needs_clarification": False
        }

    async def _llm_understand(self, original_query: str, normalized_query: str) -> Dict[str, Any]:
        """
        Deep planner for complex queries.
        """
        logger.info("Using LLM Planner for Query Understanding.")
        prompt = f"""You are a query analysis engine. Your job is to decompose the user's query into a structured plan for search and answer generation.

Rules:
1. Classify the intent: 'comparison', 'research', 'multi_hop', 'troubleshooting', 'ambiguous'.
2. Identify the core concepts (entities/topics).
3. Determine required sections for a complete answer (e.g., ['definition', 'differences', 'examples']).
4. If the query is highly ambiguous and requires clarification from the user, set 'needs_clarification' to true.
5. Return ONLY a JSON object matching this schema:
{{
    "intent": "string",
    "concepts": ["string"],
    "required_sections": ["string"],
    "needs_clarification": boolean
}}

Query: {original_query}
"""
        try:
            response = await self.client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            content = response.choices[0].message.content
            parsed = json.loads(content)
            
            return {
                "original_query": original_query,
                "normalized_query": normalized_query,
                "intent": parsed.get("intent", "research"),
                "concepts": parsed.get("concepts", []),
                "required_sections": parsed.get("required_sections", []),
                "mode": "clarify" if parsed.get("needs_clarification") else "web_rag",
                "is_complex": True,
                "needs_clarification": parsed.get("needs_clarification", False)
            }
        except Exception as e:
            logger.error(f"Error in LLM understanding: {e}")
            # Fallback to simple deterministic if LLM fails
            return self._deterministic_understand(original_query, normalized_query)

    async def understand(self, query: str, has_documents: bool = False) -> Dict[str, Any]:
        """
        Main entry point. Enforces strict escalation framework.
        """
        norm_q = self._normalize_query(query)
        
        # If user explicitly references a document, or there are docs and it's not a generic web query
        explicit_doc = bool(re.search(r'\b(document|file|text|pdf|upload|this|it)\b', norm_q))
        if has_documents and explicit_doc:
            result = self._deterministic_understand(query, norm_q)
            result["mode"] = "doc_rag"
            result["intent"] = "document_lookup"
            return result

        if self._is_simple_query(norm_q):
            return self._deterministic_understand(query, norm_q)
        else:
            return await self._llm_understand(query, norm_q)
