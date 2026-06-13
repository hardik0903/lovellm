import re
from logger import logger

class QueryRouter:
    def route(self, query: str) -> str:
        """
        Analyzes the query to return a fixed query class: `bm25`, `dense`, or `hybrid`.
        - bm25: prioritized for specific IDs, names, exact phrases.
        - dense: prioritized for semantic/conceptual questions.
        - hybrid: prioritized when both are needed, or when router confidence is borderline.
        """
        logger.info(f"Routing query: {query}")
        query_lower = query.lower()
        
        # Check for semantic indicators
        semantic_indicators = ["why", "how", "what is", "explain", "describe", "difference between", "compare"]
        is_semantic = any(indicator in query_lower for indicator in semantic_indicators)
        
        # Check for lexical indicators (IDs, numbers, capitalized names, quotes)
        has_id_or_number = bool(re.search(r'\b(?:[A-Z]{2,}[0-9-][A-Z0-9-]*|[0-9]+[A-Z-][A-Z0-9-]*)\b|\b\d+\b', query))
        has_quotes = '"' in query or "'" in query
        
        # Length check
        word_count = len(query.split())
        
        if is_semantic and not has_id_or_number and not has_quotes:
            logger.info("Router decision: dense")
            return "dense"
            
        if has_quotes or (has_id_or_number and word_count < 6 and not is_semantic):
            logger.info("Router decision: bm25")
            return "bm25"
            
        # Default or borderline confidence
        logger.info("Router decision: hybrid")
        return "hybrid"
