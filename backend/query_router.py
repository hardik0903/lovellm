import re
from logger import logger

class MainQueryRouter:
    def route(self, query: str, has_documents: bool = False) -> str:
        """
        Routes the query into one of three modes:
        - doc_rag: If documents are uploaded.
        - direct_web: Simple factual lookups, definitions, direct what/who/when.
        - web_rag: Comparative, analytical, multi-step, synthesis.
        """
        logger.info(f"Main routing for query: {query}")
        
        # 1. Web RAG indicators: complex reasoning, synthesis
        query_lower = query.lower()
        complex_indicators = [
            "compare", "difference between", "vs", "versus", "analyze", 
            "explain why", "how do", "step by step", "pros and cons",
            "relationship between", "summarize"
        ]
        
        is_complex = any(indicator in query_lower for indicator in complex_indicators)
        word_count = len(query.split())
        
        explicit_doc_reference = bool(re.search(r'\b(document|file|text|pdf|upload|this|it)\b', query_lower))

        # 2. Routing Rules
        # If the user explicitly mentions the document, or if documents are present and it's not clearly a generic web lookup
        
        is_simple_web = word_count < 10 and any(query_lower.startswith(prefix) for prefix in ["what is", "who is", "when is", "where is"])
        
        if explicit_doc_reference and has_documents:
            logger.info("Main router decision: doc_rag (explicit doc reference)")
            return "doc_rag"
            
        if is_simple_web:
            logger.info("Main router decision: direct_web (simple lookup)")
            return "direct_web"
            
        if is_complex:
            logger.info("Main router decision: web_rag (complex/analytical)")
            return "web_rag"
            
        if has_documents:
            logger.info("Main router decision: doc_rag (fallback to documents)")
            return "doc_rag"
            
        # Default to web_rag if long, otherwise direct_web
        if word_count > 12:
            logger.info("Main router decision: web_rag (long query, no docs)")
            return "web_rag"
            
        logger.info("Main router decision: direct_web (default)")
        return "direct_web"
