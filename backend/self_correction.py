import asyncio
from typing import Dict, Any, List
from logger import logger
from web_search import WebSearcher
from evidence_ranker import EvidenceRanker
from source_selector import SourceSelector
from retrieval_memory import RetrievalMemory

class MultiPassRetriever:
    """
    Orchestrates the retrieval process with self-correction and memory.
    """
    def __init__(self, searcher: WebSearcher):
        self.searcher = searcher
        self.ranker = EvidenceRanker()
        self.selector = SourceSelector()
        
    async def retrieve(self, query_plan: Dict[str, Any], memory: RetrievalMemory) -> List[Dict]:
        budget = query_plan.get("budget", {})
        max_passes = budget.get("max_search_passes", 1)
        max_sources = budget.get("max_sources", 3)
        
        current_query = query_plan.get("normalized_query", "")
        
        for pass_num in range(1, max_passes + 1):
            logger.info(f"Retrieval Pass {pass_num} for query: {current_query}")
            
            # Avoid repeating exact same query
            if memory.is_query_searched(current_query):
                logger.info(f"Query already searched: {current_query}. Breaking loop.")
                break
                
            memory.add_query(current_query)
            
            # 1. Search
            results = self.searcher.search(current_query, max_results=7) # Fetch more to allow filtering
            
            # Filter out already rejected sources
            valid_results = [r for r in results if not memory.is_source_rejected(r["url"])]
            
            # 2. Score Evidence
            ranked_results = self.ranker.filter_and_rank(query_plan, valid_results, threshold=30)
            
            # Update memory with rejected
            for res in valid_results:
                if res not in ranked_results:
                    memory.reject_source(res["url"])
                    
            # 3. Source Selection
            selected_sources = self.selector.select_diverse_sources(ranked_results, max_sources)
            
            # 4. Evaluate if we have enough good evidence
            if len(selected_sources) >= max_sources or (len(selected_sources) > 0 and pass_num == max_passes):
                for s in selected_sources:
                    memory.accept_source(s)
                return selected_sources
                
            # If we don't have enough good sources and we have more passes, rewrite query
            if pass_num < max_passes:
                logger.info("Insufficient good evidence. Triggering self-correction search.")
                concepts = query_plan.get("concepts", [])
                if concepts:
                    # Simple deterministic rewrite for next pass
                    current_query = f"{concepts[0]} explanation definition"
                else:
                    break
                    
        return memory.get_accepted_sources()
