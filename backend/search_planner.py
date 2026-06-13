from typing import Dict, Any

class SearchPlanner:
    """
    Determines retrieval budgets and strategies based on query understanding output.
    """
    def plan(self, query_plan: Dict[str, Any]) -> Dict[str, Any]:
        intent = query_plan.get("intent", "fact_lookup")
        mode = query_plan.get("mode", "direct_web")
        is_complex = query_plan.get("is_complex", False)
        
        # Default budgets
        budget = {
            "max_search_passes": 1,
            "max_sources": 3,
            "max_tokens_for_context": 4000,
            "retrieval_strategy": mode
        }
        
        if intent in ["comparison", "research"]:
            budget["max_search_passes"] = 2
            budget["max_sources"] = 5
            budget["max_tokens_for_context"] = 6000
            budget["retrieval_strategy"] = "web_rag"
            
        elif intent == "multi_hop":
            budget["max_search_passes"] = 3
            budget["max_sources"] = 7
            budget["max_tokens_for_context"] = 8000
            budget["retrieval_strategy"] = "web_rag"
            
        elif intent == "definition":
            budget["max_search_passes"] = 1
            budget["max_sources"] = 2
            budget["max_tokens_for_context"] = 2000
            budget["retrieval_strategy"] = "direct_web"
            
        # Doc RAG specific
        if mode == "doc_rag":
            budget["max_search_passes"] = 1
            budget["max_sources"] = 5
            budget["max_tokens_for_context"] = 4000
            budget["retrieval_strategy"] = "doc_rag"
            
        elif mode == "math" or intent == "math":
            budget["max_search_passes"] = 0
            budget["max_sources"] = 0
            budget["max_tokens_for_context"] = 0
            budget["retrieval_strategy"] = "math"
            
        # Return merged plan
        final_plan = {**query_plan, "budget": budget}
        final_plan["mode"] = budget["retrieval_strategy"]
        return final_plan
