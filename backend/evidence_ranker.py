import re
from typing import Dict, Any, List

class EvidenceRanker:
    """
    Computes a deterministic score (0-100) for a piece of evidence.
    """
    
    def __init__(self):
        # Basic domain authority heuristics
        self.high_quality_domains = [
            "wikipedia.org", "britannica.com", "nature.com", "sciencedirect.com",
            "arxiv.org", "github.com", "stackoverflow.com", "mit.edu", "stanford.edu",
            "khanacademy.org", "investopedia.com", "developer.mozilla.org", "docs.python.org"
        ]
        
    def score_evidence(self, query_plan: Dict[str, Any], snippet: str, url: str) -> int:
        score = 0
        snippet_lower = snippet.lower()
        url_lower = url.lower()
        
        concepts = query_plan.get("concepts", [])
        if not concepts:
            concepts = [query_plan.get("normalized_query", "")]
            
        # 1. Semantic/Concept Match (Max 40 points)
        concept_match_score = 0
        for concept in concepts:
            # check if concept exists in snippet
            if concept.lower() in snippet_lower:
                concept_match_score += (40 // len(concepts))
        score += concept_match_score
        
        # 2. Source Quality (Max 30 points)
        quality_score = 10 # base score for any result
        for domain in self.high_quality_domains:
            if domain in url_lower:
                quality_score = 30
                break
        if ".edu" in url_lower or ".gov" in url_lower:
            quality_score = 30
            
        score += quality_score
        
        # 3. Intent Match (Max 30 points)
        intent = query_plan.get("intent", "")
        intent_score = 10 # Base
        
        if intent == "definition":
            if any(term in snippet_lower for term in ["is a", "is defined as", "refers to", "stands for"]):
                intent_score = 30
                
        elif intent == "comparison":
            if any(term in snippet_lower for term in ["difference", "compared to", "vs", "versus", "while"]):
                intent_score = 30
                
        elif intent == "troubleshooting":
            if any(term in snippet_lower for term in ["error", "fix", "solution", "resolve"]):
                intent_score = 30
        else:
            intent_score = 20 # Give a decent default for unhandled intents if it matched concept
            
        score += intent_score
        
        # Penalties
        # e.g., if looking for eigenvector and hits C++ library without asking for it
        if "eigenvector" in [c.lower() for c in concepts]:
            if "eigen" in url_lower and "c++" in snippet_lower:
                score -= 50 # massive penalty
                
        return max(0, min(100, score))

    def filter_and_rank(self, query_plan: Dict[str, Any], results: List[Dict[str, Any]], threshold: int = 40) -> List[Dict[str, Any]]:
        ranked = []
        for res in results:
            score = self.score_evidence(query_plan, res.get("snippet", ""), res.get("url", ""))
            res["evidence_score"] = score
            if score >= threshold:
                ranked.append(res)
                
        ranked.sort(key=lambda x: x["evidence_score"], reverse=True)
        return ranked
