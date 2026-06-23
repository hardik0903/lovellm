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
            "khanacademy.org", "investopedia.com", "developer.mozilla.org", "docs.python.org",
            # Cloud / tech documentation & tutorial sites
            "aws.amazon.com", "docs.aws.amazon.com", "cloud.google.com",
            "learn.microsoft.com", "azure.microsoft.com",
            "digitalocean.com", "medium.com", "dev.to",
            "geeksforgeeks.org", "javatpoint.com", "tutorialspoint.com",
            "baeldung.com", "freecodecamp.org", "realpython.com",
            "towardsdatascience.com", "analyticsvidhya.com",
        ]
        
    def _tokenize(self, text: str) -> List[str]:
        """Split text into lowercase word tokens."""
        return re.findall(r'\b\w+\b', text.lower())

    def _concept_token_overlap(self, concept: str, snippet_lower: str) -> float:
        """Return a 0.0-1.0 score for how well *concept* matches *snippet_lower*.

        Instead of requiring the exact concept string to appear verbatim
        (e.g. "aws ec2"), we tokenize the concept and check what fraction
        of its tokens are present anywhere in the snippet.  A full match
        still scores 1.0; partial matches get proportional credit.
        """
        # Fast path: exact substring match
        if concept.lower() in snippet_lower:
            return 1.0
        tokens = self._tokenize(concept)
        if not tokens:
            return 0.0
        hits = sum(1 for t in tokens if t in snippet_lower)
        return hits / len(tokens)

    def score_evidence(self, query_plan: Dict[str, Any], snippet: str, url: str) -> int:
        score = 0
        snippet_lower = snippet.lower()
        url_lower = url.lower()
        
        concepts = query_plan.get("concepts", [])
        if not concepts:
            concepts = [query_plan.get("normalized_query", "")]
            
        # 1. Semantic/Concept Match (Max 40 points)
        # Token-level overlap: each concept contributes proportionally
        concept_match_score = 0
        per_concept_max = 40 // max(len(concepts), 1)
        for concept in concepts:
            overlap = self._concept_token_overlap(concept, snippet_lower)
            concept_match_score += int(per_concept_max * overlap)
        score += min(concept_match_score, 40)
        
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
            if any(term in snippet_lower for term in ["difference", "compared to", "vs", "versus", "while", "unlike", "whereas"]):
                intent_score = 30
            # Partial credit: snippet mentions both sides of a comparison
            elif len(concepts) >= 2:
                sides_mentioned = sum(
                    1 for c in concepts
                    if self._concept_token_overlap(c, snippet_lower) >= 0.5
                )
                if sides_mentioned >= 2:
                    intent_score = 25
                
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

