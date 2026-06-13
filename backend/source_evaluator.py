from typing import List, Dict, Any
from urllib.parse import urlparse

class SourceEvaluator:
    def __init__(self):
        # Known high-authority domains
        self.high_authority = [
            "edu", "gov", "org", "wikipedia.org", "nature.com", "science.org",
            "nytimes.com", "bbc.com", "reuters.com", "bloomberg.com"
        ]
        
    def evaluate(self, sources: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        evaluated = []
        for src in sources:
            url = src.get("url", "")
            domain = urlparse(url).netloc.lower()
            
            score = 0.5 # Base score
            
            # Domain authority heuristic
            if any(domain.endswith(auth) for auth in self.high_authority):
                score += 0.3
            elif ".com" in domain or ".net" in domain:
                score += 0.1
                
            # We can also add more heuristics here if we parse the snippet or title for relevance
            
            src["credibility_score"] = min(1.0, score)
            evaluated.append(src)
            
        return evaluated
