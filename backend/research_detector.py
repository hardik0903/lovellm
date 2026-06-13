import re
from typing import Dict, Any

class ResearchDetector:
    def __init__(self):
        self.keywords = [
            r"latest", r"recent", r"current state", r"what is the state of",
            r"overview", r"history of", r"compare", r"pros and cons", 
            r"what happened", r"update on", r"news", r"today", r"this week",
            r"this year", r"in 202"
        ]
        self.pattern = re.compile(r'\b(' + '|'.join(self.keywords) + r')\b', re.IGNORECASE)

    def detect(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        match = self.pattern.search(query)
        
        confidence = 0.0
        reasoning = []
        
        if match:
            confidence += 0.6
            reasoning.append("Contains research or recency keywords")
            
        # If it's a very long query, it might be more complex than simple knowledge
        word_count = len(query.split())
        if word_count > 15 and not match:
            confidence += 0.3
            reasoning.append("Long query might require synthesis")
            
        confidence = min(1.0, confidence)
        
        return {
            "is_match": confidence >= 0.5,
            "confidence": confidence,
            "reasoning": " | ".join(reasoning) if reasoning else "No research indicators found"
        }
