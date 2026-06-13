import re
from typing import Dict, Any

class KnowledgeDetector:
    def __init__(self):
        # Keywords that strongly suggest a knowledge/conceptual question
        self.knowledge_keywords = [
            r"what is", r"what are", r"explain", r"define", r"how does\s+.+\s+work",
            r"how do\s+.+\s+work", r"what['`]?s the difference between", r"compare",
            r"concept of", r"meaning of"
        ]
        
        # Keywords that suggest it needs research instead of static knowledge
        self.research_keywords = [
            r"latest", r"recent", r"current state", r"news", r"today", r"this week",
            r"this year", r"what happened", r"update on"
        ]
        
        self.knowledge_pattern = re.compile(r'\b(' + '|'.join(self.knowledge_keywords) + r')\b', re.IGNORECASE)
        self.research_pattern = re.compile(r'\b(' + '|'.join(self.research_keywords) + r')\b', re.IGNORECASE)

    def detect(self, query: str) -> Dict[str, Any]:
        """
        Detects if the query is a static knowledge query.
        """
        knowledge_match = self.knowledge_pattern.search(query)
        research_match = self.research_pattern.search(query)
        
        confidence = 0.0
        if knowledge_match:
            confidence = 0.6 # Base confidence for knowledge
            
            # If it's short and conceptual, higher confidence
            word_count = len(query.split())
            if word_count < 15:
                confidence += 0.2
                
        # If it looks like it needs research, we drop the knowledge confidence significantly
        if research_match:
            confidence -= 0.5
            
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "is_match": confidence >= 0.5,
            "confidence": confidence,
            "reasoning": "Detected conceptual keywords" if knowledge_match else "No knowledge keywords found"
        }
