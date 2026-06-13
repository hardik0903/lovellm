import re
from typing import Dict, Any

class DataDetector:
    def __init__(self):
        self.keywords = [
            r"analyze", r"chart", r"plot", r"trend", r"average", 
            r"distribution", r"correlation", r"compare columns", 
            r"top \d+", r"how many", r"max", r"min", r"mean", r"median"
        ]
        self.pattern = re.compile(r'\b(' + '|'.join(self.keywords) + r')\b', re.IGNORECASE)

    def detect(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        has_data = context.get("has_data_file", False) if context else False
        
        if not has_data:
            return {
                "is_match": False,
                "confidence": 0.0,
                "reasoning": "No tabular data files (csv, xlsx, json) in session"
            }
            
        match = self.pattern.search(query)
        confidence = 0.4 # Base confidence if data is uploaded
        reasoning = ["Tabular data uploaded in session"]
        
        if match:
            confidence += 0.5
            reasoning.append("Contains data analysis keywords")
            
        confidence = min(1.0, confidence)
        
        return {
            "is_match": confidence >= 0.5,
            "confidence": confidence,
            "reasoning": " | ".join(reasoning)
        }
