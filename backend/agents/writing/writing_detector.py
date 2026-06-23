import re
from typing import Dict, Any

class WritingDetector:
    def __init__(self):
        self.keywords = [
            r"write", r"draft", r"rewrite", r"edit", r"proofread", r"improve",
            r"more formal", r"condense", r"expand", r"email", r"essay",
            r"report", r"blog post", r"cover letter", r"fix grammar", r"rephrase"
        ]
        self.pattern = re.compile(r'\b(' + '|'.join(self.keywords) + r')\b', re.IGNORECASE)

    def detect(self, query: str) -> Dict[str, Any]:
        match = self.pattern.search(query)
        
        confidence = 0.0
        reasoning = []
        
        if match:
            confidence += 0.5
            reasoning.append("Contains writing action/document keyword")
            
            # If the query is long, it might contain the text to edit
            word_count = len(query.split())
            if word_count > 30 and ("edit" in query.lower() or "rewrite" in query.lower() or "proofread" in query.lower()):
                confidence += 0.3
                reasoning.append("Contains large block of text likely for editing")
                
        confidence = min(1.0, confidence)
        
        return {
            "is_match": confidence >= 0.5,
            "confidence": confidence,
            "reasoning": " | ".join(reasoning) if reasoning else "No writing keywords found"
        }
