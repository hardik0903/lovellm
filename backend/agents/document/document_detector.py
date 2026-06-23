import re
from typing import Dict, Any

class DocumentDetector:
    def __init__(self):
        self.keywords = [
            r"according to this document", r"in this document", r"summarize", r"what does the document say about",
            r"extract", r"find all", r"compare sections", r"table of contents",
            r"page \d+", r"this pdf", r"this paper"
        ]
        self.pattern = re.compile(r'\b(' + '|'.join(self.keywords) + r')\b', re.IGNORECASE)

    def detect(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        has_documents = context.get("has_documents", False) if context else False
        
        # If no documents are uploaded, confidence is zero.
        if not has_documents:
            return {
                "is_match": False,
                "confidence": 0.0,
                "reasoning": "No documents uploaded in session"
            }
            
        match = self.pattern.search(query)
        confidence = 0.4 # Base confidence if a document is present, since they might just ask a direct question
        reasoning = ["Documents are present in session"]
        
        if match:
            confidence += 0.5
            reasoning.append("Contains document-specific keywords")
            
        confidence = min(1.0, confidence)
        
        return {
            "is_match": confidence >= 0.5,
            "confidence": confidence,
            "reasoning": " | ".join(reasoning)
        }
