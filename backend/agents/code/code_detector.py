import re
from typing import Dict, Any

class CodeDetector:
    def __init__(self):
        self.languages = [
            r"python", r"javascript", r"java\b", r"rust", r"sql", 
            r"bash", r"typescript", r"c\+\+", r"golang", r"go\b"
        ]
        self.actions = [
            r"debug", r"fix", r"write a function", r"explain this code", 
            r"convert", r"optimize", r"what does this code do", r"error", 
            r"exception", r"traceback", r"refactor"
        ]
        
        self.lang_pattern = re.compile(r'\b(' + '|'.join(self.languages) + r')\b', re.IGNORECASE)
        self.action_pattern = re.compile(r'\b(' + '|'.join(self.actions) + r')\b', re.IGNORECASE)
        
        # Detect backticks (code blocks) or stacktrace patterns
        self.code_block_pattern = re.compile(r'```[a-z]*\n.*?\n```', re.DOTALL)
        self.stacktrace_pattern = re.compile(r'Traceback \(most recent call last\):|TypeError:|ValueError:|ReferenceError:', re.IGNORECASE)

    def detect(self, query: str) -> Dict[str, Any]:
        has_lang = bool(self.lang_pattern.search(query))
        has_action = bool(self.action_pattern.search(query))
        has_code_block = bool(self.code_block_pattern.search(query))
        has_stacktrace = bool(self.stacktrace_pattern.search(query))
        
        confidence = 0.0
        reasoning = []
        
        if has_code_block or has_stacktrace:
            confidence += 0.5
            reasoning.append("Contains code block or stacktrace")
            
        if has_action:
            confidence += 0.3
            reasoning.append("Contains code action keyword")
            
        if has_lang:
            confidence += 0.2
            reasoning.append("Contains programming language keyword")
            
        confidence = min(1.0, confidence)
        
        return {
            "is_match": confidence >= 0.5,
            "confidence": confidence,
            "reasoning": " | ".join(reasoning) if reasoning else "No code patterns detected"
        }
