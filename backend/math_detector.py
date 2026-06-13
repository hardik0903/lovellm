import re
from typing import Dict, Any, List

class MathDetector:
    def __init__(self):
        self.keywords = [
            "solve", "simplify", "factor", "expand", "derivative", "integrate", "integral", "limit",
            "area", "perimeter", "volume", "angle", "mean", "median", "variance", "probability",
            "prime", "gcd", "lcm", "modulo", "equation", "calculate", "compute", "math"
        ]
        self.expression_patterns = [
            r'\d+\s*[a-zA-Z]\s*[\+\-\*\/]\s*\d+\s*=\s*\d+', # e.g. 2x + 5 = 11
            r'∫.*dx', # integrals
            r'sin\(|cos\(|tan\(', # trig
            r'\d+!', # factorials
            r'log_?\d*\(', # logarithms
            r'\\[a-zA-Z]+' # LaTeX macros
        ]

    def detect(self, query: str) -> Dict[str, Any]:
        q_lower = query.lower()
        signals = []
        score = 0.0
        
        matched_keywords = [kw for kw in self.keywords if kw in q_lower]
        if matched_keywords:
            signals.append("keyword")
            score += min(len(matched_keywords) * 0.2, 0.4)
            
        matched_patterns = []
        for pat in self.expression_patterns:
            if re.search(pat, query):
                matched_patterns.append(pat)
        
        if matched_patterns:
            signals.append("numeric_expression")
            score += 0.5
            
        if re.search(r'\d+', query) and re.search(r'[\+\-\*\/\=\^]', query):
            if "numeric_expression" not in signals:
                signals.append("basic_math")
            score += 0.3
            
        if '\\' in query and ('_' in query or '^' in query or '{' in query):
            signals.append("latex")
            score += 0.4
            
        confidence = min(score, 1.0)
        
        return {
            "is_math": confidence >= 0.6,
            "confidence": confidence,
            "detected_signals": signals
        }
