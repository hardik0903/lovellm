from typing import Dict, Any, List

class AnswerPlanner:
    """
    Pre-computes the structure of the final answer to ensure completeness.
    """
    def create_plan(self, query_plan: Dict[str, Any]) -> Dict[str, Any]:
        intent = query_plan.get("intent", "fact_lookup")
        sections = query_plan.get("required_sections", [])
        
        # If the query understanding layer didn't provide sections, we use defaults
        if not sections:
            if intent == "definition":
                sections = ["Core Definition", "Intuition / Analogy", "Common Examples"]
            elif intent == "comparison":
                sections = ["Definition of Concept A", "Definition of Concept B", "Key Differences", "When to use which"]
            elif intent == "troubleshooting":
                sections = ["Problem Identification", "Common Causes", "Step-by-Step Solution"]
            else:
                sections = ["Direct Answer", "Supporting Details"]
                
        return {
            "required_sections": sections,
            "tone": "objective and analytical",
            "formatting": "markdown with clear headings"
        }
