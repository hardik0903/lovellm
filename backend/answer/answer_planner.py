from typing import Dict, Any, List

class AnswerPlanner:
    """
    Pre-computes the structure of the final answer to ensure completeness.
    """
    def create_plan(self, query_plan: Dict[str, Any]) -> Dict[str, Any]:
        intent = query_plan.get("intent", "fact_lookup")
        sections = query_plan.get("required_sections", [])
        original_query = query_plan.get("original_query", "")
        ql = (original_query or "").lower()

        technical_markers = (
            "precedence",
            "priority",
            "takes precedence",
            "which comes first",
            "order of",
            "body length",
            "nullification",
            "mechanism",
            "habeas corpus",
            "ex post facto",
            "supremacy clause",
            "http",
            "rfc",
            "header",
            "status code",
        )
        ordered_response = any(marker in ql for marker in technical_markers)
        legal_mechanism = any(marker in ql for marker in ("nullification", "habeas corpus", "ex post facto", "supremacy clause"))

        # If the query understanding layer didn't provide sections, we use defaults
        if not sections:
            if ordered_response:
                sections = ["Direct Answer", "Ordered Rule / Precedence", "Why This Is the Correct Order"]
            elif intent == "definition":
                sections = ["Core Definition", "Intuition / Analogy", "Common Examples"]
            elif intent == "comparison":
                sections = ["Definition of Concept A", "Definition of Concept B", "Key Differences", "When to use which"]
            elif intent == "troubleshooting":
                sections = ["Problem Identification", "Common Causes", "Step-by-Step Solution"]
            elif legal_mechanism:
                sections = ["Direct Answer", "Constitutional Mechanism", "Short Explanation"]
            else:
                sections = ["Direct Answer", "Supporting Details"]

        response_style = "ordered" if ordered_response else ("legal_mechanism" if legal_mechanism else "standard")

        return {
            "required_sections": sections,
            "tone": "objective and analytical",
            "formatting": "markdown with clear headings",
            "response_style": response_style,
            "original_query": original_query,
        }