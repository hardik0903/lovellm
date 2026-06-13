import os
from typing import Dict, Any, List
from groq import AsyncGroq
from logger import logger

import re

class AnswerVerifier:
    """
    Post-generation verification to ensure the answer meets quality standards.
    Uses deterministic checks. For a true production system, you might use a fast LLM pass here.
    """
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"

    async def verify(self, answer_obj: Dict[str, Any], answer_plan: Dict[str, Any]) -> Dict[str, Any]:
        answer_text = answer_obj.get("answer", "").lower()
        
        # 1. Coverage Check
        sections = answer_plan.get("required_sections", [])
        missing_sections = []
        for section in sections:
            # Simple heuristic: check if key words from the section exist
            keywords = section.lower().split()
            if not any(kw in answer_text for kw in keywords if len(kw) > 3):
                missing_sections.append(section)
                
        # 2. Citation Coverage
        sources = answer_obj.get("sources", [])
        has_citations = False
        
        if answer_plan.get("mode") == "direct_web":
            has_citations = True
        elif not sources:
            has_citations = True # No sources to cite
        else:
            if re.search(r'\[\d+\]', answer_text):
                has_citations = True
            else:
                for src in sources:
                    title = src.get("title", "").lower()
                    url = src.get("url", "").lower()
                    if title and len(title) > 5 and title in answer_text:
                        has_citations = True
                        break
                    domain = url.replace("https://", "").replace("http://", "").split("/")[0]
                    if domain and len(domain) > 3 and domain in answer_text:
                        has_citations = True
                        break
        
        # 3. Completeness
        # Is the answer long enough given the intent?
        is_complete = True
        if len(answer_text.split()) < 20 and len(sections) > 2:
            is_complete = False
            
        # 4. Relevance (Compare against question)
        original_query = answer_plan.get("original_query", "").lower()
        concept = original_query
        for prefix in ["what is a ", "what is an ", "what is ", "who is ", "define "]:
            if original_query.startswith(prefix):
                concept = original_query[len(prefix):].strip("? ")
                break
                
        is_relevant = True
        # Simple heuristic: If the answer is extremely short and doesn't mention the core concept, it might be irrelevant.
        concept_stem = concept if concept else ""
        if concept_stem:
            if concept_stem.endswith('es') and not concept_stem.endswith('ses'):
                concept_stem = concept_stem[:-2]
            elif concept_stem.endswith('s') and not concept_stem.endswith('ss'):
                concept_stem = concept_stem[:-1]
        if concept_stem and len(concept_stem) >= 3 and concept_stem not in answer_text:
            is_relevant = False
            
        verification_result = {
            "passed": len(missing_sections) == 0 and (has_citations or len(sources) == 0) and is_complete and is_relevant,
            "missing_sections": missing_sections,
            "has_citations": has_citations,
            "is_complete": is_complete,
            "is_relevant": is_relevant
        }
        
        # If deterministic verification failed, do a tiny LLM pass to double check
        if not verification_result["passed"] and answer_text:
            logger.info("Deterministic verification failed, falling back to LLM verifier pass.")
            try:
                response = await self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an answer verification assistant. Answer strictly with YES or NO."},
                        {"role": "user", "content": f"Does the following answer provide a reasonable and relevant response to the query: '{original_query}'?\n\nAnswer: {answer_text}"}
                    ],
                    model=self.model,
                    temperature=0.0,
                    max_tokens=5
                )
                if "yes" in response.choices[0].message.content.lower():
                    logger.info("LLM verifier approved the answer.")
                    verification_result["passed"] = True
                    verification_result["llm_override"] = True
            except Exception as e:
                logger.error(f"Error in LLM verification pass: {e}")
                
        answer_obj["verification"] = verification_result
        return answer_obj
