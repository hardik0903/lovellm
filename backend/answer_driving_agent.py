import json
import asyncio
import re
import string
from typing import AsyncGenerator, Dict, Any, List
from web_search import WebSearcher
from web_scraper import WebScraper
from logger import logger

class AnswerDrivingAgent:
    """
    A deterministic, non-LLM agent that drives the direct_web fast path.
    It searches multiple candidates, filters out bad sources, extracts passages,
    and returns a clean, highly confident definitional sentence, or escalates to Groq.
    """
    def __init__(self):
        self.searcher = WebSearcher()
        self.scraper = WebScraper()

    def filter_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid = []
        for c in candidates:
            url = c.get("url", "")
            # Skip invalid URLs and known tracking/redirect links
            if not url.startswith("http") or "/clev?event=" in url:
                continue
            valid.append(c)
        return valid

    def extract_and_rank_passages(self, query: str, text: str) -> tuple[str, int]:
        """
        Splits text into sentences, rejects fragments, and scores them.
        Returns the best passage and its score.
        """
        if not text:
            return "", 0
            
        keywords = [word.lower() for word in query.split() if len(word) > 3]
        query_lower = query.lower()
        
        # Identify the core concept by stripping common question words
        concept = query_lower
        for prefix in ["what is a ", "what is an ", "what is ", "who is ", "define "]:
            if query_lower.startswith(prefix):
                concept = query_lower[len(prefix):].strip("? ")
                break

        # Normalize whitespace (replace newlines with spaces)
        text_normalized = re.sub(r'\s+', ' ', text)
        sentences = re.split(r'(?<=[.!?])\s+', text_normalized)
        
        best_idx = -1
        best_score = 0
        
        for i, s in enumerate(sentences):
            s_clean = ''.join(c for c in s if c.isprintable()).strip()
            if not s_clean: continue
            
            first_alpha = next((c for c in s_clean if c.isalpha()), None)
            
            # Reject mid-clause fragments
            if (first_alpha and first_alpha.islower()) or s_clean[0] in ",;:)}]":
                continue
                
            words = s_clean.split()
            if len(words) < 8 or len(words) > 60:
                continue
                
            s_lower = s_clean.lower()
            
            # Penalize sentences that are just other questions
            if s_clean.endswith("?") and (s_lower.startswith("what") or s_lower.startswith("how")):
                continue
                
            # Require the core concept to be present
            if concept and concept not in s_lower:
                continue
                
            score = sum(1 for kw in keywords if kw in s_lower)
            
            # Boost score heavily for explicit definitional structures
            if s_lower.startswith(concept + " is ") or s_lower.startswith("an " + concept + " is ") or s_lower.startswith("a " + concept + " is "):
                score += 10
            elif " is a " in s_lower or " is an " in s_lower or " are " in s_lower:
                score += 3
                
            if score > best_score:
                best_score = score
                best_idx = i
                
        if best_idx != -1 and best_score >= 3:
            extracted_sentences = [sentences[best_idx].strip()]
            # Grab up to 1 more sentence for completeness
            if best_idx + 1 < len(sentences):
                next_s = sentences[best_idx + 1].strip()
                if next_s and next_s[0].isalpha() and next_s[0].isupper() and 5 <= len(next_s.split()) <= 40:
                    extracted_sentences.append(next_s)
                    
            final_text = " ".join(extracted_sentences)
            if final_text[-1] not in ".!?":
                final_text += "."
            return final_text, best_score
            
        return "", 0

    async def answer(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        logger.info(f"[AGENT] AnswerDrivingAgent started for: {query}")
        
        candidates = self.searcher.search(query, max_results=5)
        candidates = self.filter_candidates(candidates)
        
        if not candidates:
            yield {"event": "final", "data": json.dumps({"mode": "direct_web", "answer": "No valid search results found.", "confidence": "low", "route": "web_rag"})}
            return
            
        best_answer = None
        best_source = None
        highest_score = 0
        
        # We check the top 3 valid candidates
        for result in candidates[:3]:
            scraped = await self.scraper.scrape(result["url"])
            if not scraped["success"]:
                continue
                
            passage, score = self.extract_and_rank_passages(query, scraped["text"])
            
            if score > highest_score:
                highest_score = score
                best_answer = passage
                best_source = result
                
            # If we found a highly definitive sentence, stop early
            if score >= 10:
                break
                
        if best_answer and highest_score >= 5:
            confidence = "high" if highest_score >= 10 else "medium"
            
            yield {
                "event": "delta",
                "data": json.dumps({"text": best_answer})
            }
            
            yield {
                "event": "final",
                "data": json.dumps({
                    "mode": "direct_web",
                    "answer": best_answer,
                    "sources": [
                        {
                            "title": best_source["title"],
                            "url": best_source["url"],
                            "type": "web"
                        }
                    ],
                    "confidence": confidence,
                    "needs_clarification": False
                })
            }
        else:
            yield {
                "event": "final",
                "data": json.dumps({
                    "mode": "direct_web",
                    "answer": "Could not deterministically extract a reliable answer.",
                    "confidence": "low"
                })
            }
