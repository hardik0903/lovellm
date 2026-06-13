from typing import List, Dict, Any
from duckduckgo_search import DDGS
from logger import logger

class WebSearcher:
    def __init__(self):
        pass

    def search(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the web using DuckDuckGo and returns top results.
        Returns a list of dicts with 'title', 'url', and 'snippet'.
        """
        logger.info(f"Performing web search for query: {query}")
        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })
        except Exception as e:
            logger.error(f"Error during web search: {e}")
            
        logger.info(f"Found {len(results)} search results.")
        return results
