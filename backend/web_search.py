from typing import List, Dict, Any
import warnings
from duckduckgo_search import DDGS
import wikipedia
from logger import logger

wikipedia.set_user_agent("lovellm_assistant/1.0 (test@example.com)")

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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=max_results):
                        results.append({
                            "title": r.get("title", ""),
                            "url": r.get("href", ""),
                            "snippet": r.get("body", "")
                        })
        except Exception as e:
            logger.error(f"Error during DuckDuckGo web search: {e}")
            
        if not results:
            logger.info("DuckDuckGo returned 0 results. Falling back to Wikipedia API.")
            try:
                wiki_results = wikipedia.search(query, results=max_results)
                for title in wiki_results:
                    try:
                        page = wikipedia.page(title, auto_suggest=False)
                        results.append({
                            "title": page.title,
                            "url": page.url,
                            "snippet": page.summary[:300] + "..."
                        })
                    except Exception as e_page:
                        logger.warning(f"Failed to fetch wikipedia page for {title}: {e_page}")
            except Exception as e_wiki:
                logger.error(f"Error during Wikipedia search: {e_wiki}")
                
        logger.info(f"Found {len(results)} search results.")
        return results
