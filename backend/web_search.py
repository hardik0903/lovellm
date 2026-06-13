from typing import List, Dict, Any
import warnings
from ddgs import DDGS
import wikipedia
from logger import logger
import asyncio

wikipedia.set_user_agent("lovellm_assistant/1.0 (test@example.com)")

class WebSearcher:
    def __init__(self):
        pass

    async def search(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the web using DuckDuckGo and returns top results.
        Returns a list of dicts with 'title', 'url', and 'snippet'.
        """
        logger.info(f"Performing web search for query: {query}")
        results = []
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                def _do_ddgs():
                    with DDGS() as ddgs:
                        return list(ddgs.text(query, max_results=max_results))
                raw_results = await asyncio.to_thread(_do_ddgs)
                for r in raw_results:
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
                def _do_wiki_search():
                    return wikipedia.search(query, results=max_results)
                wiki_results = await asyncio.to_thread(_do_wiki_search)
                
                for title in wiki_results:
                    try:
                        def _do_wiki_page(t=title):
                            return wikipedia.page(t, auto_suggest=False)
                        page = await asyncio.to_thread(_do_wiki_page)
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
