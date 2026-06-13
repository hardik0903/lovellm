import json
import asyncio
from typing import AsyncGenerator, Dict, Any, List
from web_search import WebSearcher
from web_scraper import WebScraper
from chunking import DocumentChunker
from generator import AnswerGenerator
from logger import logger

class WebAnswerer:
    def __init__(self, generator: AnswerGenerator):
        self.searcher = WebSearcher()
        self.scraper = WebScraper()
        self.chunker = DocumentChunker()
        self.generator = generator

    def _extract_direct_answer(self, query: str, text: str, snippet: str) -> tuple[str, str]:
        """
        Improved heuristic to extract an answer without using an LLM.
        Returns a tuple of (answer_text, confidence).
        """
        if not text:
            return snippet, "low"
            
        keywords = [word.lower() for word in query.split() if len(word) > 3]
        query_lower = query.lower()
        
        paragraphs = text.split('\n')
        best_p = None
        best_score = 0
        
        for p in paragraphs:
            p_lower = p.lower()
            words = p.split()
            if len(words) < 8 or len(words) > 100:
                continue
                
            score = sum(1 for kw in keywords if kw in p_lower)
            
            # Boost score if paragraph starts with definition patterns
            if " is " in p_lower[:30] or " are " in p_lower[:30]:
                score += 2
                
            if score > best_score:
                best_score = score
                best_p = p
                
        if best_p and best_score >= max(1, len(keywords) // 2):
            confidence = "high" if best_score >= len(keywords) else "medium"
            return best_p[:500] + ("..." if len(best_p) > 500 else ""), confidence
            
        return snippet, "low"

    async def answer_direct_web(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        logger.info(f"Executing direct_web for query: {query}")
        
        # 1. Search
        results = self.searcher.search(query, max_results=1)
        if not results:
            fallback = {
                "mode": "direct_web",
                "answer": "I could not find any web results for that query.",
                "sources": [],
                "confidence": "low",
                "needs_clarification": False
            }
            yield {"event": "final", "data": json.dumps(fallback)}
            return
            
        top_result = results[0]
        
        # 2. Scrape
        scraped = await self.scraper.scrape(top_result["url"])
        
        # 3. Extract answer directly
        extracted_answer, extraction_confidence = self._extract_direct_answer(query, scraped["text"], top_result["snippet"])
        
        # 4. Stream response (mimic streaming for immediate UI rendering)
        yield {
            "event": "delta",
            "data": json.dumps({"text": extracted_answer})
        }
        
        final_json = {
            "mode": "direct_web",
            "answer": extracted_answer,
            "sources": [
                {
                    "title": top_result["title"],
                    "url": top_result["url"],
                    "type": "web"
                }
            ],
            "confidence": extraction_confidence,
            "needs_clarification": False
        }
        
        yield {
            "event": "final",
            "data": json.dumps(final_json)
        }

    async def answer_web_rag(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        logger.info(f"Executing web_rag for query: {query}")
        
        # 1. Search
        results = self.searcher.search(query, max_results=3)
        if not results:
            fallback = {
                "mode": "web_rag",
                "answer": "I could not find any web results to synthesize an answer.",
                "sources": [],
                "confidence": "low",
                "needs_clarification": False
            }
            yield {"event": "final", "data": json.dumps(fallback)}
            return

        # 2. Scrape concurrently
        tasks = [self.scraper.scrape(r["url"]) for r in results]
        scraped_results = await asyncio.gather(*tasks)
        
        # 3. Chunk
        context_chunks = []
        source_metadata_map = {} # to keep track of title/url for generator
        
        for i, (search_res, scrape_res) in enumerate(zip(results, scraped_results)):
            if scrape_res["success"]:
                doc_id = f"web_{i}"
                source_metadata_map[doc_id] = {
                    "title": search_res["title"],
                    "url": search_res["url"],
                    "type": "web"
                }
                
                # Chunk the scraped text
                chunks = self.chunker.chunk_document(
                    text=scrape_res["text"], 
                    document_id=doc_id, 
                    metadata={"source_file": search_res["url"], "is_web": True}
                )
                context_chunks.extend(chunks)

        if not context_chunks:
            # Fallback if scraping failed for all
            fallback = {
                "mode": "web_rag",
                "answer": "I found web results but could not extract readable content from them.",
                "sources": [],
                "confidence": "low",
                "needs_clarification": False
            }
            yield {"event": "final", "data": json.dumps(fallback)}
            return
            
        # We limit the chunks to the top N so we don't blow up the context window.
        # Since these aren't reranked against the query (just top web results), we just take the first ~15 chunks.
        top_chunks = context_chunks[:15]
        
        # 4. Generate via Groq
        # We pass the mode "web_rag" to the generator, along with the source map so it can construct citations.
        async for event in self.generator.generate_stream(query, top_chunks, mode="web_rag", source_map=source_metadata_map):
            yield event
