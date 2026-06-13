import json
import asyncio
from typing import AsyncGenerator, Dict, Any

from logger import logger
from query_understanding import QueryUnderstandingEngine
from search_planner import SearchPlanner
from retrieval_memory import RetrievalMemory
from self_correction import MultiPassRetriever
from answer_planner import AnswerPlanner
from answer_verifier import AnswerVerifier
from conversation_context import ConversationContext
from generator import AnswerGenerator
from web_search import WebSearcher
from web_scraper import WebScraper
from chunking import DocumentChunker
from rank_bm25 import BM25Okapi
import re

def _bm25_rank_chunks(query: str, chunks: list, top_k: int = 20) -> list:
    """Rank chunks by BM25 relevance to the query and return top_k."""
    if not chunks:
        return []
    
    def tokenize(text: str) -> list:
        return re.findall(r'\b\w+\b', text.lower())
        
    corpus = [tokenize(c.get("context_text", c.get("text", ""))) for c in chunks]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))
    
    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_indices = [i for i, score in scored[:top_k] if score > 0]
    
    if not top_indices:
        return chunks[:top_k]
        
    return [chunks[i] for i in top_indices]

class PipelineOrchestrator:
    def __init__(self, generator: AnswerGenerator, local_retriever):
        self.qu_engine = QueryUnderstandingEngine()
        self.search_planner = SearchPlanner()
        self.answer_planner = AnswerPlanner()
        self.answer_verifier = AnswerVerifier()
        self.context = ConversationContext()
        
        self.searcher = WebSearcher()
        self.scraper = WebScraper()
        self.chunker = DocumentChunker()
        
        self.web_retriever = MultiPassRetriever(self.searcher)
        self.local_retriever = local_retriever
        self.generator = generator

    async def execute(self, raw_query: str, has_documents: bool = False) -> AsyncGenerator[Dict[str, Any], None]:
        # 0. Conversation Context (Resolve entities)
        resolved_query = self.context.resolve_references(raw_query)
        logger.info(f"Resolved Query: {resolved_query}")
        
        # 1. Query Understanding
        query_plan = await self.qu_engine.understand(resolved_query, has_documents)
        
        # Update context
        self.context.update_context(query_plan)
        
        # 2. Search Planning
        search_plan = self.search_planner.plan(query_plan)
        
        mode = search_plan.get("mode", "direct_web")
        if mode == "clarify":
            yield {
                "event": "final",
                "data": json.dumps({
                    "mode": "clarify",
                    "answer": "Could you please clarify your question? I'm not entirely sure what you mean.",
                    "sources": [],
                    "confidence": "low",
                    "needs_clarification": True,
                    "interpretation": search_plan
                })
            }
            return

        # 3. Retrieval Memory setup
        memory = RetrievalMemory()
        
        # 4-7. Retrieval (Multi-pass web search or Local doc search)
        context_chunks = []
        source_metadata_map = {}
        
        if mode == "doc_rag":
            # For local documents, we just use the hybrid retriever once
            top_candidates = self.local_retriever.retrieve(resolved_query, top_k=5)
            if not top_candidates:
                yield {
                    "event": "final",
                    "data": json.dumps({
                        "mode": mode,
                        "answer": "I could not find any relevant information in the uploaded documents.",
                        "sources": [],
                        "confidence": "low",
                        "needs_clarification": False,
                        "interpretation": search_plan
                    })
                }
                return
            context_chunks = top_candidates
            # We don't have a source map for local docs built the same way, generator handles it natively
        else:
            selected_sources = await self.web_retriever.retrieve(search_plan, memory)
            
            if not selected_sources:
                 yield {
                    "event": "final",
                    "data": json.dumps({
                        "mode": mode,
                        "answer": "I could not find any reliable evidence to answer your query.",
                        "sources": [],
                        "confidence": "low",
                        "needs_clarification": False,
                        "interpretation": search_plan
                    })
                 }
                 return
                 
            # Scrape the selected sources
            tasks = [self.scraper.scrape(s["url"]) for s in selected_sources]
            scraped_results = await asyncio.gather(*tasks)
            
            for i, (src, scrape_res) in enumerate(zip(selected_sources, scraped_results)):
                if scrape_res["success"]:
                    doc_id = f"source_{i}"
                    source_metadata_map[doc_id] = {
                        "title": src.get("title", src["url"]),
                        "url": src["url"],
                        "type": "web"
                    }
                    chunks = self.chunker.chunk_document(
                        text=scrape_res["text"], 
                        document_id=doc_id, 
                        metadata={"source_file": src["url"], "is_web": True}
                    )
                    context_chunks.extend(chunks)
                    
        # 8. Answer Planning
        answer_plan = self.answer_planner.create_plan(search_plan)
        
        # 9. Groq Synthesis
        # We will wrap the generator to intercept the final JSON
        final_answer_obj = None
        
        # Pass answer plan into generator via prompt (in a real system we'd modify generator to take it)
        # We will inject the requested sections into the context chunks for now, or generator prompt.
        
        async for event in self.generator.generate_stream(resolved_query, _bm25_rank_chunks(resolved_query, context_chunks, top_k=20), mode=mode, source_map=source_metadata_map, answer_plan=answer_plan):
            if event["event"] == "final":
                try:
                    final_answer_obj = json.loads(event["data"])
                except:
                    final_answer_obj = {"answer": "Generation failed."}
            else:
                yield event
                
        if not final_answer_obj:
            return
            
        # 10. Answer Verification
        verified_obj = self.answer_verifier.verify(final_answer_obj, answer_plan)
        
        # Attach interpretation metadata for the UI
        verified_obj["interpretation"] = {
            "intent": search_plan.get("intent"),
            "rewritten_query": search_plan.get("normalized_query"),
            "mode": search_plan.get("mode")
        }
        
        # 11. Final Response
        yield {
            "event": "final",
            "data": json.dumps(verified_obj)
        }
