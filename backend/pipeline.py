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
from answer_driving_agent import AnswerDrivingAgent
from rank_bm25 import BM25Okapi
import re
from display_agent import DisplayFormattingAgent

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
        self.answer_driving_agent = AnswerDrivingAgent()
        self.display_agent = DisplayFormattingAgent()
        # In-process memory cache. Note: if server restarts or multiple workers are used, cache is lost.
        self.query_cache = {}

    async def execute(self, raw_query: str, mode: str = "auto", has_documents: bool = False) -> AsyncGenerator[Dict[str, Any], None]:
        # 0. Conversation Context (Resolve entities)
        resolved_query = self.context.resolve_references(raw_query)
        logger.info(f"[ROUTER] Resolved Query: {resolved_query}")
        
        # Determine Display Format & Ambiguity BEFORE query understanding
        display_context = self.display_agent.resolve_and_detect(
            raw_query=resolved_query,
            conversation_history=self.context.history_objects[-5:]
        )
        
        # Master Router Interception (Step 0.5)
        from master_router import master_router_instance
        
        # Prepare context for the router
        router_context = {
            "has_documents": has_documents,
            "has_data_file": has_documents, # Assuming data uploads are handled similarly for now
            "local_retriever": self.local_retriever,
            "display_context": display_context
        }
        
        # Master router is skipped when the user has explicitly forced a
        # document-grounded or web-grounded pipeline via `mode`, since those
        # overrides are handled later in the general pipeline and would
        # otherwise be hijacked by specialist agents (math/knowledge/etc.)
        if mode in ("doc", "web"):
            route_decision = {"selected_agent": None, "confidence": 0.0, "reasoning": f"Explicit mode override: {mode}"}
        else:
            route_decision = master_router_instance.route(resolved_query, router_context)
        confidence = route_decision["confidence"]
        
        if route_decision["selected_agent"] and confidence >= 0.5:
            agent_name = route_decision["selected_agent"]
            logger.info(f"[MASTER ROUTER] Dispatching to {agent_name} Agent with confidence {confidence:.2f}")
            
            # Fetch the agent from registry
            agent = master_router_instance.registry.get_agent(agent_name)
            if agent:
                uncertainty_flag = (0.5 <= confidence < 0.7)
                
                # We yield the agent's stream
                async for event in agent.solve(resolved_query, router_context):
                    # Attach uncertainty flag to final event if needed
                    if uncertainty_flag and event["event"] == "final":
                        try:
                            data = json.loads(event["data"])
                            data["uncertainty_flag"] = True
                            data["routed_agent"] = agent_name
                            event["data"] = json.dumps(data)
                        except:
                            pass
                            
                    # Inject routed_agent to all final events for the UI badge
                    if event["event"] == "final":
                        try:
                            data = json.loads(event["data"])
                            data["routed_agent"] = agent_name
                            event["data"] = json.dumps(data)
                        except:
                            pass
                            
                    yield event
                return
        else:
            logger.info(f"[MASTER ROUTER] Confidence too low ({confidence:.2f}), falling through to general pipeline.")
        # Use the disambiguated query for further downstream routing
        final_query = display_context["resolved_query"]
        logger.info(f"[ROUTER] Disambiguated Query: {final_query}")

        # Record the user's turn in history_objects
        self.context.history_objects.append({"role": "user", "content": final_query})

        # Check cache
        norm_q = self.qu_engine._normalize_query(final_query)
        if norm_q in self.query_cache:
            logger.info(f"[CACHE] Cache hit for query: {norm_q}")
            yield self.query_cache[norm_q]
            return
            
        # 1. Query Understanding
        query_plan = await self.qu_engine.understand(final_query, norm_q, has_documents)
        
        # Update context
        self.context.update_context(query_plan)
        
        # 2. Search Planning
        search_plan = self.search_planner.plan(query_plan)
        
        # Override mode if explicit
        if mode != "auto":
            if mode == "doc":
                search_plan["mode"] = "doc_rag"
            elif mode == "web":
                search_plan["mode"] = "direct_web"
            else:
                search_plan["mode"] = mode
                
        plan_mode = search_plan.get("mode", "direct_web")
        is_complex = search_plan.get("is_complex", False)
        intent = search_plan.get("intent", "unknown")
        
        logger.info(f"[ROUTER] Intent: {intent} | Complex: {is_complex} | Target Route: {plan_mode}")

        if plan_mode == "clarify":
            logger.info(f"[ROUTER] Route executed: clarify | used_groq: True | confidence: low")
            final_event = {
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
            self.query_cache[norm_q] = final_event
            yield final_event
            return

        # =========================================================
        # FAST PATH (direct_web)
        # =========================================================
        if plan_mode == "direct_web":
            # Buffer the direct web generator to check confidence before yielding
            direct_events = []
            final_direct_obj = None
            async for event in self.answer_driving_agent.answer(final_query):
                direct_events.append(event)
                if event["event"] == "final":
                    try:
                        final_direct_obj = json.loads(event["data"])
                    except:
                        pass
            
            if final_direct_obj and final_direct_obj.get("confidence") in ["high", "medium"]:
                logger.info(f"[ROUTER] Route executed: direct_web | used_groq: False | confidence: {final_direct_obj.get('confidence')}")
                final_direct_obj["interpretation"] = {
                    "intent": intent,
                    "rewritten_query": search_plan.get("normalized_query"),
                    "mode": "direct_web"
                }
                direct_events[-1]["data"] = json.dumps(final_direct_obj)
                self.query_cache[norm_q] = direct_events[-1]
                for event in direct_events:
                    yield event
                return
            else:
                logger.info("[ROUTER] Fast path extraction confidence was low. Escalating to web_rag.")
                plan_mode = "web_rag"
                search_plan["mode"] = "web_rag"

        # =========================================================
        # COMPLEX PATH (doc_rag or web_rag)
        # =========================================================

        # 3. Retrieval Memory setup
        memory = RetrievalMemory()
        
        # 4-7. Retrieval (Multi-pass web search or Local doc search)
        context_chunks = []
        source_metadata_map = {}
        
        if plan_mode == "doc_rag":
            # For local documents, we just use the hybrid retriever once
            top_candidates = self.local_retriever.retrieve(final_query, top_k=5)
            if not top_candidates:
                final_event = {
                    "event": "final",
                    "data": json.dumps({
                        "mode": plan_mode,
                        "answer": "I could not find any relevant information in the uploaded documents.",
                        "sources": [],
                        "confidence": "low",
                        "needs_clarification": False,
                        "interpretation": search_plan
                    })
                }
                self.query_cache[norm_q] = final_event
                yield final_event
                return
            context_chunks = top_candidates
            # We don't have a source map for local docs built the same way, generator handles it natively
        else:
            selected_sources = await self.web_retriever.retrieve(search_plan, memory)
            
            if intent == "definition" and selected_sources:
                selected_sources = selected_sources[:1]
                
            if not selected_sources:
                 final_event = {
                    "event": "final",
                    "data": json.dumps({
                        "mode": plan_mode,
                        "answer": "I could not find any reliable evidence to answer your query.",
                        "sources": [],
                        "confidence": "low",
                        "needs_clarification": False,
                        "interpretation": search_plan
                    })
                 }
                 self.query_cache[norm_q] = final_event
                 yield final_event
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
        answer_plan["original_query"] = final_query
        answer_plan["mode"] = plan_mode
        
        # 9. Groq Synthesis
        # We will wrap the generator to intercept the final JSON
        final_answer_obj = None
        
        display_injection = self.display_agent.get_prompt_injection(display_context)
        
        async for event in self.generator.generate_stream(final_query, _bm25_rank_chunks(final_query, context_chunks, top_k=20), mode=plan_mode, source_map=source_metadata_map, answer_plan=answer_plan, display_injection=display_injection):
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
        verified_obj = await self.answer_verifier.verify(final_answer_obj, answer_plan)
        
        # Apply Display Formatting (exclude direct_web)
        if plan_mode != "direct_web":
            verified_obj = self.display_agent.process(verified_obj, display_context)
            
        # Attach interpretation metadata for the UI
        verified_obj["interpretation"] = {
            "intent": search_plan.get("intent"),
            "rewritten_query": search_plan.get("normalized_query"),
            "mode": search_plan.get("mode")
        }
        
        # Record assistant response in history
        assistant_turn = {"role": "assistant"}
        if "display" in verified_obj:
            assistant_turn["display"] = verified_obj["display"]
        else:
            assistant_turn["content"] = verified_obj.get("answer", "")
        self.context.history_objects.append(assistant_turn)
        
        final_confidence = verified_obj.get("confidence", "unknown")
        logger.info(f"[ROUTER] Route executed: {plan_mode} | used_groq: True | confidence: {final_confidence}")
        
        # 11. Final Response
        final_event = {
            "event": "final",
            "data": json.dumps(verified_obj)
        }
        self.query_cache[norm_q] = final_event
        yield final_event