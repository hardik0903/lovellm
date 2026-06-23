import _path_setup  # noqa: F401
import json
import asyncio
import time
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
from context_packer import AdaptiveContextPacker

# FIX (#5): the query cache previously had no TTL and no invalidation hook.
# A stale answer for a normalized query string could be served forever
# within a worker's lifetime, even after the underlying corpus changed via
# re-ingestion, edits, or deletion. CACHE_TTL_SECONDS puts a hard ceiling on
# staleness even if nobody remembers to call invalidate_cache(); the
# generation counter (see invalidate_cache()) gives ingestion an explicit,
# immediate way to blow away every cached answer the moment new documents
# land, rather than waiting out the TTL.
CACHE_TTL_SECONDS = 15 * 60  # 15 minutes

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
        # Pass retriever + generator callbacks so AnswerVerifier can run the
        # Self-RAG faithfulness loop for doc_rag queries.
        self.answer_verifier = AnswerVerifier(
            retriever=None,   # set after local_retriever is stored below
            generator=None,   # set after generator is stored below
        )
        self.context = ConversationContext()
        
        self.searcher = WebSearcher()
        self.scraper = WebScraper()
        self.chunker = DocumentChunker()
        
        self.web_retriever = MultiPassRetriever(self.searcher)
        self.local_retriever = local_retriever
        self.generator = generator
        self.answer_driving_agent = AnswerDrivingAgent()
        self.display_agent = DisplayFormattingAgent()
        self.context_packer = AdaptiveContextPacker()
        # Wire Self-RAG callbacks now that both retriever + generator are assigned.
        self.answer_verifier._retriever = self._doc_retriever_callback
        self.answer_verifier._generator = self.generator.generate_stream
        # In-process memory cache. Note: if server restarts or multiple workers are used, cache is lost.
        # Each entry is now (event, cached_at_ts, corpus_generation) instead
        # of a bare event -- see _cache_get / _cache_put / invalidate_cache.
        self.query_cache: Dict[str, Any] = {}
        # Bumped by invalidate_cache() (called after any document
        # upload/edit/delete). Any cache entry stamped with an older
        # generation is treated as a miss, regardless of its TTL.
        self._corpus_generation = 0

    def invalidate_cache(self) -> None:
        """Call this after ingesting, editing, or deleting any document.

        Cheapest correct option: bump the generation counter so every
        existing cache entry is treated as stale on next lookup, without
        having to walk and delete every key (which matters once the cache
        has many entries across many users/sessions in a single worker).
        """
        self._corpus_generation += 1
        logger.info(f"[CACHE] Invalidated (corpus_generation={self._corpus_generation}); "
                    f"{len(self.query_cache)} stale entries will be skipped on next lookup.")

    def _cache_get(self, norm_q: str):
        entry = self.query_cache.get(norm_q)
        if entry is None:
            return None
        event, cached_at, generation = entry
        if generation != self._corpus_generation:
            logger.info(f"[CACHE] Stale entry for '{norm_q}' (corpus changed since caching), treating as miss.")
            del self.query_cache[norm_q]
            return None
        if (time.time() - cached_at) > CACHE_TTL_SECONDS:
            logger.info(f"[CACHE] Expired entry for '{norm_q}' (TTL={CACHE_TTL_SECONDS}s), treating as miss.")
            del self.query_cache[norm_q]
            return None
        return event

    def _cache_put(self, norm_q: str, event) -> None:
        self.query_cache[norm_q] = (event, time.time(), self._corpus_generation)

    def _doc_retriever_callback(self, query: str, top_k: int = 7) -> list:
        """Thin wrapper so AnswerVerifier can call local_retriever.retrieve()
        with retrieval_complexity drawn from the last query_plan, enabling
        the Self-RAG retry to use the same routing the original call used."""
        complexity = getattr(self, "_last_retrieval_complexity", None)
        use_summary_nodes = complexity in ("simple", "multi_hop", "global")
        try:
            return self.local_retriever.retrieve(
                query,
                top_k=top_k,
                complexity=complexity,
                use_summary_nodes=use_summary_nodes,
            )
        except TypeError:
            # Backward-compatible fallback for older retriever implementations.
            return self.local_retriever.retrieve(query, top_k=top_k, complexity=complexity)
        
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
                # FIX (#10): margin-based ambiguity (see master_router.py) is
                # a different signal from the absolute-confidence band above
                # -- it fires whenever the runner-up agent was close behind,
                # regardless of where the winner's own score falls. A
                # margin-thin win was decided more by priority_order than by
                # the detectors actually disagreeing strongly, and the user
                # deserves the same "this routing decision was uncertain"
                # signal in that case as in the absolute-confidence case.
                margin_ambiguous_flag = bool(route_decision.get("ambiguous"))
                
                # We yield the agent's stream
                async for event in agent.solve(resolved_query, router_context):
                    # Attach uncertainty flag to final event if needed
                    if (uncertainty_flag or margin_ambiguous_flag) and event["event"] == "final":
                        try:
                            data = json.loads(event["data"])
                            if uncertainty_flag:
                                data["uncertainty_flag"] = True
                            if margin_ambiguous_flag:
                                data["margin_ambiguous_flag"] = True
                                data["runner_up_agent"] = route_decision.get("runner_up_agent")
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

        # Check cache (TTL + corpus-generation aware -- see _cache_get)
        norm_q = self.qu_engine._normalize_query(final_query)
        cached_event = self._cache_get(norm_q)
        if cached_event is not None:
            logger.info(f"[CACHE] Cache hit for query: {norm_q}")
            yield cached_event
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
            self._cache_put(norm_q, final_event)
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
                self._cache_put(norm_q, direct_events[-1])
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
            # Pass retrieval_complexity from query_plan so the retriever can gate
            # strategy (simple=fast dense, global=RAPTOR summaries, multi_hop=full).
            retrieval_complexity = query_plan.get("retrieval_complexity")
            self._last_retrieval_complexity = retrieval_complexity  # for Self-RAG retry callback
            logger.info(f"[DOC RAG] retrieval_complexity={retrieval_complexity!r}")
            top_candidates = self.local_retriever.retrieve(
                final_query, top_k=5, complexity=retrieval_complexity
            )
            # FIX (visibility): a BM25 or dense arm can fail silently inside
            # the retriever and still produce a plausible-looking result set
            # from the surviving arm alone. Surface that here so the final
            # answer's confidence reflects degraded evidence quality instead
            # of looking identical to a fully healthy retrieval.
            self._last_retrieval_degraded = bool(
                getattr(getattr(self.local_retriever, "_inner", self.local_retriever), "_last_search_pair_failures", None)
            )
            if self._last_retrieval_degraded:
                logger.error(
                    "[DOC RAG] Retrieval was degraded (BM25 and/or dense arm failed) for query: %s",
                    final_query,
                )
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
                self._cache_put(norm_q, final_event)
                yield final_event
                return
            context_chunks = top_candidates
            # We don't have a source map for local docs built the same way, generator handles it natively
        else:
            self._last_retrieval_degraded = False
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
                 self._cache_put(norm_q, final_event)
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
        packed_context_chunks = self.context_packer.pack(
            final_query,
            context_chunks,
            answer_plan=answer_plan,
            mode=plan_mode,
        )
        
        async for event in self.generator.generate_stream(final_query, packed_context_chunks, mode=plan_mode, source_map=source_metadata_map, answer_plan=answer_plan, display_injection=display_injection):
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
        verified_obj = await self.answer_verifier.verify(
            final_answer_obj, answer_plan,
            retrieved_chunks=packed_context_chunks if plan_mode == "doc_rag" else None,
        )
        
        # Apply Display Formatting (exclude direct_web)
        if plan_mode != "direct_web":
            verified_obj = self.display_agent.process(verified_obj, display_context)

        # FIX (visibility): if retrieval ran in degraded mode (BM25 and/or
        # dense arm failed and we're only seeing results from the surviving
        # arm), don't let the final answer look identical to a fully healthy
        # retrieval. Downgrade confidence and flag it explicitly so the UI
        # and any downstream eval can distinguish "found nothing because
        # nothing exists" from "found less than we should have because part
        # of the pipeline broke".
        if getattr(self, "_last_retrieval_degraded", False):
            verified_obj["degraded_retrieval"] = True
            if verified_obj.get("confidence") == "high":
                verified_obj["confidence"] = "medium"
            elif verified_obj.get("confidence") == "medium":
                verified_obj["confidence"] = "low"

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
        self._cache_put(norm_q, final_event)
        yield final_event