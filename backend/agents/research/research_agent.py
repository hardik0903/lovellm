"""
ResearchAgent — Plan → Search → Evaluate → Synthesise → Gap-Check loop.

Old:  plan → search → scrape → one-shot synthesis.

New loop:
  Step 1 (Plan)      — decompose query into sub-questions + search queries
  Step 2 (Search)    — run searches, evaluate source credibility
  Step 3 (Evaluate)  — score and filter sources; flag conflicting claims
  Step 4 (Synthesise)— build a structured report section by section
  Step 5 (Gap-check) — reflect: are there sub-questions the report didn't answer?
                       if yes, run one targeted follow-up search and patch the gap
"""

import asyncio
import json
from typing import Dict, Any, AsyncGenerator, List
from logger import logger
from agent_base import BaseAgent, AgentMemory, AgentStep
from research_detector import ResearchDetector
from research_planner import ResearchPlanner
from source_evaluator import SourceEvaluator
from research_synthesizer import ResearchSynthesizer
from web_search import WebSearcher
from web_scraper import WebScraper


class ResearchAgent(BaseAgent):
    def __init__(self):
        self.detector = ResearchDetector()
        self.planner = ResearchPlanner()
        self.evaluator = SourceEvaluator()
        self.synthesizer = ResearchSynthesizer()
        self.searcher = WebSearcher()
        self.scraper = WebScraper()

    def detect(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.detector.detect(query, context)

    async def classify(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return await self.planner.plan(query)

    # ------------------------------------------------------------------ #
    # Internal tools                                                      #
    # ------------------------------------------------------------------ #

    async def _search_and_scrape(
        self, search_queries: List[str], threshold: float
    ) -> List[Dict[str, Any]]:
        """Run all search queries, deduplicate URLs, evaluate, scrape top sources."""
        all_sources = []
        for sq in search_queries:
            try:
                results = await self.searcher.search(sq, max_results=3)
                all_sources.extend(results)
            except Exception as e:
                logger.error(f"[ResearchAgent] Search failed for '{sq}': {e}")

        # Deduplicate
        seen, unique = set(), []
        for src in all_sources:
            url = src.get("url")
            if url and url not in seen:
                seen.add(url)
                unique.append(src)

        # Evaluate credibility
        evaluated = self.evaluator.evaluate(unique, "")
        top = sorted(
            [s for s in evaluated if s.get("credibility_score", 0) >= threshold],
            key=lambda x: x.get("credibility_score", 0), reverse=True
        )[:5]

        if not top:
            return []

        # Scrape
        tasks = [self.scraper.scrape(s["url"]) for s in top]
        scraped = await asyncio.gather(*tasks, return_exceptions=True)

        valid = []
        for src, res in zip(top, scraped):
            if isinstance(res, Exception):
                continue
            if res.get("success"):
                src["text"] = res["text"]
                valid.append(src)
        return valid

    def _check_gaps(self, sub_questions: List[str], report: Dict[str, Any]) -> List[str]:
        """
        Reflect: which sub-questions are not addressed in the report sections?
        Simple keyword check — fast, no LLM needed.
        """
        report_text = json.dumps(report).lower()
        missing = []
        for sq in sub_questions:
            keywords = [w for w in sq.lower().split() if len(w) > 4]
            if not any(kw in report_text for kw in keywords):
                missing.append(sq)
        return missing

    # ------------------------------------------------------------------ #
    # Main agent loop                                                     #
    # ------------------------------------------------------------------ #

    async def solve(self, query: str, context: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        memory = AgentMemory(query)
        agent_name = "research"

        try:
            # ================================================================
            # STEP 1 — PLAN
            # ================================================================
            yield self._thinking_event(agent_name, "Planning research strategy…", step=1)
            plan = await self.planner.plan(query)

            search_queries = plan.get("search_queries", [query]) or [query]
            sub_questions = plan.get("subtopics", [])
            threshold = plan.get("credibility_threshold", 0.5)

            step1 = AgentStep(
                step_num=1,
                thought="Decompose the research question into sub-topics and search queries.",
                action="plan",
                observation=(
                    f"{len(search_queries)} search queries planned. "
                    f"{len(sub_questions)} sub-questions to cover."
                ),
                result={"search_queries": search_queries, "sub_questions": sub_questions},
            )
            memory.add_step(step1)
            yield self._step_event(agent_name, step1)

            # ================================================================
            # STEP 2 — SEARCH + SCRAPE
            # ================================================================
            yield self._thinking_event(
                agent_name,
                f"Running {len(search_queries)} search queries…",
                step=2,
            )
            valid_sources = await self._search_and_scrape(search_queries, threshold)

            step2 = AgentStep(
                step_num=2,
                thought="Execute searches, evaluate source credibility, scrape top sources.",
                action="search_and_scrape",
                observation=f"{len(valid_sources)} credible sources retrieved.",
                result={"source_count": len(valid_sources), "urls": [s.get("url") for s in valid_sources]},
            )
            memory.add_step(step2)
            yield self._step_event(agent_name, step2)

            reflect2 = self._reflect(memory, {"ok": len(valid_sources) > 0})
            if len(valid_sources) == 0:
                yield self._reflection_event(agent_name, "escalate", "No credible sources found.")
                yield self._final_event(
                    mode=agent_name,
                    answer="Could not find credible sources for this research query.",
                    sources=[],
                    confidence="low",
                    memory=memory,
                )
                return

            yield self._reflection_event(agent_name, "continue", f"{len(valid_sources)} sources ready for synthesis.")

            # ================================================================
            # STEP 3 — SYNTHESISE
            # ================================================================
            yield self._thinking_event(agent_name, "Synthesising research report…", step=3)
            report = await self.synthesizer.synthesize(plan, valid_sources)

            step3 = AgentStep(
                step_num=3,
                thought="Build a structured research report from the scraped sources.",
                action="synthesise",
                observation=f"Report has {len(report.get('sections', []))} sections. "
                            f"Conflicting claims: {len(report.get('conflicting_claims', []))}.",
                result={
                    "sections": len(report.get("sections", [])),
                    "conflicts": len(report.get("conflicting_claims", [])),
                },
            )
            memory.add_step(step3)
            yield self._step_event(agent_name, step3)

            # ================================================================
            # STEP 4 — REFLECT: Gap check
            # ================================================================
            yield self._thinking_event(agent_name, "Checking for unanswered sub-questions…", step=4)
            gaps = self._check_gaps(sub_questions, report)

            step4 = AgentStep(
                step_num=4,
                thought="Verify that each planned sub-question is addressed in the report.",
                action="gap_check",
                observation=(
                    "All sub-questions covered." if not gaps
                    else f"Gaps found: {gaps}. Running follow-up search."
                ),
                result={"gaps": gaps},
            )
            memory.add_step(step4)
            yield self._step_event(agent_name, step4)
            yield self._reflection_event(
                agent_name,
                "stop" if not gaps else "patch",
                step4.observation,
            )

            # ================================================================
            # STEP 5 (conditional) — PATCH GAPS with follow-up search
            # ================================================================
            if gaps:
                yield self._thinking_event(
                    agent_name,
                    f"Running follow-up search to fill {len(gaps)} gap(s)…",
                    step=5,
                )
                gap_queries = [f"{query} {g}" for g in gaps[:2]]  # max 2 follow-up searches
                patch_sources = await self._search_and_scrape(gap_queries, max(0.3, threshold - 0.2))

                if patch_sources:
                    patch_report = await self.synthesizer.synthesize(
                        {"subtopics": gaps, "search_queries": gap_queries},
                        patch_sources,
                    )
                    # Merge patch sections into main report
                    report.setdefault("sections", []).extend(patch_report.get("sections", []))
                    report.setdefault("knowledge_gaps", [])
                    report["knowledge_gaps"] = [
                        g for g in report["knowledge_gaps"] if g not in gaps
                    ]

                step5 = AgentStep(
                    step_num=5,
                    thought="Perform targeted follow-up searches for unanswered sub-questions.",
                    action="patch_gaps",
                    observation=f"Patch search returned {len(patch_sources)} sources. Gaps merged into report.",
                    result={"patch_sources": len(patch_sources)},
                )
                memory.add_step(step5)
                yield self._step_event(agent_name, step5)

            # ================================================================
            # Final answer
            # ================================================================
            answer_text = (
                report.get("summary")
                or report.get("executive_summary")
                or "Research report generated."
            )
            sources = [
                {"title": s.get("title", ""), "url": s.get("url", ""), "type": "web"}
                for s in valid_sources
            ]

            yield {"event": "delta", "data": json.dumps({"text": answer_text})}
            yield self._final_event(
                mode=agent_name,
                answer=answer_text,
                display=report,
                sources=sources,
                memory=memory,
            )

        except Exception as e:
            logger.error(f"[ResearchAgent] Unhandled error: {e}")
            yield self._error_event("research", str(e))