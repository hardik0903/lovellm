import json
import asyncio
from typing import Dict, Any, AsyncGenerator
from logger import logger
from agent_base import BaseAgent
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
        # Research planner serves as our classifier
        return await self.planner.plan(query)

    async def solve(self, query: str, context: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            plan = await self.classify(query)

            yield {
                "event": "research_thinking",
                "data": json.dumps({"status": "Planning research approach..."})
            }

            search_queries = plan.get("search_queries", [query]) or [query]

            yield {
                "event": "research_thinking",
                "data": json.dumps({"status": f"Running {len(search_queries)} search queries..."})
            }

            all_sources = []
            for sq in search_queries:
                try:
                    results = await self.searcher.search(sq, max_results=3)
                    all_sources.extend(results)
                except Exception as e_search:
                    logger.error(f"[ResearchAgent] Search failed for '{sq}': {e_search}")

            # Deduplicate sources by URL
            seen = set()
            unique_sources = []
            for src in all_sources:
                url = src.get("url")
                if url and url not in seen:
                    seen.add(url)
                    unique_sources.append(src)

            evaluated_sources = self.evaluator.evaluate(unique_sources, query)

            # Filter sources based on threshold
            threshold = plan.get("credibility_threshold", 0.5)
            top_sources = [s for s in evaluated_sources if s.get("credibility_score", 0.0) >= threshold]

            # Take up to top 5
            top_sources = sorted(top_sources, key=lambda x: x.get("credibility_score", 0.0), reverse=True)[:5]

            if not top_sources:
                yield {
                    "event": "final",
                    "data": json.dumps({
                        "mode": "research",
                        "answer": "Could not find any highly credible sources for this topic.",
                        "sources": [],
                        "confidence": "low",
                        "needs_clarification": False,
                        "display": None
                    })
                }
                return

            yield {
                "event": "research_thinking",
                "data": json.dumps({"status": f"Scraping {len(top_sources)} sources..."})
            }

            tasks = [self.scraper.scrape(s["url"]) for s in top_sources]
            scraped_results = await asyncio.gather(*tasks, return_exceptions=True)

            valid_sources = []
            for src, scrape_res in zip(top_sources, scraped_results):
                if isinstance(scrape_res, Exception):
                    logger.error(f"[ResearchAgent] Scrape failed for {src.get('url')}: {scrape_res}")
                    continue
                if scrape_res.get("success"):
                    src["text"] = scrape_res["text"]
                    valid_sources.append(src)

            if not valid_sources:
                yield {
                    "event": "final",
                    "data": json.dumps({
                        "mode": "research",
                        "answer": "Found sources for this topic but could not retrieve their content.",
                        "sources": [{"title": s.get("title", ""), "url": s.get("url", ""), "type": "web"} for s in top_sources],
                        "confidence": "low",
                        "needs_clarification": False,
                        "display": None
                    })
                }
                return

            yield {
                "event": "research_thinking",
                "data": json.dumps({"status": "Synthesizing research report..."})
            }

            report = await self.synthesizer.synthesize(plan, valid_sources)

            # Extract a human-readable answer for consumers that expect the 'answer' field
            answer_text = (
                report.get("summary") or
                report.get("answer") or
                report.get("executive_summary") or
                ""
            )
            if not answer_text:
                answer_text = "Here is the research report for your query."

            yield {
                "event": "delta",
                "data": json.dumps({"text": answer_text})
            }
            yield {
                "event": "final",
                "data": json.dumps({
                    "mode": "research",
                    "answer": answer_text,
                    "sources": [{"title": s.get("title", ""), "url": s.get("url", ""), "type": "web"} for s in valid_sources],
                    "confidence": "high",
                    "needs_clarification": False,
                    "display": report
                })
            }

        except Exception as e:
            logger.error(f"[ResearchAgent] Unhandled error: {e}")
            fallback_answer = "An error occurred while researching this topic."
            yield {
                "event": "delta",
                "data": json.dumps({"text": fallback_answer})
            }
            yield {
                "event": "final",
                "data": json.dumps({
                    "mode": "research",
                    "answer": fallback_answer,
                    "sources": [],
                    "confidence": "low",
                    "needs_clarification": False,
                    "display": None
                })
            }