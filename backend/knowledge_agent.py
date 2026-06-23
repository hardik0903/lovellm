"""
KnowledgeAgent — Observe → Plan → Act → Reflect loop.

Old behaviour: single Groq call with a JSON schema stuffed into a prompt.

New behaviour:
  Step 1 (Observe)   — detect query complexity and expertise level
  Step 2 (Plan)      — decide whether a direct answer is enough, or whether
                       we need to break the topic into sub-concepts first
  Step 3 (Act)       — for complex queries, first extract sub-concepts, then
                       synthesise them; for simple queries, go direct
  Step 4 (Reflect)   — score completeness: did the answer cover the required
                       sub-concepts? if not, fill the gaps in a second pass
"""

import os
import json
from typing import Dict, Any, AsyncGenerator
from groq import AsyncGroq
from logger import logger
from agent_base import BaseAgent, AgentMemory, AgentStep
from knowledge_detector import KnowledgeDetector
from knowledge_classifier import KnowledgeClassifier


class KnowledgeAgent(BaseAgent):
    def __init__(self):
        self.detector = KnowledgeDetector()
        self.classifier = KnowledgeClassifier()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.fast_model = "llama-3.1-8b-instant"
        self.strong_model = "llama-3.3-70b-versatile"

    # ------------------------------------------------------------------ #
    # detect / classify                                                    #
    # ------------------------------------------------------------------ #

    def detect(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.detector.detect(query)

    async def classify(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return await self.classifier.classify(query)

    # ------------------------------------------------------------------ #
    # Internal tools called during Act steps                              #
    # ------------------------------------------------------------------ #

    async def _decompose_topic(self, query: str, memory: AgentMemory) -> Dict[str, Any]:
        """
        Act: for a complex topic, ask a fast model to list the sub-concepts
        that a complete explanation must cover.  This structures the next Act step.
        """
        prior = memory.get_context_summary()
        prompt = f"""You decompose knowledge questions into sub-concepts.
Prior reasoning:
{prior}

Query: "{query}"

Return JSON:
{{
  "core_concept": "main concept name",
  "is_complex": true/false,
  "sub_concepts": ["sub1", "sub2"],   // empty list if simple
  "analogy_domain": "domain for analogy (e.g. cooking, sports, architecture)"
}}"""
        resp = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.fast_model,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)

    async def _synthesise(
        self,
        query: str,
        decomposition: Dict[str, Any],
        expertise: str,
        memory: AgentMemory,
    ) -> Dict[str, Any]:
        """
        Act: build a structured knowledge response.
        Uses the decomposition from the previous step so the answer
        is guaranteed to cover each sub-concept.
        If sub_concepts exist, each one gets its own section.
        """
        prior = memory.get_context_summary()
        sub_concepts = decomposition.get("sub_concepts", [])
        analogy_domain = decomposition.get("analogy_domain", "everyday life")

        sections_schema = [
            {"sub_concept": sc, "explanation": "...", "example": "..."}
            for sc in sub_concepts
        ] if sub_concepts else []

        schema = {
            "concept": decomposition.get("core_concept", ""),
            "expertise_level": expertise,
            "definition": "one sentence definition",
            "intuition": f"analogy from the domain of {analogy_domain}",
            "sections": sections_schema,   # filled for complex topics
            "examples": ["example 1", "example 2"],
            "common_misconceptions": ["misconception 1"],
            "related_concepts": ["concept A", "concept B"],
            "gaps_in_previous_answer": "none | list what was missing",
        }

        prompt = f"""You are an expert Knowledge Agent calibrated to '{expertise}' level.

Prior reasoning in this session:
{prior}

The query has been decomposed into these sub-concepts that MUST be addressed:
{json.dumps(sub_concepts, indent=2) if sub_concepts else "None — treat as a direct question."}

Use an analogy from: {analogy_domain}

Return ONLY valid JSON matching:
{json.dumps(schema, indent=2)}

Query: "{query}"
"""
        resp = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.strong_model,
            response_format={"type": "json_object"},
            temperature=0.2,
            stream=True,
        )
        buf = ""
        async for chunk in resp:
            content = chunk.choices[0].delta.content
            if content:
                buf += content
        return json.loads(buf)

    async def _reflection_check(
        self,
        query: str,
        answer: Dict[str, Any],
        required_sub_concepts: list,
        memory: AgentMemory,
    ) -> Dict[str, Any]:
        """
        Reflect: check whether every required sub-concept was addressed.
        Returns {covered: bool, missing: [...], patch: str or None}
        """
        if not required_sub_concepts:
            return {"covered": True, "missing": [], "patch": None}

        answer_text = json.dumps(answer)
        missing = [
            sc for sc in required_sub_concepts
            if sc.lower() not in answer_text.lower()
        ]

        if not missing:
            return {"covered": True, "missing": [], "patch": None}

        # One more targeted pass to fill gaps
        prior = memory.get_context_summary()
        patch_prompt = f"""The following sub-concepts were not covered in the previous answer:
{missing}

Prior reasoning:
{prior}

Original query: "{query}"

Write a concise supplementary paragraph (plain text, no JSON) that addresses ONLY these missing points.
"""
        resp = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": patch_prompt}],
            model=self.fast_model,
            temperature=0.1,
        )
        patch_text = resp.choices[0].message.content.strip()
        return {"covered": False, "missing": missing, "patch": patch_text}

    # ------------------------------------------------------------------ #
    # Main agent loop                                                     #
    # ------------------------------------------------------------------ #

    async def solve(self, query: str, context: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        memory = AgentMemory(query)
        agent_name = "knowledge"

        try:
            # ---- OBSERVE ----
            observation = self._observe(memory, context or {})
            yield self._thinking_event(agent_name, "Observing query structure…", step=0)

            # ---- classify (provides expertise level + domain) ----
            classification = await self.classify(query)
            expertise = classification.get("expertise_level", "intermediate")
            domain = classification.get("domain", "General")
            yield self._thinking_event(
                agent_name,
                f"Classified as {domain} at {expertise} level. Planning sub-tasks…",
                step=1,
            )

            # ---- PLAN ----
            plan = self._plan(observation, {
                "intent": "explain",
                "sub_tasks": ["decompose", "synthesise", "reflect"],
                "requires_tools": ["decompose_topic", "synthesise"],
            })

            # ================================================================
            # STEP 1 — ACT: Decompose the topic
            # ================================================================
            yield self._thinking_event(agent_name, "Decomposing topic into sub-concepts…", step=2)
            decomposition = await self._decompose_topic(query, memory)

            step1 = AgentStep(
                step_num=1,
                thought=f"I need to understand what sub-concepts make up '{query}' before explaining.",
                action="decompose_topic",
                observation=f"Found sub-concepts: {decomposition.get('sub_concepts', [])}",
                result=decomposition,
            )
            memory.add_step(step1)
            yield self._step_event(agent_name, step1)

            reflect1 = self._reflect(memory, decomposition)
            yield self._reflection_event(agent_name, reflect1["decision"], reflect1["reason"])

            if reflect1["decision"] == "escalate":
                yield self._error_event(agent_name, "Failed to decompose the topic.")
                return

            # ================================================================
            # STEP 2 — ACT: Synthesise the full answer
            # ================================================================
            yield self._thinking_event(
                agent_name,
                f"Synthesising explanation covering {len(decomposition.get('sub_concepts', []))} sub-concepts…",
                step=3,
            )
            synthesis = await self._synthesise(query, decomposition, expertise, memory)

            step2 = AgentStep(
                step_num=2,
                thought="Using the decomposition to write a structured explanation calibrated to the user's expertise.",
                action="synthesise",
                observation=f"Produced definition + {len(synthesis.get('sections', []))} sections.",
                result={"sections_count": len(synthesis.get("sections", [])), "has_analogy": bool(synthesis.get("intuition"))},
            )
            memory.add_step(step2)
            yield self._step_event(agent_name, step2)

            # ================================================================
            # STEP 3 — REFLECT: Check sub-concept coverage
            # ================================================================
            yield self._thinking_event(agent_name, "Reflecting: checking sub-concept coverage…", step=4)
            required_sub = decomposition.get("sub_concepts", [])
            reflection = await self._reflection_check(query, synthesis, required_sub, memory)

            step3 = AgentStep(
                step_num=3,
                thought="Verifying that every required sub-concept was addressed in the synthesis.",
                action="reflect_coverage",
                observation=(
                    "All sub-concepts covered."
                    if reflection["covered"]
                    else f"Missing: {reflection['missing']}. Patching."
                ),
                result=reflection,
            )
            memory.add_step(step3)
            yield self._step_event(agent_name, step3)
            yield self._reflection_event(
                agent_name,
                "stop" if reflection["covered"] else "patch",
                step3.observation,
            )

            # Apply patch if gaps were found
            if not reflection["covered"] and reflection.get("patch"):
                existing = synthesis.get("definition", "")
                synthesis["definition"] = existing + "\n\n" + reflection["patch"]
                synthesis["gaps_filled"] = reflection["missing"]

            # ================================================================
            # Final answer
            # ================================================================
            answer_text = synthesis.get("definition", "") or synthesis.get("intuition", "")
            if not answer_text:
                answer_text = "Here is a knowledge-based explanation for your query."

            yield {"event": "delta", "data": json.dumps({"text": answer_text})}
            yield self._final_event(
                mode=agent_name,
                answer=answer_text,
                display=synthesis,
                memory=memory,
            )

        except Exception as e:
            logger.error(f"[KnowledgeAgent] Unhandled error: {e}")
            yield self._error_event("knowledge", str(e))