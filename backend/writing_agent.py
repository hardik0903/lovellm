"""
WritingAgent — Observe → Plan → Draft → Critique → Revise loop.

Old:  single Groq call → structured JSON output.

New loop:
  Step 1 (Observe)   — classify intent + extract text-to-edit if present
  Step 2 (Plan)      — decompose into quality axes to optimise
                       (grammar, clarity, tone, structure, conciseness)
  Step 3 (Act:Draft) — produce the initial draft / edit
  Step 4 (Reflect)   — critique the draft against the quality axes; score each
  Step 5 (Revise)    — apply targeted revisions only where the score is low
                       (avoids unnecessary rewrites when the draft is already good)
"""

import os
import json
from typing import Dict, Any, AsyncGenerator, List
from groq import AsyncGroq
from logger import logger
from agent_base import BaseAgent, AgentMemory, AgentStep
from writing_detector import WritingDetector
from writing_classifier import WritingClassifier

QUALITY_AXES = ["grammar", "clarity", "tone", "structure", "conciseness"]


class WritingAgent(BaseAgent):
    def __init__(self):
        self.detector = WritingDetector()
        self.classifier = WritingClassifier()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.fast_model = "llama-3.1-8b-instant"
        self.strong_model = "llama-3.3-70b-versatile"

    def detect(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.detector.detect(query)

    async def classify(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return await self.classifier.classify(query)

    # ------------------------------------------------------------------ #
    # Internal tools                                                      #
    # ------------------------------------------------------------------ #

    async def _plan_quality_axes(
        self, query: str, intent: str, doc_type: str, memory: AgentMemory
    ) -> Dict[str, Any]:
        """
        Plan: given the intent and doc_type, decide which quality axes matter most
        and what the target tone/style is.
        """
        prior = memory.get_context_summary()
        prompt = f"""You are a writing quality planner.
Prior reasoning:
{prior}

Intent: {intent}
Document type: {doc_type}
Query: "{query}"

Decide which quality axes are most important for this task and what tone is needed.
Return JSON:
{{
  "priority_axes": ["list from: grammar, clarity, tone, structure, conciseness"],
  "target_tone": "formal / casual / persuasive / empathetic / technical",
  "target_audience": "who will read this",
  "key_constraints": ["e.g. keep under 200 words", "no jargon"]
}}
"""
        resp = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.fast_model,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)

    async def _draft(
        self, query: str, intent: str, doc_type: str,
        plan: Dict[str, Any], memory: AgentMemory
    ) -> Dict[str, Any]:
        """Act Step 1 — produce initial draft."""
        prior = memory.get_context_summary()
        schema = {
            "intent": intent,
            "document_type": doc_type,
            "tone": plan.get("target_tone", "neutral"),
            "original": "original text if editing, else empty",
            "result": "the drafted or edited text",
            "changes": [
                {
                    "type": "grammar / clarity / tone / structure / conciseness",
                    "original_phrase": "old",
                    "revised_phrase": "new",
                    "reason": "why changed",
                }
            ],
            "word_count_before": 0,
            "word_count_after": 0,
            "summary_of_changes": "short summary",
        }

        prompt = f"""You are an expert writing assistant.
Prior reasoning:
{prior}

Intent: {intent} | Document type: {doc_type}
Target tone: {plan.get('target_tone')} | Audience: {plan.get('target_audience')}
Constraints: {json.dumps(plan.get('key_constraints', []))}

Produce your best first draft and return ONLY valid JSON matching:
{json.dumps(schema, indent=2)}

Query: "{query}"
"""
        resp = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.strong_model,
            response_format={"type": "json_object"},
            stream=True,
            temperature=0.5,
        )
        buf = ""
        async for chunk in resp:
            c = chunk.choices[0].delta.content
            if c:
                buf += c
        return json.loads(buf)

    async def _critique(
        self, draft_text: str, priority_axes: List[str], memory: AgentMemory
    ) -> Dict[str, Any]:
        """
        Reflect: score the draft on each quality axis (0–10).
        Only axes with score < 7 will trigger a revision.
        """
        prior = memory.get_context_summary()
        axes_schema = {ax: {"score": 0, "issue": "what is wrong or 'ok'"} for ax in QUALITY_AXES}

        prompt = f"""You are a writing critic. Score the draft on each quality axis.
Prior reasoning:
{prior}

Priority axes for this task: {priority_axes}

Draft:
\"\"\"
{draft_text}
\"\"\"

Return JSON:
{{
  "axes": {json.dumps(axes_schema)},
  "overall": 0,
  "needs_revision": true/false,
  "revision_focus": ["list of axes with score < 7"]
}}
Be strict. A score of 10 means perfect. 7+ means acceptable. Below 7 means revision needed.
"""
        resp = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.fast_model,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)

    async def _revise(
        self, draft_text: str, critique: Dict[str, Any],
        plan: Dict[str, Any], memory: AgentMemory
    ) -> str:
        """Act Step 2 — targeted revision only for low-scoring axes."""
        revision_focus = critique.get("revision_focus", [])
        if not revision_focus:
            return draft_text

        prior = memory.get_context_summary()
        issues = {
            ax: critique["axes"][ax]["issue"]
            for ax in revision_focus
            if ax in critique.get("axes", {})
        }

        prompt = f"""You are a writing revisor. Fix ONLY the listed issues in the draft.
Prior reasoning:
{prior}

Issues to fix (do not change anything else):
{json.dumps(issues, indent=2)}

Target tone: {plan.get('target_tone')}
Constraints: {json.dumps(plan.get('key_constraints', []))}

Original draft:
\"\"\"
{draft_text}
\"\"\"

Return ONLY the revised text (plain text, no JSON, no preamble).
"""
        resp = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.strong_model,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------------------ #
    # Main agent loop                                                     #
    # ------------------------------------------------------------------ #

    async def solve(self, query: str, context: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        memory = AgentMemory(query)
        agent_name = "writing"

        try:
            # OBSERVE
            observation = self._observe(memory, context or {})
            yield self._thinking_event(agent_name, "Observing writing task…", step=0)

            # Classify
            classification = await self.classify(query)
            intent = classification.get("intent", "draft")
            doc_type = classification.get("document_type", "general")

            # ================================================================
            # STEP 1 — PLAN: Quality axes
            # ================================================================
            yield self._thinking_event(agent_name, f"Planning quality targets for {intent} of {doc_type}…", step=1)
            plan = await self._plan_quality_axes(query, intent, doc_type, memory)

            step1 = AgentStep(
                step_num=1,
                thought=f"For a {doc_type} {intent} task I need to identify which quality axes matter most.",
                action="plan_quality_axes",
                observation=f"Priority axes: {plan.get('priority_axes')} | Tone: {plan.get('target_tone')}",
                result=plan,
            )
            memory.add_step(step1)
            yield self._step_event(agent_name, step1)

            # ================================================================
            # STEP 2 — ACT: Draft
            # ================================================================
            yield self._thinking_event(agent_name, "Drafting…", step=2)
            draft_obj = await self._draft(query, intent, doc_type, plan, memory)
            draft_text = draft_obj.get("result", "")

            step2 = AgentStep(
                step_num=2,
                thought="Produce the first draft using the planned tone and constraints.",
                action="draft",
                observation=f"Draft produced ({len(draft_text.split())} words).",
                result={"word_count": len(draft_text.split())},
            )
            memory.add_step(step2)
            yield self._step_event(agent_name, step2)

            # ================================================================
            # STEP 3 — REFLECT: Critique the draft
            # ================================================================
            yield self._thinking_event(agent_name, "Critiquing draft quality…", step=3)
            critique = await self._critique(draft_text, plan.get("priority_axes", QUALITY_AXES), memory)

            step3 = AgentStep(
                step_num=3,
                thought="Self-critique the draft to see which quality axes need improvement.",
                action="critique",
                observation=(
                    f"Overall score: {critique.get('overall')}/10. "
                    f"Revision needed: {critique.get('needs_revision')}. "
                    f"Focus: {critique.get('revision_focus', [])}"
                ),
                result={"overall": critique.get("overall"), "needs_revision": critique.get("needs_revision")},
            )
            memory.add_step(step3)
            yield self._step_event(agent_name, step3)
            yield self._reflection_event(
                agent_name,
                "revise" if critique.get("needs_revision") else "stop",
                step3.observation,
            )

            # ================================================================
            # STEP 4 — ACT: Revise (only if critique found issues)
            # ================================================================
            final_text = draft_text
            if critique.get("needs_revision"):
                yield self._thinking_event(
                    agent_name,
                    f"Revising axes: {critique.get('revision_focus', [])}…",
                    step=4,
                )
                revised = await self._revise(draft_text, critique, plan, memory)

                step4 = AgentStep(
                    step_num=4,
                    thought="Apply targeted revisions to only the axes that scored below 7.",
                    action="revise",
                    observation=f"Revision applied. New word count: {len(revised.split())}.",
                    result={"revised_word_count": len(revised.split())},
                )
                memory.add_step(step4)
                yield self._step_event(agent_name, step4)

                final_text = revised
                draft_obj["result"] = final_text
                draft_obj["critique_scores"] = critique.get("axes", {})
                draft_obj["revision_applied"] = critique.get("revision_focus", [])

            # ================================================================
            # Final answer
            # ================================================================
            answer_text = final_text or draft_obj.get("summary_of_changes") or "Writing task complete."

            yield {"event": "delta", "data": json.dumps({"text": answer_text})}
            yield self._final_event(
                mode=agent_name,
                answer=answer_text,
                display=draft_obj,
                memory=memory,
            )

        except Exception as e:
            logger.error(f"[WritingAgent] Unhandled error: {e}")
            yield self._error_event("writing", str(e))