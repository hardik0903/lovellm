"""
CodeSolver — multi-step agent loop.

Old:  single Groq call → structured JSON output.

New loop:
  Step 1 (Observe + Plan)  — parse intent, extract the code block if any
  Step 2 (Act: analyse)    — identify the root cause / spec before writing code
  Step 3 (Act: generate)   — write / fix / optimise the code
  Step 4 (Reflect: verify) — fast self-review pass: does the generated code
                             contain obvious bugs or violate the user's requirements?
                             If yes, do one corrective pass (max 1 retry).
"""

import os
import json
import re
from typing import AsyncGenerator, Dict, Any, Optional
from groq import AsyncGroq
from logger import logger
from agent_base import AgentMemory, AgentStep


class CodeSolver:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.fast_model = "llama-3.1-8b-instant"
        self.strong_model = "llama-3.3-70b-versatile"

    # ------------------------------------------------------------------ #
    # Step 1 — Observe: extract code block from query if present          #
    # ------------------------------------------------------------------ #

    def _extract_code_block(self, query: str) -> Optional[str]:
        match = re.search(r"```[a-z]*\n(.*?)```", query, re.DOTALL)
        return match.group(1).strip() if match else None

    # ------------------------------------------------------------------ #
    # Step 2 — Act: Analyse                                               #
    # ------------------------------------------------------------------ #

    async def _analyse(
        self, query: str, intent: str, language: str,
        code_block: Optional[str], memory: AgentMemory
    ) -> Dict[str, Any]:
        """
        Before writing a single line of code, reason about the problem.
        For 'debug'/'review'/'optimise': identify root cause.
        For 'write'/'convert': identify requirements and edge cases.
        """
        prior = memory.get_context_summary()
        code_section = f"\nCode under analysis:\n```{language}\n{code_block}\n```" if code_block else ""

        prompt = f"""You are a senior {language} engineer performing a pre-coding analysis.
Prior reasoning:
{prior}

Task: {intent}
{code_section}

Return JSON:
{{
  "root_cause": "one sentence — the core problem or what needs to be built",
  "requirements": ["req 1", "req 2"],
  "edge_cases": ["edge case 1"],
  "chosen_approach": "brief description of the approach you will use",
  "why_not_alternative": "why the chosen approach is better than the obvious alternative",
  "complexity_target": "O(?) time, O(?) space"
}}

Query: "{query}"
"""
        resp = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.fast_model,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)

    # ------------------------------------------------------------------ #
    # Step 3 — Act: Generate                                              #
    # ------------------------------------------------------------------ #

    async def _generate(
        self, query: str, intent: str, language: str,
        analysis: Dict[str, Any], code_block: Optional[str],
        memory: AgentMemory
    ) -> Dict[str, Any]:
        """
        Generate the actual code using the analysis from step 2.
        """
        prior = memory.get_context_summary()
        approach = analysis.get("chosen_approach", "")
        requirements = analysis.get("requirements", [])
        edge_cases = analysis.get("edge_cases", [])
        code_section = f"\nOriginal code:\n```{language}\n{code_block}\n```" if code_block else ""

        schema = {
            "intent": intent,
            "language": language,
            "problem_summary": analysis.get("root_cause", ""),
            "code_before": code_block or "",
            "code_after": f"# your {language} code here",
            "diff": [
                {"line": 1, "before": "old line", "after": "new line", "reason": "why changed"}
            ],
            "explanation": "step-by-step explanation of the solution",
            "time_complexity": analysis.get("complexity_target", "O(?)"),
            "space_complexity": "O(?)",
            "edge_cases_handled": edge_cases,
            "alternative_approaches": [analysis.get("why_not_alternative", "")],
            "common_mistakes": ["mistake to avoid"],
        }

        prompt = f"""You are an expert {language} developer.
Prior reasoning in this session:
{prior}

Approach decided: {approach}
Requirements to satisfy: {json.dumps(requirements)}
Edge cases to handle: {json.dumps(edge_cases)}
{code_section}

Implement the solution and return ONLY valid JSON matching:
{json.dumps(schema, indent=2)}

Query: "{query}"
"""
        resp = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.strong_model,
            response_format={"type": "json_object"},
            stream=True,
            temperature=0.05,
        )
        buf = ""
        async for chunk in resp:
            c = chunk.choices[0].delta.content
            if c:
                buf += c
        return json.loads(buf)

    # ------------------------------------------------------------------ #
    # Step 4 — Reflect: Self-review                                       #
    # ------------------------------------------------------------------ #

    async def _self_review(
        self, query: str, language: str,
        generated: Dict[str, Any], requirements: list,
        memory: AgentMemory
    ) -> Dict[str, Any]:
        """
        A separate fast model reads the generated code and flags issues.
        Returns {ok: bool, issues: [...], fixed_code: str or None}
        """
        code = generated.get("code_after", "")
        prior = memory.get_context_summary()

        prompt = f"""You are a code reviewer. Read the code below and check it against the requirements.
Prior reasoning:
{prior}

Requirements: {json.dumps(requirements)}
Language: {language}

Code to review:
```{language}
{code}
```

Return JSON:
{{
  "ok": true/false,
  "issues": ["issue 1 (empty list if ok)"],
  "fixed_code": "corrected code if ok is false, else null"
}}

Be strict. Flag logic errors, missing edge case handling, and off-by-one errors.
"""
        resp = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.fast_model,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)

    # ------------------------------------------------------------------ #
    # Public interface — called by CodeAgent                              #
    # ------------------------------------------------------------------ #

    async def solve(
        self, query: str, classification: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        intent = classification.get("intent", "write")
        language = classification.get("language", "unknown")
        memory = AgentMemory(query)

        # ================================================================
        # OBSERVE
        # ================================================================
        code_block = self._extract_code_block(query)
        yield {
            "event": "code_thinking",
            "data": json.dumps({
                "status": f"Observing: {intent} task in {language}. Code block present: {code_block is not None}",
                "step": 0,
            }),
        }

        # ================================================================
        # STEP 1 — ACT: Analyse the problem before touching code
        # ================================================================
        yield {
            "event": "code_thinking",
            "data": json.dumps({"status": "Analysing root cause / requirements…", "step": 1}),
        }
        analysis = await self._analyse(query, intent, language, code_block, memory)

        step1 = AgentStep(
            step_num=1,
            thought="Before writing code I must understand the root cause and constraints.",
            action="analyse",
            observation=f"Root cause: {analysis.get('root_cause', 'unknown')} | Approach: {analysis.get('chosen_approach', '')}",
            result=analysis,
        )
        memory.add_step(step1)
        yield {"event": "code_step", "data": json.dumps(step1.to_dict())}

        reflect1 = {
            "decision": "escalate" if analysis.get("error") else "continue",
            "reason": analysis.get("error", "Analysis successful."),
        }
        yield {"event": "code_reflection", "data": json.dumps(reflect1)}
        if reflect1["decision"] == "escalate":
            yield {
                "event": "final",
                "data": json.dumps({
                    "mode": "code", "answer": "Failed during problem analysis.",
                    "sources": [], "confidence": "low", "display": None,
                }),
            }
            return

        # ================================================================
        # STEP 2 — ACT: Generate the code
        # ================================================================
        yield {
            "event": "code_thinking",
            "data": json.dumps({"status": f"Generating {language} code using chosen approach…", "step": 2}),
        }
        requirements = analysis.get("requirements", [])
        generated = await self._generate(query, intent, language, analysis, code_block, memory)

        step2 = AgentStep(
            step_num=2,
            thought="Implementing the solution using the approach decided in step 1.",
            action="generate",
            observation=f"Generated code ({len(generated.get('code_after', ''))} chars). "
                        f"Complexity: {generated.get('time_complexity', '?')}",
            result={"time_complexity": generated.get("time_complexity"), "has_diff": bool(generated.get("diff"))},
        )
        memory.add_step(step2)
        yield {"event": "code_step", "data": json.dumps(step2.to_dict())}

        # ================================================================
        # STEP 3 — REFLECT: Self-review
        # ================================================================
        yield {
            "event": "code_thinking",
            "data": json.dumps({"status": "Self-reviewing the generated code for correctness…", "step": 3}),
        }
        review = await self._self_review(query, language, generated, requirements, memory)

        step3 = AgentStep(
            step_num=3,
            thought="Reading my own output as a reviewer to catch issues before delivering.",
            action="self_review",
            observation=(
                "Code passed review." if review.get("ok")
                else f"Issues found: {review.get('issues', [])}. Applying fix."
            ),
            result={"ok": review.get("ok"), "issues": review.get("issues", [])},
        )
        memory.add_step(step3)
        yield {"event": "code_step", "data": json.dumps(step3.to_dict())}
        yield {
            "event": "code_reflection",
            "data": json.dumps({
                "decision": "stop" if review.get("ok") else "patch",
                "reason": step3.observation,
            }),
        }

        # Apply the fix if review found issues
        if not review.get("ok") and review.get("fixed_code"):
            generated["code_after"] = review["fixed_code"]
            generated["review_issues_fixed"] = review.get("issues", [])

        # ================================================================
        # Final answer
        # ================================================================
        answer_text = generated.get("explanation") or generated.get("problem_summary") or "Here is the code solution."

        yield {"event": "delta", "data": json.dumps({"text": answer_text})}
        yield {
            "event": "final",
            "data": json.dumps({
                "mode": "code",
                "answer": answer_text,
                "sources": [],
                "confidence": "high",
                "needs_clarification": False,
                "display": generated,
                "agent_trace": [s.to_dict() for s in memory.steps],
            }),
        }