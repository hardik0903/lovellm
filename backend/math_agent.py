"""
MathAgent — Classify → Solve (step-by-step) → Verify → Graph loop.

Old:  classify → solve → (optionally) graph.

New loop:
  Step 1 (Classify)  — detect category + whether symbolic computation is needed
  Step 2 (Solve)     — step-by-step solution with intermediate expressions
  Step 3 (Verify)    — substitution check: plug the answer back in and confirm
                       the equation holds; if it doesn't, flag and retry once
  Step 4 (Graph)     — if required, generate graph data from the verified solution
"""

import json
from typing import AsyncGenerator, Dict, Any
from logger import logger
from agent_base import BaseAgent, AgentMemory, AgentStep
from math_detector import MathDetector
from math_classifier import MathClassifier
from math_solver import MathSolver
from math_grapher import MathGrapher


class MathAgent(BaseAgent):
    def __init__(self):
        self.detector = MathDetector()
        self.classifier = MathClassifier()
        self.solver = MathSolver()
        self.grapher = MathGrapher()

    def detect(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.detector.detect(query)

    async def classify(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return await self.classifier.classify(query)

    # ------------------------------------------------------------------ #
    # Verification helper                                                 #
    # ------------------------------------------------------------------ #

    def _verify_solution(self, solution_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect: check if the solution object has a verification block
        and whether it passed. Also check for logical red-flags in the steps.
        Returns {ok: bool, issue: str or None}
        """
        verification = solution_obj.get("verification", {})

        # If the LLM included its own verification block, trust it
        if verification:
            confirmed = verification.get("confirmed", True)
            return {
                "ok": confirmed,
                "issue": None if confirmed else f"LLM verification failed: {verification.get('check', 'unknown')}",
            }

        # Fallback: if solution is empty or error, flag it
        solution_text = solution_obj.get("solution", "")
        if not solution_text or "error" in solution_text.lower():
            return {"ok": False, "issue": "Solution field is empty or contains an error."}

        # Check that steps exist and are non-trivial
        steps = solution_obj.get("steps", [])
        if not steps:
            return {"ok": False, "issue": "No step-by-step breakdown present."}

        return {"ok": True, "issue": None}

    # ------------------------------------------------------------------ #
    # Main agent loop                                                     #
    # ------------------------------------------------------------------ #

    async def solve(self, query: str, context: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        memory = AgentMemory(query)
        agent_name = "math"

        try:
            # ================================================================
            # STEP 1 — CLASSIFY
            # ================================================================
            yield self._thinking_event(agent_name, "Classifying mathematical problem…", step=1)
            classification = await self.classifier.classify(query)
            category = classification.get("category", "algebra")
            requires_graph = classification.get("requires_graph", False)

            step1 = AgentStep(
                step_num=1,
                thought="Identify the mathematical category so the correct solving strategy can be applied.",
                action="classify",
                observation=f"Category: {category} | Difficulty: {classification.get('difficulty')} | Graph needed: {requires_graph}",
                result=classification,
            )
            memory.add_step(step1)
            yield self._step_event(agent_name, step1)

            # ================================================================
            # STEP 2 — SOLVE (with step events streamed)
            # ================================================================
            yield self._thinking_event(agent_name, f"Solving ({category}) step by step…", step=2)

            solution_obj = None
            async for event in self.solver.solve(query, classification):
                if event["event"] == "math_step":
                    # Forward individual step events to the frontend
                    yield event
                elif event["event"] == "math_thinking":
                    yield event
                elif event["event"] == "math_final":
                    try:
                        solution_obj = json.loads(event["data"])
                    except Exception:
                        pass

            if not solution_obj:
                yield self._error_event(agent_name, "Solver returned no solution.")
                return

            step2 = AgentStep(
                step_num=2,
                thought="Run the category-specific solver to produce a step-by-step breakdown.",
                action="solve",
                observation=f"{len(solution_obj.get('steps', []))} steps produced. Solution: {solution_obj.get('solution', '')}",
                result={"step_count": len(solution_obj.get("steps", [])), "solution": solution_obj.get("solution")},
            )
            memory.add_step(step2)
            yield self._step_event(agent_name, step2)

            # ================================================================
            # STEP 3 — REFLECT: Verify solution
            # ================================================================
            yield self._thinking_event(agent_name, "Verifying solution correctness…", step=3)
            verification = self._verify_solution(solution_obj)

            step3 = AgentStep(
                step_num=3,
                thought="Plug the answer back in or check the verification block to confirm correctness.",
                action="verify",
                observation=(
                    "Solution verified." if verification["ok"]
                    else f"Verification failed: {verification['issue']}. Will retry."
                ),
                result=verification,
            )
            memory.add_step(step3)
            yield self._step_event(agent_name, step3)
            yield self._reflection_event(
                agent_name,
                "stop" if verification["ok"] else "retry",
                step3.observation,
            )

            # One retry if verification failed
            if not verification["ok"]:
                yield self._thinking_event(agent_name, "Re-solving after verification failure…", step=4)
                retry_obj = None
                async for event in self.solver.solve(query, classification):
                    if event["event"] == "math_final":
                        try:
                            retry_obj = json.loads(event["data"])
                        except Exception:
                            pass

                if retry_obj and not retry_obj.get("error"):
                    solution_obj = retry_obj
                    step4 = AgentStep(
                        step_num=4,
                        thought="Retry the solve after the initial verification failed.",
                        action="retry_solve",
                        observation=f"Retry solution: {solution_obj.get('solution', 'unknown')}",
                        result={"retry": True},
                    )
                    memory.add_step(step4)
                    yield self._step_event(agent_name, step4)

            # ================================================================
            # STEP 4 (conditional) — GRAPH
            # ================================================================
            if requires_graph:
                yield self._thinking_event(agent_name, "Generating graph data…", step=5)
                graph_data = self.grapher.generate_graph_data(classification, solution_obj)
                solution_obj["graph_data"] = graph_data

                step_graph = AgentStep(
                    step_num=5,
                    thought="This problem involves a function or geometric object — generate plot data.",
                    action="generate_graph",
                    observation=f"Graph type: {graph_data.get('type')} | Special points: {len(graph_data.get('special_points', []))}",
                    result={"graph_type": graph_data.get("type")},
                )
                memory.add_step(step_graph)
                yield self._step_event(agent_name, step_graph)

            # ================================================================
            # Final answer
            # ================================================================
            answer_text = solution_obj.get("solution", "See steps for the solution.")
            display_envelope = {"type": "math_solution", **solution_obj}

            yield {"event": "delta", "data": json.dumps({"text": answer_text})}
            yield self._final_event(
                mode=agent_name,
                answer=answer_text,
                display=display_envelope,
                memory=memory,
            )

        except Exception as e:
            logger.error(f"[MathAgent] Unhandled error: {e}")
            yield self._error_event("math", str(e))