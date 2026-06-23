import abc
import json
import time
from typing import Dict, Any, AsyncGenerator, List, Optional
from logger import logger


class AgentStep:
    """Represents one completed step in an agent's reasoning loop."""
    def __init__(self, step_num: int, thought: str, action: str, observation: str, result: Any = None):
        self.step_num = step_num
        self.thought = thought
        self.action = action
        self.observation = observation
        self.result = result
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_num,
            "thought": self.thought,
            "action": self.action,
            "observation": self.observation,
            "result": self.result,
        }


class AgentMemory:
    """
    Short-term scratchpad memory for a single agent execution.
    Stores the chain of steps so each iteration can see what was tried before.
    This enables genuine reflection — the agent can say 'step 1 failed because X,
    so I will try Y instead' rather than blindly retrying the same prompt.
    """
    def __init__(self, query: str):
        self.query = query
        self.steps: List[AgentStep] = []
        self.facts: Dict[str, Any] = {}        # Key facts extracted across steps
        self.failed_actions: List[str] = []    # Actions that produced bad results

    def add_step(self, step: AgentStep):
        self.steps.append(step)
        if step.result and isinstance(step.result, dict) and step.result.get("error"):
            self.failed_actions.append(step.action)

    def get_context_summary(self) -> str:
        """Returns a formatted string of all prior steps for injection into the next prompt."""
        if not self.steps:
            return "No prior steps."
        lines = []
        for s in self.steps:
            lines.append(
                f"Step {s.step_num} | Action: {s.action}\n"
                f"  Thought: {s.thought}\n"
                f"  Observation: {s.observation}"
            )
        return "\n".join(lines)

    def add_fact(self, key: str, value: Any):
        self.facts[key] = value

    def has_fact(self, key: str) -> bool:
        return key in self.facts


class BaseAgent(abc.ABC):
    """
    Abstract base class for all specialized agents.

    Architecture — Observe → Plan → Act → Reflect loop:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  OBSERVE   Read the query + any context (docs, data, prior steps)   │
    │  PLAN      Decompose into sub-tasks; choose which tools to call     │
    │  ACT       Execute one tool call (LLM, retriever, search, sympy…)   │
    │  REFLECT   Evaluate the result; decide whether to stop or retry     │
    └─────────────────────────────────────────────────────────────────────┘

    Subclasses must implement:
      detect()   — fast heuristic: should this agent handle the query?
      classify() — deeper intent decomposition (may call a small LLM)
      solve()    — the full agent loop (yields SSE events)

    The loop helpers below (observe / plan / act / reflect) are provided so
    subclasses can call them in a consistent pattern rather than reimplementing
    the bookkeeping each time.
    """

    MAX_STEPS: int = 4          # Hard cap — prevents infinite loops
    CONFIDENCE_THRESHOLD: float = 0.6   # Minimum quality score to accept an answer

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def detect(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Fast (< 1 ms) heuristic check.
        Must return at minimum: {'confidence': float, 'reasoning': str}
        """
        pass

    @abc.abstractmethod
    async def classify(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Deeper intent classification. May call a small/fast LLM (8b-instant).
        Must return a dict describing the task sub-type.
        """
        pass

    @abc.abstractmethod
    async def solve(self, query: str, context: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        The main agent loop. Yields SSE-formatted events.
        Must yield at least one {'event': 'final', 'data': json_string} event.
        """
        pass

    # ------------------------------------------------------------------ #
    # Loop helpers — call these inside solve()                            #
    # ------------------------------------------------------------------ #

    def _observe(self, memory: AgentMemory, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 1 — Observe.
        Aggregates query + context into a structured observation dict.
        """
        return {
            "query": memory.query,
            "prior_steps": len(memory.steps),
            "known_facts": memory.facts,
            "failed_actions": memory.failed_actions,
            "context_keys": list(context.keys()) if context else [],
        }

    def _plan(self, observation: Dict[str, Any], classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 2 — Plan.
        Builds a simple action plan from the observation + classification.
        Subclasses can override this for domain-specific planning.
        """
        return {
            "primary_action": classification.get("intent", "answer"),
            "sub_tasks": classification.get("sub_tasks", []),
            "requires_tools": classification.get("requires_tools", []),
            "fallback_action": "direct_answer",
        }

    def _reflect(self, memory: AgentMemory, result: Dict[str, Any]) -> Dict[str, str]:
        """
        Step 4 — Reflect.
        Evaluates whether the result is good enough to stop,
        or whether the loop should continue with a different approach.
        Returns {'decision': 'stop' | 'retry' | 'escalate', 'reason': str}
        """
        if not result:
            return {"decision": "retry", "reason": "Empty result — will retry with adjusted prompt."}

        if result.get("error"):
            if len(memory.failed_actions) >= self.MAX_STEPS - 1:
                return {"decision": "escalate", "reason": "Max retries reached after repeated failures."}
            return {"decision": "retry", "reason": f"Action failed: {result['error']}"}

        # Check confidence if available
        confidence = result.get("confidence", 1.0)
        if isinstance(confidence, str):
            confidence = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(confidence, 0.5)

        if confidence < self.CONFIDENCE_THRESHOLD and len(memory.steps) < self.MAX_STEPS:
            return {
                "decision": "retry",
                "reason": f"Confidence {confidence:.2f} below threshold {self.CONFIDENCE_THRESHOLD}. Refining."
            }

        return {"decision": "stop", "reason": "Result meets quality threshold."}

    # ------------------------------------------------------------------ #
    # SSE event helpers                                                   #
    # ------------------------------------------------------------------ #

    def _thinking_event(self, agent_name: str, status: str, step: Optional[int] = None) -> Dict[str, Any]:
        data: Dict[str, Any] = {"status": status, "agent": agent_name}
        if step is not None:
            data["step"] = step
        return {"event": f"{agent_name}_thinking", "data": json.dumps(data)}

    def _step_event(self, agent_name: str, step: AgentStep) -> Dict[str, Any]:
        return {"event": f"{agent_name}_step", "data": json.dumps(step.to_dict())}

    def _reflection_event(self, agent_name: str, decision: str, reason: str) -> Dict[str, Any]:
        return {
            "event": f"{agent_name}_reflection",
            "data": json.dumps({"decision": decision, "reason": reason})
        }

    def _final_event(self, mode: str, answer: str, display: Any = None,
                     sources: List = None, confidence: str = "high",
                     memory: Optional[AgentMemory] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "mode": mode,
            "answer": answer,
            "sources": sources or [],
            "confidence": confidence,
            "needs_clarification": False,
            "display": display,
        }
        if memory:
            payload["agent_trace"] = [s.to_dict() for s in memory.steps]
        return {"event": "final", "data": json.dumps(payload)}

    def _error_event(self, mode: str, error: str) -> Dict[str, Any]:
        return {
            "event": "final",
            "data": json.dumps({
                "mode": mode,
                "answer": f"An error occurred: {error}",
                "sources": [],
                "confidence": "low",
                "needs_clarification": False,
                "display": None,
            })
        }