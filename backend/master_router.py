from typing import Dict, Any, Tuple
import asyncio
from agent_registry import AgentRegistry
from confidence_calibrator import ConfidenceCalibrator
from agent_telemetry import AgentTelemetry
from logger import logger

class MasterRouter:
    """
    The main routing engine that intercepts queries and dispatches them to
    specialized agents if confidence is high enough.
    """
    def __init__(self):
        self.registry = AgentRegistry()
        self.calibrator = ConfidenceCalibrator()
        
        # Routing priority when scores are tied or very close
        self.priority_order = [
            "math",
            "code",
            "data",
            "document",
            "writing",
            "research",
            "knowledge"
        ]

    def route(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Runs all detectors synchronously (since they are fast regex/heuristics),
        normalizes their scores, and picks a winner.
        """
        raw_scores = {}
        reasoning_map = {}
        
        # Run all detectors
        for agent_name, agent in self.registry.get_all_agents().items():
            try:
                detection = agent.detect(query, context)
                raw_scores[agent_name] = detection.get("confidence", 0.0)
                reasoning_map[agent_name] = detection.get("reasoning", "")
            except Exception as e:
                logger.error(f"[MasterRouter] Detector error for {agent_name}: {e}")
                raw_scores[agent_name] = 0.0
                
        # Calibrate
        calibrated_scores = self.calibrator.calibrate(raw_scores)
        
        # Select best
        best_agent = None
        best_score = -1.0
        
        # Iterate based on priority order to handle ties correctly
        for agent_name in self.priority_order:
            score = calibrated_scores.get(agent_name, 0.0)
            # If the score is higher, or if it's tied but this agent has higher priority
            # Actually, we want a significant margin to override priority.
            # If the best score is 0.8 and this one is 0.82, we take it.
            # But if best is math (0.8) and code is (0.8), math wins because it was checked first.
            if score > best_score:
                best_score = score
                best_agent = agent_name

        routing_result = {
            "selected_agent": best_agent if best_score >= 0.5 else None,
            "confidence": best_score,
            "reasoning": reasoning_map.get(best_agent, "Fallback") if best_score >= 0.5 else "Confidence too low, falling back",
            "all_scores": calibrated_scores,
            "fallback_agent": "general"
        }
        
        # Telemetry
        AgentTelemetry.log_routing_decision(query, routing_result)
        
        return routing_result

master_router_instance = MasterRouter()
