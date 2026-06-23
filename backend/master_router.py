from typing import Dict, Any, Tuple
import asyncio
from agent_registry import AgentRegistry
try:
    from calibration.confidence_calibration import ConfidenceCalibrator
except Exception:
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

        # Selection threshold applied to CALIBRATED scores.
        #
        # NOTE: this is NOT the same scale as the old raw-score threshold
        # (0.5). Platt scaling maps raw heuristic scores onto
        # P(this agent is the correct route), fit on calibration/labeled_queries.json.
        # Because several detectors produce their highest raw score (e.g.
        # KnowledgeDetector's 0.8) for BOTH true positives and a large
        # number of false positives in the labeled set, the calibrated
        # probability for those scores sits below 0.5 even when the agent
        # is, relative to all other agents, by far the best match.
        #
        # 0.5 on a calibrated probability means "more likely wrong than
        # right in isolation" -- but routing only needs "best among
        # competitors", not "more likely than not in a vacuum". This
        # threshold was tuned by sweeping calibration/evaluate_routing.py
        # over the labeled dataset: 0.20 maximizes routing accuracy
        # (91.5%, vs 85.6% for the old raw-score/0.5 setup). Retune by
        # re-running that sweep whenever calibration_params.json or
        # labeled_queries.json change.
        self.selection_threshold = 0.20

        # FIX (#10): the absolute-confidence uncertainty band (0.5-0.7,
        # applied downstream in pipeline.py) only fires when the WINNING
        # agent's own score lands in that range. It says nothing about how
        # close the runner-up was. A query where math=0.42 and document=0.40
        # is just as ambiguous as one where math=0.55 sits alone in the
        # uncertainty band -- arguably more so, since both candidates are
        # below the "uncertain" band entirely and neither flag would fire
        # under the old logic. This margin is compared against the gap
        # between the top score and the second-best score, independent of
        # where either falls on the absolute scale, and is surfaced in
        # routing_result so callers can treat a margin-thin win differently
        # from a clear win even when priority_order had to break the tie.
        self.ambiguous_margin = 0.05

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
        runner_up_score = -1.0
        runner_up_agent = None
        
        # Iterate based on priority order to handle ties correctly
        for agent_name in self.priority_order:
            score = calibrated_scores.get(agent_name, 0.0)
            # If the score is higher, or if it's tied but this agent has higher priority
            # Actually, we want a significant margin to override priority.
            # If the best score is 0.8 and this one is 0.82, we take it.
            # But if best is math (0.8) and code is (0.8), math wins because it was checked first.
            if score > best_score:
                # The previous best (if any) becomes the runner-up candidate.
                if best_agent is not None and best_score > runner_up_score:
                    runner_up_score = best_score
                    runner_up_agent = best_agent
                best_score = score
                best_agent = agent_name
            elif score > runner_up_score:
                runner_up_score = score
                runner_up_agent = agent_name

        # FIX (#10): margin-based ambiguity, independent of priority_order
        # and independent of where best_score sits on the absolute scale.
        # A thin margin means the winner was decided more by priority_order
        # position than by the detectors actually disagreeing strongly.
        margin = best_score - runner_up_score if runner_up_agent is not None else best_score
        is_ambiguous = (
            best_agent is not None
            and best_score >= self.selection_threshold
            and margin < self.ambiguous_margin
        )
        if is_ambiguous:
            logger.info(
                f"[MasterRouter] Ambiguous routing: '{best_agent}' ({best_score:.3f}) beat "
                f"'{runner_up_agent}' ({runner_up_score:.3f}) by only {margin:.3f} "
                f"(< ambiguous_margin={self.ambiguous_margin}). priority_order decided this tie."
            )

        routing_result = {
            "selected_agent": best_agent if best_score >= self.selection_threshold else None,
            "confidence": best_score,
            "reasoning": reasoning_map.get(best_agent, "Fallback") if best_score >= self.selection_threshold else "Confidence too low, falling back",
            "all_scores": calibrated_scores,
            "fallback_agent": "general",
            # New fields (additive -- existing consumers that only read the
            # keys above are unaffected):
            "runner_up_agent": runner_up_agent,
            "runner_up_score": runner_up_score if runner_up_agent is not None else None,
            "margin": margin,
            "ambiguous": is_ambiguous,
        }
        
        # Telemetry
        AgentTelemetry.log_routing_decision(query, routing_result)
        
        return routing_result

master_router_instance = MasterRouter()