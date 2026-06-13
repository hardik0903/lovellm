from typing import Dict

class ConfidenceCalibrator:
    """
    Normalizes confidence scores across different agents so they are comparable.
    Raw scores from different detectors (regex vs LLM vs heuristics) are often on different scales.
    """
    def __init__(self):
        # Base multipliers or thresholds per agent type, if needed.
        # These can be tuned based on telemetry.
        self.weights = {
            "math": 1.0,
            "code": 1.0,
            "data": 1.0,
            "document": 1.0,
            "writing": 1.0,
            "research": 1.0,
            "knowledge": 1.0
        }

    def calibrate(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Applies calibration weights to raw confidence scores.
        """
        calibrated = {}
        for agent_name, raw_score in raw_scores.items():
            weight = self.weights.get(agent_name, 1.0)
            calibrated[agent_name] = min(1.0, raw_score * weight)
        return calibrated
