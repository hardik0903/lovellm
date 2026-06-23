import json
import math
import os
from typing import Dict, Optional

from logger import logger


class ConfidenceCalibrator:
    """
    Calibrates raw detector confidence scores into comparable probabilities.

    The legacy root-level calibrator used a no-op weight map. This version keeps
    the same public API but loads fitted Platt-scaling parameters when they are
    available and falls back to identity calibration when they are not.

    Backward-compatible behavior:
    - accepts `params_path=...`
    - exposes `params`
    - exposes `weights` for older callers
    - provides `reload()`
    """

    DEFAULT_CANDIDATE_PATHS = (
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration", "calibration_params.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_params.json"),
    )

    def __init__(self, params_path: Optional[str] = None):
        self.params_path = params_path or self._resolve_default_path()
        self.params: Dict[str, Dict[str, float]] = {}
        self.weights: Dict[str, float] = {
            "math": 1.0,
            "code": 1.0,
            "data": 1.0,
            "document": 1.0,
            "writing": 1.0,
            "research": 1.0,
            "knowledge": 1.0,
        }
        self._load_params()

    def _resolve_default_path(self) -> str:
        for candidate in self.DEFAULT_CANDIDATE_PATHS:
            if os.path.exists(candidate):
                return candidate
        return self.DEFAULT_CANDIDATE_PATHS[0]

    def _load_params(self) -> None:
        if not os.path.exists(self.params_path):
            logger.warning(
                f"[ConfidenceCalibrator] No calibration params found at {self.params_path}. "
                "Falling back to identity calibration."
            )
            self.params = {}
            return

        try:
            with open(self.params_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            self.params = {}
            for agent, values in loaded.items():
                if not isinstance(values, dict):
                    continue
                if "a" not in values or "b" not in values:
                    continue
                self.params[agent] = {"a": float(values["a"]), "b": float(values["b"])}

            logger.info(
                f"[ConfidenceCalibrator] Loaded Platt-scaling parameters for {len(self.params)} agents "
                f"from {self.params_path}"
            )
        except Exception as exc:
            logger.error(
                f"[ConfidenceCalibrator] Failed to load calibration params from {self.params_path}: {exc}. "
                "Falling back to identity calibration."
            )
            self.params = {}

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def calibrate(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Apply per-agent calibration when fitted parameters exist.

        When a fitted (a, b) pair is available:
            calibrated = sigmoid(a * raw + b)

        Otherwise the score passes through unchanged.
        """
        calibrated: Dict[str, float] = {}
        for agent_name, raw_score in raw_scores.items():
            try:
                raw_value = max(0.0, min(1.0, float(raw_score)))
            except Exception:
                raw_value = 0.0

            params = self.params.get(agent_name)
            if not params:
                calibrated[agent_name] = raw_value
                continue

            a = params["a"]
            b = params["b"]
            calibrated[agent_name] = max(0.0, min(1.0, self._sigmoid(a * raw_value + b)))

        return calibrated

    def reload(self) -> None:
        """Re-read the calibration file from disk."""
        self._load_params()
