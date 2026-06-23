"""
train_calibrator.py
--------------------
Fits a per-agent Platt-scaling (1D logistic regression) calibration model
that maps a detector's RAW heuristic confidence score to a CALIBRATED
probability that the agent is actually the correct route for the query.

Why this exists
================
The detectors (math_detector, code_detector, ...) each produce a "confidence"
score in [0, 1] using ad-hoc, hand-tuned additive heuristics (keyword hits,
regex matches, base offsets). These scores are NOT probabilities: a score of
0.7 from MathDetector and a score of 0.7 from WritingDetector do not mean the
same thing, because the two detectors were built with different scoring
scales and different numbers of contributing signals.

Platt scaling fits, for each agent, a sigmoid:

    P(correct_agent | raw_score) = sigmoid(a * raw_score + b)

using maximum-likelihood logistic regression on a labeled dataset of
(query, gold_agent) pairs. The fitted (a, b) per agent are saved to
calibration_params.json and loaded by ConfidenceCalibrator at runtime.

This turns "confidence calibration" from a documented-but-fake no-op
(weights all 1.0) into a real, measurable, retrainable component:
 - it is FIT from data (calibration/labeled_queries.json)
 - it is EVALUATED with standard calibration metrics (Brier score, ECE)
 - it is PERSISTED (calibration_params.json) and reproducible
 - it can be RETRAINED as more labeled/telemetry data accumulates

Usage
=====
    python train_calibrator.py
    python train_calibrator.py --data calibration/labeled_queries.json \
                                --out calibration_params.json \
                                --epochs 2000 --lr 0.1

Output
======
 - calibration_params.json   : {agent_name: {"a": float, "b": float, "n": int}}
 - Printed report             : per-agent Brier score & ECE, before vs after
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple

# Make sibling modules importable when run from the backend/ directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOTE: We import the individual *detector* classes directly rather than the
# full AgentRegistry / agent classes. The agents themselves pull in heavy,
# network-bound dependencies (groq, chromadb, sentence-transformers, ddgs,
# ...) that are irrelevant to routing calibration -- only the lightweight
# regex/heuristic `detect()` methods matter here, and those live in the
# *_detector modules with no heavy imports.
from math_detector import MathDetector  # noqa: E402
from code_detector import CodeDetector  # noqa: E402
from data_detector import DataDetector  # noqa: E402
from document_detector import DocumentDetector  # noqa: E402
from writing_detector import WritingDetector  # noqa: E402
from research_detector import ResearchDetector  # noqa: E402
from knowledge_detector import KnowledgeDetector  # noqa: E402

AGENT_NAMES = ["math", "code", "data", "document", "writing", "research", "knowledge"]


def build_detectors() -> Dict[str, object]:
    """Instantiate every detector. Signatures differ slightly (some take
    `context`, some don't), so calls are normalized in build_training_pairs."""
    return {
        "math": MathDetector(),
        "code": CodeDetector(),
        "data": DataDetector(),
        "document": DocumentDetector(),
        "writing": WritingDetector(),
        "research": ResearchDetector(),
        "knowledge": KnowledgeDetector(),
    }


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #

def load_dataset(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["examples"]


def build_training_pairs(
    examples: List[dict], detectors: Dict[str, object]
) -> Dict[str, List[Tuple[float, int]]]:
    """
    For every (query, gold_agent) example, run every agent's detector and
    record (raw_score, label) where label = 1 if that agent is the gold
    agent for this query, else 0.

    Returns: {agent_name: [(raw_score, label), ...]}
    """
    pairs: Dict[str, List[Tuple[float, int]]] = {name: [] for name in AGENT_NAMES}

    for ex in examples:
        query = ex["query"]
        gold = ex["gold_agent"]
        context = ex.get("context", {})

        for agent_name in AGENT_NAMES:
            detector = detectors.get(agent_name)
            if detector is None:
                continue
            try:
                # MathDetector.detect(query), WritingDetector.detect(query)
                # take only a query; Data/Document/Research detectors also
                # accept an optional context dict.
                if agent_name in ("math", "code", "writing", "knowledge"):
                    detection = detector.detect(query)
                else:
                    detection = detector.detect(query, context)
                raw_score = float(detection.get("confidence", 0.0))
            except Exception as e:
                print(f"  [warn] detector error for {agent_name} on {query!r}: {e}")
                raw_score = 0.0

            label = 1 if gold == agent_name else 0
            pairs[agent_name].append((raw_score, label))

    return pairs


# --------------------------------------------------------------------------- #
# Platt scaling: 1D logistic regression fit via gradient descent              #
# --------------------------------------------------------------------------- #

def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def fit_platt(
    pairs: List[Tuple[float, int]], epochs: int = 2000, lr: float = 0.1, l2: float = 1e-4
) -> Tuple[float, float]:
    """
    Fit P(y=1 | x) = sigmoid(a*x + b) by minimizing binary cross-entropy
    with a small L2 penalty on `a` (ridge regularization), via batch
    gradient descent. Pure-Python, no external ML dependency.

    Returns (a, b).
    """
    n = len(pairs)
    if n == 0:
        return 1.0, 0.0

    # Initialize a=1, b=0 -> identity-ish starting point (matches the old
    # no-op behaviour before any training signal is applied).
    a, b = 1.0, 0.0

    for _ in range(epochs):
        grad_a, grad_b = 0.0, 0.0
        for x, y in pairs:
            p = sigmoid(a * x + b)
            err = p - y
            grad_a += err * x
            grad_b += err
        grad_a = grad_a / n + l2 * a
        grad_b = grad_b / n

        a -= lr * grad_a
        b -= lr * grad_b

    return a, b


# --------------------------------------------------------------------------- #
# Calibration quality metrics                                                 #
# --------------------------------------------------------------------------- #

def brier_score(pairs: List[Tuple[float, int]], a: float = 1.0, b: float = 0.0,
                 use_sigmoid: bool = False) -> float:
    """
    Mean squared error between predicted probability and the binary label.
    Lower is better. 0 = perfect, 0.25 = uninformative (always predict 0.5).
    """
    if not pairs:
        return float("nan")
    total = 0.0
    for x, y in pairs:
        p = sigmoid(a * x + b) if use_sigmoid else x
        p = min(max(p, 0.0), 1.0)
        total += (p - y) ** 2
    return total / len(pairs)


def expected_calibration_error(
    pairs: List[Tuple[float, int]], a: float = 1.0, b: float = 0.0,
    use_sigmoid: bool = False, n_bins: int = 5
) -> float:
    """
    ECE: bins predictions into n_bins buckets by predicted probability,
    and measures the weighted average gap between mean predicted
    probability and observed accuracy (fraction of positives) in each bin.
    Lower is better. 0 = perfectly calibrated.
    """
    if not pairs:
        return float("nan")

    bins = [[] for _ in range(n_bins)]
    for x, y in pairs:
        p = sigmoid(a * x + b) if use_sigmoid else x
        p = min(max(p, 0.0), 1.0 - 1e-9)
        idx = int(p * n_bins)
        bins[idx].append((p, y))

    n = len(pairs)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        mean_pred = sum(p for p, _ in bucket) / len(bucket)
        mean_actual = sum(y for _, y in bucket) / len(bucket)
        ece += (len(bucket) / n) * abs(mean_pred - mean_actual)
    return ece


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train Platt-scaling calibration for routing confidence.")
    parser.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "labeled_queries.json"))
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration_params.json"))
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    print(f"Loading labeled dataset from {args.data} ...")
    examples = load_dataset(args.data)
    print(f"  {len(examples)} labeled queries")

    label_counts = {}
    for ex in examples:
        label_counts[ex["gold_agent"]] = label_counts.get(ex["gold_agent"], 0) + 1
    print(f"  Label distribution: {label_counts}")

    print("\nRunning all detectors on every example to build training pairs...")
    detectors = build_detectors()
    pairs_by_agent = build_training_pairs(examples, detectors)

    params = {}
    print("\n" + "=" * 78)
    print(f"{'Agent':<10} {'n':>4} {'pos':>4}   {'Brier(raw)':>11} {'Brier(cal)':>11}   {'ECE(raw)':>9} {'ECE(cal)':>9}   a,b")
    print("-" * 78)

    for agent_name in AGENT_NAMES:
        pairs = pairs_by_agent[agent_name]
        n = len(pairs)
        n_pos = sum(y for _, y in pairs)

        a, b = fit_platt(pairs, epochs=args.epochs, lr=args.lr)

        brier_raw = brier_score(pairs, use_sigmoid=False)
        brier_cal = brier_score(pairs, a=a, b=b, use_sigmoid=True)
        ece_raw = expected_calibration_error(pairs, use_sigmoid=False)
        ece_cal = expected_calibration_error(pairs, a=a, b=b, use_sigmoid=True)

        params[agent_name] = {"a": round(a, 4), "b": round(b, 4), "n": n, "n_positive": n_pos}

        print(f"{agent_name:<10} {n:>4} {n_pos:>4}   {brier_raw:>11.4f} {brier_cal:>11.4f}   "
              f"{ece_raw:>9.4f} {ece_cal:>9.4f}   a={a:.3f}, b={b:.3f}")

    print("=" * 78)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"\nWrote calibration parameters to {args.out}")
    print("\nInterpretation:")
    print("  - Brier score: mean squared error of predicted prob vs actual label (lower=better, 0.25=uninformative baseline)")
    print("  - ECE: expected calibration error, weighted gap between predicted prob and observed accuracy per bin (lower=better)")
    print("  - 'cal' columns should be <= 'raw' columns for agents where calibration helped.")
    print("  - (a, b) define: calibrated = sigmoid(a * raw_score + b)")


if __name__ == "__main__":
    main()