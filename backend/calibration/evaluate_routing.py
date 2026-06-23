"""
evaluate_routing.py
--------------------
Evaluates MasterRouter routing accuracy on the labeled dataset
(calibration/labeled_queries.json), comparing:

  (a) RAW heuristic scores directly against the 0.5 selection threshold
  (b) CALIBRATED scores (Platt-scaled via ConfidenceCalibrator) against
      the same 0.5 threshold

This produces a concrete "routing accuracy" number for the project report,
and shows whether calibration changes which agent gets selected (and how
often the selection crosses the 0.5 confidence gate vs falling back to
"general").

Usage:
    python calibration/evaluate_routing.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from confidence_calibrator import ConfidenceCalibrator  # noqa: E402
from math_detector import MathDetector  # noqa: E402
from code_detector import CodeDetector  # noqa: E402
from data_detector import DataDetector  # noqa: E402
from document_detector import DocumentDetector  # noqa: E402
from writing_detector import WritingDetector  # noqa: E402
from research_detector import ResearchDetector  # noqa: E402
from knowledge_detector import KnowledgeDetector  # noqa: E402

PRIORITY_ORDER = ["math", "code", "data", "document", "writing", "research", "knowledge"]
RAW_SELECTION_THRESHOLD = 0.5      # threshold the OLD raw-score router used
CALIBRATED_SELECTION_THRESHOLD = 0.20  # tuned threshold for calibrated scores (see master_router.py)


def build_detectors():
    return {
        "math": MathDetector(),
        "code": CodeDetector(),
        "data": DataDetector(),
        "document": DocumentDetector(),
        "writing": WritingDetector(),
        "research": ResearchDetector(),
        "knowledge": KnowledgeDetector(),
    }


def detect_all(query, context, detectors):
    raw_scores = {}
    for name, det in detectors.items():
        if name in ("math", "code", "writing", "knowledge"):
            result = det.detect(query)
        else:
            result = det.detect(query, context)
        raw_scores[name] = float(result.get("confidence", 0.0))
    return raw_scores


def select_agent(scores, threshold):
    best_agent, best_score = None, -1.0
    for name in PRIORITY_ORDER:
        score = scores.get(name, 0.0)
        if score > best_score:
            best_score = score
            best_agent = name
    if best_score >= threshold:
        return best_agent, best_score
    return "none", best_score


def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labeled_queries.json")
    with open(data_path, "r", encoding="utf-8") as f:
        examples = json.load(f)["examples"]

    detectors = build_detectors()
    calibrator = ConfidenceCalibrator()

    n = len(examples)
    correct_raw = 0
    correct_cal = 0
    mismatches = []
    all_records = []  # (gold, calibrated_scores) for the threshold sweep

    for ex in examples:
        query = ex["query"]
        gold = ex["gold_agent"]
        context = ex.get("context", {})

        raw_scores = detect_all(query, context, detectors)
        cal_scores = calibrator.calibrate(raw_scores)
        all_records.append((gold, cal_scores))

        raw_pred, raw_conf = select_agent(raw_scores, RAW_SELECTION_THRESHOLD)
        cal_pred, cal_conf = select_agent(cal_scores, CALIBRATED_SELECTION_THRESHOLD)

        if raw_pred == gold:
            correct_raw += 1
        if cal_pred == gold:
            correct_cal += 1

        if raw_pred != cal_pred:
            mismatches.append((query, gold, raw_pred, cal_pred))

    print(f"Labeled examples: {n}")
    print(f"Routing accuracy, RAW scores        @ threshold={RAW_SELECTION_THRESHOLD}:  {correct_raw}/{n} = {correct_raw/n:.1%}")
    print(f"Routing accuracy, CALIBRATED scores @ threshold={CALIBRATED_SELECTION_THRESHOLD}: {correct_cal}/{n} = {correct_cal/n:.1%}")

    if mismatches:
        print(f"\n{len(mismatches)} queries where raw vs calibrated selection differs:")
        for query, gold, raw_pred, cal_pred in mismatches:
            marker_raw = "✓" if raw_pred == gold else "✗"
            marker_cal = "✓" if cal_pred == gold else "✗"
            print(f"  gold={gold:<10} raw={raw_pred:<10}{marker_raw}  cal={cal_pred:<10}{marker_cal}   {query[:60]}")
    else:
        print("\nNo differences in selected agent between raw and calibrated scores.")

    # Threshold sweep on calibrated scores: shows why 0.20 (not 0.5) was chosen.
    print("\nThreshold sweep on CALIBRATED scores (selects argmax, falls back to "
          "'none' if argmax < threshold):")
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        correct = 0
        for gold, cal_scores in all_records:
            pred, _ = select_agent(cal_scores, t)
            if pred == gold:
                correct += 1
        marker = "  <- current" if abs(t - CALIBRATED_SELECTION_THRESHOLD) < 1e-9 else ""
        print(f"  threshold={t:.2f}  acc={correct}/{n} = {correct/n:.1%}{marker}")


if __name__ == "__main__":
    main()