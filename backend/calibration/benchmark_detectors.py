"""
Benchmark routing approaches on a held-out test split of
calibration/labeled_queries.json:

  1. Regex/heuristic ensemble  -- MasterRouter's detectors + ConfidenceCalibrator
                                   (the system as currently shipped)
  2. TF-IDF + Logistic Regression -- a trained multiclass classifier
                                      (calibration/baseline_classifiers.py)
  3. LLM classifier (Groq)     -- single-call LLM routing (requires GROQ_API_KEY)

For each approach, reports:
  - overall accuracy on the test split
  - per-class precision / recall / F1
  - mean latency per query

Usage:
    python calibration/benchmark_detectors.py
    python calibration/benchmark_detectors.py --test-size 0.3 --seed 42
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Load .env early so GROQ_API_KEY is available before importing LLM code
# --------------------------------------------------------------------------- #

import time

def llm_predict_with_retry(llm_clf, query, context, retries=5):
    delay = 0.2
    for attempt in range(retries):
        try:
            return llm_clf.predict(query, context)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("Groq rate limit persisted after retries")

def load_env_file() -> None:
    """
    Lightweight .env loader:
    - looks for .env in backend/ (parent of calibration/)
    - also checks current working directory as a fallback
    - supports simple KEY=VALUE lines
    - does not override existing environment variables
    """
    script_dir = Path(__file__).resolve().parent
    backend_dir = script_dir.parent

    candidates = [
        backend_dir / ".env",   # backend/.env
        Path.cwd() / ".env",    # fallback
    ]

    env_path = next((p for p in candidates if p.exists()), None)
    if env_path is None:
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as e:
        print(f"[WARN] Could not load .env file: {e}")


load_env_file()

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

from baseline_classifiers import TfidfLogRegClassifier, LLMRoutingClassifier, AGENT_NAMES  # noqa: E402

PRIORITY_ORDER = ["math", "code", "data", "document", "writing", "research", "knowledge"]
CALIBRATED_SELECTION_THRESHOLD = 0.20


# --------------------------------------------------------------------------- #
# Regex/heuristic ensemble (matches MasterRouter logic exactly)
# --------------------------------------------------------------------------- #

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


def regex_predict(query: str, context: dict, detectors: dict, calibrator: ConfidenceCalibrator) -> str:
    raw_scores = {}
    for name, det in detectors.items():
        if name in ("math", "code", "writing", "knowledge"):
            result = det.detect(query)
        else:
            result = det.detect(query, context)
        raw_scores[name] = float(result.get("confidence", 0.0))

    calibrated = calibrator.calibrate(raw_scores)

    best_agent, best_score = None, -1.0
    for name in PRIORITY_ORDER:
        score = calibrated.get(name, 0.0)
        if score > best_score:
            best_score = score
            best_agent = name

    return best_agent if best_score >= CALIBRATED_SELECTION_THRESHOLD else "none"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def precision_recall_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Dict[str, float]]:
    metrics = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(1 for t in y_true if t == label)

        metrics[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return metrics


def print_report(name: str, y_true: List[str], y_pred: List[str], latencies: List[float]):
    n = len(y_true)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    macro_f1_vals = []

    print(f"\n{'=' * 70}")
    print(f"{name}")
    print(f"{'=' * 70}")
    print(f"Accuracy: {accuracy:.1%}  ({sum(1 for t, p in zip(y_true, y_pred) if t == p)}/{n})")
    if latencies:
        mean_lat = sum(latencies) / len(latencies)
        print(f"Mean latency: {mean_lat * 1000:.3f} ms/query")

    print(f"\n{'Class':<10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
    metrics = precision_recall_f1(y_true, y_pred, AGENT_NAMES)
    for label, m in metrics.items():
        if m["support"] == 0:
            continue
        macro_f1_vals.append(m["f1"])
        print(f"{label:<10} {m['precision']:>10.2f} {m['recall']:>8.2f} {m['f1']:>8.2f} {m['support']:>8}")

    if macro_f1_vals:
        print(f"\nMacro F1: {sum(macro_f1_vals) / len(macro_f1_vals):.3f}")

    return accuracy


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def train_test_split(examples: List[dict], test_size: float, seed: int) -> Tuple[List[dict], List[dict]]:
    import random
    rng = random.Random(seed)

    by_label = defaultdict(list)
    for ex in examples:
        by_label[ex["gold_agent"]].append(ex)

    train, test = [], []
    for _, items in by_label.items():
        items = items[:]
        rng.shuffle(items)
        n_test = max(1, int(round(len(items) * test_size)))
        test.extend(items[:n_test])
        train.extend(items[n_test:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def main():
    parser = argparse.ArgumentParser(description="Benchmark routing approaches on labeled_queries.json")
    parser.add_argument(
        "--data",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "labeled_queries.json"),
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        examples = json.load(f)["examples"]

    train_examples, test_examples = train_test_split(examples, args.test_size, args.seed)
    print(f"Total examples: {len(examples)}  |  Train: {len(train_examples)}  |  Test: {len(test_examples)}")

    y_true = [ex["gold_agent"] for ex in test_examples]

    # --- 1. Regex/heuristic ensemble --------------------------------------- #
    detectors = build_detectors()
    calibrator = ConfidenceCalibrator()

    y_pred_regex, lat_regex = [], []
    for ex in test_examples:
        t0 = time.perf_counter()
        pred = regex_predict(ex["query"], ex.get("context", {}), detectors, calibrator)
        lat_regex.append(time.perf_counter() - t0)
        y_pred_regex.append(pred)

    acc_regex = print_report(
        "1. Regex/heuristic ensemble (current system: detectors + Platt calibration)",
        y_true,
        y_pred_regex,
        lat_regex,
    )

    # --- 2. TF-IDF + Logistic Regression ----------------------------------- #
    train_queries = [ex["query"] for ex in train_examples]
    train_labels = [ex["gold_agent"] for ex in train_examples]

    clf = TfidfLogRegClassifier()
    clf.fit(train_queries, train_labels)

    y_pred_tfidf, lat_tfidf = [], []
    for ex in test_examples:
        t0 = time.perf_counter()
        pred = clf.predict(ex["query"])
        lat_tfidf.append(time.perf_counter() - t0)
        y_pred_tfidf.append(pred)

    acc_tfidf = print_report(
        f"2. TF-IDF + Logistic Regression (trained on {len(train_examples)} examples)",
        y_true,
        y_pred_tfidf,
        lat_tfidf,
    )

    results_summary = [
        ("Regex/heuristic ensemble", acc_regex, sum(lat_regex) / len(lat_regex) * 1000),
        ("TF-IDF + LogReg", acc_tfidf, sum(lat_tfidf) / len(lat_tfidf) * 1000),
    ]

    # --- 3. LLM classifier (Groq) ------------------------------------------ #
    try:
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is missing. Add it to backend/.env.")

        llm_clf = LLMRoutingClassifier()

        y_pred_llm, lat_llm = [], []
        for ex in test_examples:
            t0 = time.perf_counter()
            # pred = llm_clf.predict(ex["query"], ex.get("context", {}))
            pred = llm_predict_with_retry(llm_clf, ex["query"], ex.get("context", {}))
            lat_llm.append(time.perf_counter() - t0)
            y_pred_llm.append(pred)

        acc_llm = print_report(
            f"3. LLM classifier (Groq, {len(test_examples)} test queries)",
            y_true,
            y_pred_llm,
            lat_llm,
        )
        results_summary.append(("LLM classifier (Groq)", acc_llm, sum(lat_llm) / len(lat_llm) * 1000))
    except Exception as e:
        print(f"\n[LLM baseline skipped] {e}")

    # --- Summary table ------------------------------------------------------ #
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Method':<40} {'Accuracy':>10} {'Latency (ms)':>14}")
    for name, acc, lat in results_summary:
        print(f"{name:<40} {acc:>10.1%} {lat:>14.4f}")


if __name__ == "__main__":
    main()