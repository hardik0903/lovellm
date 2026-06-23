import os
import time
import json
import threading
from typing import Dict, Any
from logger import logger

# Routing decisions are appended here (one JSON object per line). This file
# is the raw material for retraining the ConfidenceCalibrator: periodically,
# a human reviews a sample of these entries, assigns a `gold_agent` label,
# and merges them into calibration/labeled_queries.json, then re-runs
# calibration/train_calibrator.py.
TELEMETRY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "telemetry")
ROUTING_LOG_PATH = os.path.join(TELEMETRY_DIR, "routing_decisions.jsonl")
EXECUTION_LOG_PATH = os.path.join(TELEMETRY_DIR, "agent_executions.jsonl")

_write_lock = threading.Lock()


class AgentTelemetry:
    """
    Logs agent routing decisions and execution times to stdout (for live
    debugging) AND appends them as JSONL to disk (for offline analysis and
    calibrator retraining).

    Persistence is intentionally simple (append-only JSONL, one file per
    event type) rather than a database: it requires no new dependencies,
    survives process restarts, and is trivial to load with
    `pandas.read_json(path, lines=True)` or plain `json.loads` per line for
    analysis notebooks / the routing-accuracy report.

    Disk writes are best-effort: if they fail (e.g. read-only filesystem),
    telemetry still goes to stdout via `logger`, so this never breaks the
    request path.
    """

    @staticmethod
    def _append_jsonl(path: str, entry: Dict[str, Any]):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            line = json.dumps(entry, ensure_ascii=False)
            with _write_lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:
            logger.error(f"[AgentTelemetry] Failed to persist telemetry to {path}: {e}")

    @staticmethod
    def log_routing_decision(query: str, routing_result: Dict[str, Any]):
        """
        Log the outcome of the master router's decision, both to stdout
        and to data/telemetry/routing_decisions.jsonl.

        FIX (#1): now also logs margin/runner_up/ambiguous (added to
        routing_result by master_router.py's margin-based tie-break, see
        Fix #10) so a human reviewing this JSONL can specifically filter for
        "ambiguous": true entries -- the queries where the router picked a
        winner mostly because of priority_order position rather than a clear
        score advantage. Over time this turns "no logging/alerting today
        that tells you when a query should have hit a special-case rule but
        didn't" into a reviewable, queryable record instead of a permanently
        invisible failure mode.
        """
        log_entry = {
            "event": "routing_decision",
            "timestamp": time.time(),
            "query": query,
            "query_preview": query[:100] + "..." if len(query) > 100 else query,
            "selected_agent": routing_result.get("selected_agent"),
            "confidence": routing_result.get("confidence"),
            "reasoning": routing_result.get("reasoning"),
            "all_scores": routing_result.get("all_scores"),
            "runner_up_agent": routing_result.get("runner_up_agent"),
            "runner_up_score": routing_result.get("runner_up_score"),
            "margin": routing_result.get("margin"),
            "ambiguous": routing_result.get("ambiguous"),
        }
        logger.info(f"[TELEMETRY] {json.dumps({k: v for k, v in log_entry.items() if k != 'query'})}")
        AgentTelemetry._append_jsonl(ROUTING_LOG_PATH, log_entry)

    @staticmethod
    def log_agent_execution(agent_name: str, duration_ms: float, success: bool):
        """
        Log the execution time and success status of a specific agent, both
        to stdout and to data/telemetry/agent_executions.jsonl.
        """
        log_entry = {
            "event": "agent_execution",
            "timestamp": time.time(),
            "agent": agent_name,
            "duration_ms": duration_ms,
            "success": success,
        }
        logger.info(f"[TELEMETRY] {json.dumps(log_entry)}")
        AgentTelemetry._append_jsonl(EXECUTION_LOG_PATH, log_entry)