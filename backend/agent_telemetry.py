import time
import sys
import json
from typing import Dict, Any
from logger import logger

class AgentTelemetry:
    """
    Logs agent routing decisions, confidence scores, and execution times to stdout/stderr.
    Useful for tuning the routing thresholds over time.
    """
    
    @staticmethod
    def log_routing_decision(query: str, routing_result: Dict[str, Any]):
        """
        Log the outcome of the master router's decision.
        """
        log_entry = {
            "event": "routing_decision",
            "timestamp": time.time(),
            "query_preview": query[:100] + "..." if len(query) > 100 else query,
            "selected_agent": routing_result.get("selected_agent"),
            "confidence": routing_result.get("confidence"),
            "reasoning": routing_result.get("reasoning"),
            "all_scores": routing_result.get("all_scores")
        }
        logger.info(f"[TELEMETRY] {json.dumps(log_entry)}")

    @staticmethod
    def log_agent_execution(agent_name: str, duration_ms: float, success: bool):
        """
        Log the execution time and success status of a specific agent.
        """
        log_entry = {
            "event": "agent_execution",
            "timestamp": time.time(),
            "agent": agent_name,
            "duration_ms": duration_ms,
            "success": success
        }
        logger.info(f"[TELEMETRY] {json.dumps(log_entry)}")
