"""
telemetry_api.py  (v2 — with auth)
------------------------------------
FastAPI router serving aggregated analytics from:
  data/telemetry/routing_decisions.jsonl
  data/telemetry/agent_executions.jsonl

Mount in api.py:

    from telemetry_auth import auth_router
    from telemetry_api  import router as telemetry_router
    app.include_router(auth_router,     prefix="/telemetry")
    app.include_router(telemetry_router, prefix="/telemetry")

All data endpoints are protected by `require_auth` from telemetry_auth.py.
The /auth/* endpoints are public (they're the login gate).
"""

import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query

from telemetry_auth import require_auth

_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "telemetry")
ROUTING_LOG   = os.path.join(_BASE, "routing_decisions.jsonl")
EXECUTION_LOG = os.path.join(_BASE, "agent_executions.jsonl")

router = APIRouter(tags=["telemetry"], dependencies=[Depends(require_auth)])

AGENTS = ["math", "code", "data", "document", "writing", "research", "knowledge"]


# ── helpers ────────────────────────────────────────────────────────────────────

def _read_jsonl(path: str, since: Optional[float] = None) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if since is None or rec.get("timestamp", 0) >= since:
                    records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


# ── endpoints ──────────────────────────────────────────────────────────────────

@router.get("/summary")
def summary(since_hours: float = Query(24.0)):
    since = time.time() - since_hours * 3600
    routing    = _read_jsonl(ROUTING_LOG, since)
    executions = _read_jsonl(EXECUTION_LOG, since)

    total_queries = len(routing)
    fallbacks     = sum(1 for r in routing if r.get("selected_agent") is None)
    confidences   = sorted(r["confidence"] for r in routing if r.get("confidence") is not None)

    agent_counts: Dict[str, int] = defaultdict(int)
    for r in routing:
        agent_counts[r.get("selected_agent") or "fallback"] += 1

    latency_by_agent: Dict[str, List[float]] = defaultdict(list)
    failure_by_agent: Dict[str, int]         = defaultdict(int)
    for e in executions:
        a = e.get("agent", "unknown")
        latency_by_agent[a].append(e.get("duration_ms", 0.0))
        if not e.get("success", True):
            failure_by_agent[a] += 1

    latency_stats: Dict[str, Dict] = {}
    for agent, lats in latency_by_agent.items():
        s = sorted(lats)
        n = len(s)
        latency_stats[agent] = {
            "count":         n,
            "mean_ms":       round(sum(s) / n, 1) if n else 0,
            "p50_ms":        round(s[int(n * 0.50)], 1) if n else 0,
            "p95_ms":        round(s[int(n * 0.95)], 1) if n else 0,
            "failure_count": failure_by_agent.get(agent, 0),
        }

    n_c = len(confidences)
    return {
        "window_hours":      since_hours,
        "total_queries":     total_queries,
        "fallback_count":    fallbacks,
        "fallback_rate":     round(fallbacks / total_queries, 4) if total_queries else 0,
        "confidence": {
            "mean": round(sum(confidences) / n_c, 4) if n_c else None,
            "p50":  round(confidences[int(n_c * 0.50)], 4) if n_c else None,
            "p95":  round(confidences[int(n_c * 0.95)], 4) if n_c else None,
            "min":  round(confidences[0], 4) if n_c else None,
            "max":  round(confidences[-1], 4) if n_c else None,
        },
        "agent_distribution": dict(agent_counts),
        "latency_by_agent":   latency_stats,
    }


@router.get("/routing/timeseries")
def routing_timeseries(since_hours: float = Query(24.0)):
    since   = time.time() - since_hours * 3600
    routing = _read_jsonl(ROUTING_LOG, since)

    series: Dict[str, Dict[str, int]] = {}
    for r in routing:
        hour  = time.strftime("%Y-%m-%dT%H:00", time.gmtime(r.get("timestamp", 0)))
        agent = r.get("selected_agent") or "fallback"
        series.setdefault(hour, {})
        series[hour][agent] = series[hour].get(agent, 0) + 1

    all_agents = AGENTS + ["fallback"]
    for hour in series:
        for a in all_agents:
            series[hour].setdefault(a, 0)

    return {"series": dict(sorted(series.items()))}


@router.get("/routing/confidence_distribution")
def confidence_distribution(
    since_hours: float = Query(24.0),
    bins: int = Query(20, ge=5, le=100),
):
    since   = time.time() - since_hours * 3600
    routing = _read_jsonl(ROUTING_LOG, since)

    values = [r["confidence"] for r in routing if r.get("confidence") is not None]
    if not values:
        return {"bins": [], "counts": []}

    width  = 1.0 / bins
    counts = [0] * bins
    for v in values:
        counts[min(int(v / width), bins - 1)] += 1

    bin_edges = [round(i * width, 3) for i in range(bins + 1)]
    return {"bin_edges": bin_edges, "counts": counts}


@router.get("/routing/score_heatmap")
def score_heatmap(since_hours: float = Query(24.0)):
    since   = time.time() - since_hours * 3600
    routing = _read_jsonl(ROUTING_LOG, since)

    rows = []
    for r in routing[-200:]:
        scores = r.get("all_scores") or {}
        rows.append({
            "ts":         r.get("timestamp"),
            "selected":   r.get("selected_agent"),
            "confidence": r.get("confidence"),
            "scores":     {a: round(scores.get(a, 0.0), 4) for a in AGENTS},
        })
    return {"queries": rows, "agents": AGENTS}


@router.get("/agents/latency")
def agent_latency(
    since_hours: float = Query(24.0),
    agent: Optional[str] = Query(None),
):
    since      = time.time() - since_hours * 3600
    executions = _read_jsonl(EXECUTION_LOG, since)

    by_agent: Dict[str, List[Dict]] = defaultdict(list)
    for e in executions:
        a = e.get("agent", "unknown")
        if agent and a != agent:
            continue
        by_agent[a].append({
            "ts":          e.get("timestamp"),
            "duration_ms": e.get("duration_ms"),
            "success":     e.get("success", True),
        })

    return {"data": {a: pts[-500:] for a, pts in by_agent.items()}}


@router.get("/agents/error_rate")
def agent_error_rate(since_hours: float = Query(24.0)):
    since      = time.time() - since_hours * 3600
    executions = _read_jsonl(EXECUTION_LOG, since)

    stats: Dict[str, Dict] = {}
    for e in executions:
        a = e.get("agent", "unknown")
        stats.setdefault(a, {"success": 0, "failure": 0})
        if e.get("success", True):
            stats[a]["success"] += 1
        else:
            stats[a]["failure"] += 1

    for a, s in stats.items():
        t = s["success"] + s["failure"]
        s["total"]      = t
        s["error_rate"] = round(s["failure"] / t, 4) if t else 0.0

    return {"agents": stats}


@router.get("/recent_decisions")
def recent_decisions(
    limit:       int   = Query(50, ge=1, le=500),
    since_hours: float = Query(24.0),
    agent: Optional[str] = Query(None),
):
    since   = time.time() - since_hours * 3600
    routing = _read_jsonl(ROUTING_LOG, since)

    if agent:
        routing = [r for r in routing if r.get("selected_agent") == agent]

    routing.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
    return {
        "decisions": [
            {
                "ts":             r.get("timestamp"),
                "query_preview":  r.get("query_preview") or r.get("query", "")[:100],
                "selected_agent": r.get("selected_agent"),
                "confidence":     round(r.get("confidence", 0), 4),
                "reasoning":      r.get("reasoning", ""),
            }
            for r in routing[:limit]
        ]
    }


@router.get("/health")
def telemetry_health():
    def info(path: str) -> Dict:
        if not os.path.exists(path):
            return {"exists": False, "size_bytes": 0, "records": 0}
        size    = os.path.getsize(path)
        records = sum(1 for line in open(path, encoding="utf-8") if line.strip())
        return {"exists": True, "size_bytes": size, "records": records}

    return {
        "routing_log":   info(ROUTING_LOG),
        "execution_log": info(EXECUTION_LOG),
    }