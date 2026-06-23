"""
raptor.py — RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
====================================================================================

Builds a small abstraction tree on top of an already-chunked document:

    Level 0:  raw child chunks (from chunking.py — unchanged)
    Level 1:  GMM-clustered groups of level-0 chunks, each summarized by Ollama
    Level 2:  GMM-clustered groups of level-1 summaries, summarized again
    ...       stops when a level collapses to a single cluster, or max_levels
              is reached, or there are too few nodes left to cluster.

Summary nodes are written back into VectorStore / BM25Store as regular chunks
(same `add_chunks` contract as DocumentChunker output) carrying extra metadata:

    node_type:       "summary"
    raptor_level:     1, 2, 3, ...
    child_chunk_ids:  comma-separated chunk_ids this node summarizes
                      (chroma metadata must be flat scalars, so this is a
                      joined string rather than a list)

This makes summary nodes searchable through the exact same dense/BM25 search
calls the rest of the system already uses — no changes needed to
VectorStore.search / BM25Store.search. The retriever decides whether to
prefer summary nodes based on query complexity (see query_understanding.py's
three-way classifier and retriever.py's level weighting).

Soft clustering note: following the RAPTOR paper, we use a Gaussian Mixture
Model (not k-means) so a chunk that's genuinely relevant to two clusters can
be summarized into both — k-means' hard partitioning would force a single
assignment and lose that ambiguous-membership context. We pick the number of
components via BIC instead of fixing k a priori, since the "natural" number
of topics in a document is unknown and varies a lot by document length.

Simplification vs. the original RAPTOR paper: the paper reduces embedding
dimensionality with UMAP before clustering. We skip that here (one fewer
heavy dependency) and run GMM directly on the 384-dim MiniLM embeddings.
This is a reasonable trade for typical document sizes in this system
(tens to low hundreds of chunks per document) but may need revisiting if
GMM starts producing degenerate clusters on much larger corpora.
"""

from __future__ import annotations

import os
import time
import uuid
import json
import hashlib
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logger import logger

try:
    from sklearn.mixture import GaussianMixture
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAPTOR_MAX_LEVELS = 3            # hard cap so pathological docs can't recurse forever
RAPTOR_MIN_NODES_TO_CLUSTER = 4  # below this, just summarize everything into one node
RAPTOR_MIN_POINTS_PER_CLUSTER = 4  # keeps GMM's candidate k range from degenerating into near-singleton components
RAPTOR_MAX_CLUSTER_SIZE_FOR_K1 = 1  # if GMM picks k=1, that level is the top -> stop
RAPTOR_SOFT_ASSIGN_THRESHOLD = 0.15  # min posterior probability to join a cluster
RAPTOR_MAX_COMPONENTS_CAP = 12    # don't try more than this many GMM components
RAPTOR_SUMMARY_MAX_CHARS = 1800   # ~450 tokens; keeps summary nodes chunk-sized

# --- Model / VRAM configuration ---
# Use the same model as generator.py (llama3.2) so Ollama doesn't need to
# hold two large models in VRAM simultaneously. The previous default
# "llama3.1:8b" caused Ollama to juggle both weights at once, exhausting GPU
# memory. Unifying on "llama3.2" means the model is already warm from
# generation and the summarizer reuses the same loaded weights.
RAPTOR_OLLAMA_MODEL = "llama3.2"
RAPTOR_OLLAMA_HOST = "http://localhost:11434"

# Maximum number of chunks fed into a single summarization prompt.
# ---- RTX 3050 4 GB budget breakdown (why 4 is the right cap) ----
# The GPU has 4 096 MB total.  After Ollama loads llama3.2 (3B Q4_K_M ≈
# 2 000 MB) and the sentence-transformer stays pinned in VRAM (all-MiniLM-L6
# ≈ 90 MB), the KV-cache budget for any single inference call is roughly
# 4 096 - 2 000 - 90 - 200 (OS/driver overhead) ≈ 1 800 MB.
#
# KV-cache memory scales with context length:  with num_ctx=2048 llama3.2
# allocates ~450 MB for the cache.  A 2-chunk prompt at 2 400 chars/chunk
# ≈ 600 tokens of passages + ~150 tokens of system/user boilerplate = ~750
# total input tokens, well under 2048.  4 chunks ≈ 1 350 tokens — still
# safe.  12 chunks ≈ 3 750 tokens, which, even at num_ctx=4096, pushes the
# KV-cache to ~900 MB and triggers the 1.8 GB allocation error seen in logs.
#
# Cap at 4: worst-case prompt ≈ 1 350 input + 600 output = 1 950 tokens,
# num_ctx=2048, KV-cache ≈ 450 MB — leaves ~1 350 MB headroom for the
# forward pass and prevents the OOM.  GMM soft-assignment means no
# information is permanently lost: a chunk can still appear in a sibling
# cluster's summary.
RAPTOR_MAX_CLUSTER_INPUT_CHUNKS = 4

# Context-window limit passed to Ollama on every summarization call.
# Llama 3 models default to 8 192 tokens which reserves ~500 MB of VRAM
# per active request. 4 096 is more than enough for 12 chunks (~3 600
# tokens of input) plus the 600-token output budget, and cuts the VRAM
# footprint roughly in half compared to the default.
RAPTOR_OLLAMA_NUM_CTX = 2048  # ~450 MB KV-cache on RTX 3050 4GB; safe with 4-chunk cap

# ollama.Client (v0.6.2) has no per-call timeout override — chat() takes no
# timeout kwarg — so this single value is applied once at Client construction
# and covers every call made through that client, including summarization.
# Sized generously for a cold/CPU-bound local model; without ANY timeout
# (the library's own default), httpx waits indefinitely with no error and no
# log line, which is what caused tree builds to hang silently for minutes.
RAPTOR_OLLAMA_TIMEOUT = 120.0

# Persist RAPTOR trees locally so the same PDF can reuse the prior tree
# rather than rebuilding from scratch on every run.
RAPTOR_TREE_CACHE_DIR = os.getenv("RAPTOR_TREE_CACHE_DIR", "./data/raptor_trees")
RAPTOR_TREE_CACHE_VERSION = "v1"


@dataclass
class RaptorNode:
    """A single node in the RAPTOR tree (always written back as a chunk-shaped dict)."""
    node_id: str
    level: int
    text: str
    document_id: str
    child_chunk_ids: List[str] = field(default_factory=list)
    source_file: str = ""

    def to_chunk_dict(self) -> Dict[str, Any]:
        """Shape this node exactly like a DocumentChunker output dict so it can
        be passed straight into VectorStore.add_chunks / BM25Store.add_chunks."""
        return {
            "chunk_id": self.node_id,
            "document_id": self.document_id,
            "text": self.text,
            "chunk_type": "summary",
            "node_type": "summary",
            "raptor_level": self.level,
            # Chroma metadata values must be flat scalars (no lists), so the
            # child id list is serialized to a delimited string. Split on "|"
            # to recover it; chunk_ids never contain "|" (see chunking.py).
            "child_chunk_ids": "|".join(self.child_chunk_ids),
            "source_file": self.source_file,
        }


@dataclass
class RaptorTree:
    document_id: str
    levels: Dict[int, List[RaptorNode]] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)

    def all_nodes(self) -> List[RaptorNode]:
        out: List[RaptorNode] = []
        for lvl in sorted(self.levels.keys()):
            out.extend(self.levels[lvl])
        return out

    def max_level(self) -> int:
        return max(self.levels.keys()) if self.levels else 0

    def is_empty(self) -> bool:
        """True if this tree has no usable summary nodes — e.g. the document
        had too few chunks to cluster (tree-build was skipped, see
        RaptorTreeBuilder.build's early-return path) or every level ended up
        empty. Callers (rag_eval_v3.py's RaptorRetrieverWrapper) use this to
        decide whether to fall back to the standard hybrid retrieval path."""
        return not self.levels or all(len(nodes) == 0 for nodes in self.levels.values())




def _safe_path_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
    return value.strip("._-") or "document"


def _stable_hash(*parts: Any) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        if part is None:
            chunk = ""
        elif isinstance(part, (dict, list, tuple)):
            chunk = json.dumps(part, sort_keys=True, ensure_ascii=False, default=str)
        else:
            chunk = str(part)
        hasher.update(chunk.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _normalize_text_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def compute_raptor_fingerprint(chunk_ids: List[str], chunk_texts: List[str]) -> str:
    """Stable identity for a chunk set. If the same PDF is ingested again and
    yields the same chunking, the RAPTOR tree cache path will be identical."""
    items = []
    for cid, txt in sorted(zip(chunk_ids, chunk_texts), key=lambda x: x[0]):
        items.append({"chunk_id": str(cid), "text": _normalize_text_for_hash(txt)})
    return _stable_hash(RAPTOR_TREE_CACHE_VERSION, items)[:24]


def get_raptor_tree_dir(document_id: str, fingerprint: str, cache_dir: str = RAPTOR_TREE_CACHE_DIR) -> Path:
    return Path(cache_dir) / _safe_path_component(document_id) / fingerprint


def _node_to_dict(node: RaptorNode) -> Dict[str, Any]:
    return {
        "node_id": node.node_id,
        "level": node.level,
        "text": node.text,
        "document_id": node.document_id,
        "child_chunk_ids": list(node.child_chunk_ids),
        "source_file": node.source_file,
    }


def _node_from_dict(payload: Dict[str, Any]) -> RaptorNode:
    return RaptorNode(
        node_id=payload["node_id"],
        level=int(payload["level"]),
        text=payload.get("text", ""),
        document_id=payload.get("document_id", ""),
        child_chunk_ids=list(payload.get("child_chunk_ids", [])),
        source_file=payload.get("source_file", ""),
    )


def serialize_raptor_tree(tree: RaptorTree) -> Dict[str, Any]:
    levels: Dict[str, List[Dict[str, Any]]] = {}
    for level, nodes in tree.levels.items():
        levels[str(level)] = [_node_to_dict(node) for node in nodes]
    return {
        "schema_version": 1,
        "document_id": tree.document_id,
        "levels": levels,
        "stats": tree.stats,
    }


def deserialize_raptor_tree(payload: Dict[str, Any]) -> RaptorTree:
    tree = RaptorTree(document_id=payload.get("document_id", ""))
    levels: Dict[int, List[RaptorNode]] = {}
    for level_str, nodes in payload.get("levels", {}).items():
        levels[int(level_str)] = [_node_from_dict(node) for node in nodes]
    tree.levels = levels
    tree.stats = payload.get("stats", {}) or {}
    return tree


def save_raptor_tree_artifacts(
    tree: RaptorTree,
    fingerprint: str,
    source_file: str = "",
    cache_dir: str = RAPTOR_TREE_CACHE_DIR,
) -> Dict[str, str]:
    """Persist the tree as JSON plus human-readable visualization files."""
    tree_dir = get_raptor_tree_dir(tree.document_id, fingerprint, cache_dir=cache_dir)
    tree_dir.mkdir(parents=True, exist_ok=True)

    payload = serialize_raptor_tree(tree)
    payload["metadata"] = {
        "fingerprint": fingerprint,
        "source_file": source_file,
        "cache_version": RAPTOR_TREE_CACHE_VERSION,
        "levels_built": len(tree.levels),
        "summary_nodes": sum(len(v) for v in tree.levels.values()),
    }

    tree_json = tree_dir / "tree.json"
    manifest_json = tree_dir / "manifest.json"
    mermaid_path = tree_dir / "tree.mmd"
    dot_path = tree_dir / "tree.dot"

    with open(tree_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(payload["metadata"], f, ensure_ascii=False, indent=2)
    with open(mermaid_path, "w", encoding="utf-8") as f:
        f.write(render_raptor_tree_mermaid(tree))
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(render_raptor_tree_dot(tree))

    return {
        "tree_json": str(tree_json),
        "manifest_json": str(manifest_json),
        "mermaid": str(mermaid_path),
        "dot": str(dot_path),
    }


def list_cached_raptor_trees(document_id: str, cache_dir: str = RAPTOR_TREE_CACHE_DIR) -> List[Dict[str, Any]]:
    doc_dir = Path(cache_dir) / _safe_path_component(document_id)
    if not doc_dir.exists():
        return []

    records: List[Dict[str, Any]] = []
    for fp_dir in sorted([p for p in doc_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        manifest = fp_dir / "manifest.json"
        record: Dict[str, Any] = {
            "document_id": document_id,
            "fingerprint": fp_dir.name,
            "path": str(fp_dir),
        }
        if manifest.exists():
            try:
                with open(manifest, "r", encoding="utf-8") as f:
                    record.update(json.load(f))
            except Exception:
                pass
        record["modified_time"] = fp_dir.stat().st_mtime
        records.append(record)
    return records


def load_cached_raptor_tree(
    document_id: str,
    fingerprint: Optional[str] = None,
    cache_dir: str = RAPTOR_TREE_CACHE_DIR,
) -> Optional[RaptorTree]:
    doc_dir = Path(cache_dir) / _safe_path_component(document_id)
    if not doc_dir.exists():
        return None

    tree_dir: Optional[Path] = None
    if fingerprint:
        candidate = doc_dir / fingerprint
        if (candidate / "tree.json").exists():
            tree_dir = candidate
    else:
        candidates = [p for p in doc_dir.iterdir() if p.is_dir() and (p / "tree.json").exists()]
        if candidates:
            tree_dir = max(candidates, key=lambda p: p.stat().st_mtime)

    if tree_dir is None:
        return None

    with open(tree_dir / "tree.json", "r", encoding="utf-8") as f:
        payload = json.load(f)
    tree = deserialize_raptor_tree(payload)
    tree.stats = tree.stats or {}
    tree.stats["loaded_from_cache"] = True
    tree.stats["cache_dir"] = str(tree_dir)
    tree.stats["fingerprint"] = tree_dir.name
    return tree



def render_raptor_tree_mermaid(tree: RaptorTree) -> str:
    """Human-readable graph for quick inspection in Markdown / Mermaid viewers."""
    def esc(label: str) -> str:
        return label.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")

    node_map: Dict[str, str] = {}
    lines = ["graph TD"]
    root_id = "root"
    root_label = f"{tree.document_id}"
    lines.append(f'  {root_id}["{esc(root_label)}"]')

    leaf_ids = set()
    for level in sorted(tree.levels.keys()):
        for node in tree.levels[level]:
            node_map[node.node_id] = f"n_{_stable_hash(node.node_id)[:12]}"
            label = f"L{level} | {node.node_id} | {node.text[:80].replace(chr(10), ' ')}"
            lines.append(f'  {node_map[node.node_id]}["{esc(label)}"]')
            leaf_ids.update(node.child_chunk_ids)

    for leaf in sorted(leaf_ids):
        leaf_key = f"leaf_{_stable_hash(leaf)[:12]}"
        node_map[leaf] = leaf_key
        lines.append(f'  {leaf_key}["{esc(leaf)}"]')

    if tree.levels:
        top_level = min(tree.levels.keys())
        for node in tree.levels[top_level]:
            lines.append(f"  {root_id} --> {node_map[node.node_id]}")

    for level in sorted(tree.levels.keys()):
        for node in tree.levels[level]:
            for child_id in node.child_chunk_ids:
                child_key = node_map.get(child_id)
                if child_key:
                    lines.append(f"  {node_map[node.node_id]} --> {child_key}")

    return "\n".join(lines) + "\n"


def render_raptor_tree_dot(tree: RaptorTree) -> str:
    def dot_esc(label: str) -> str:
        return label.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")

    node_map: Dict[str, str] = {}
    lines = ["digraph RAPTOR {", '  rankdir="TB";', '  node [shape=box, style="rounded"];']
    lines.append(f'  root [label="{dot_esc(tree.document_id)}", shape=folder];')

    leaf_ids = set()
    for level in sorted(tree.levels.keys()):
        for node in tree.levels[level]:
            node_map[node.node_id] = f"n_{_stable_hash(node.node_id)[:12]}"
            label = f"L{level}\\n{node.node_id}\\n{node.text[:90].replace(chr(10), ' ')}"
            lines.append(f'  {node_map[node.node_id]} [label="{dot_esc(label)}"];')
            leaf_ids.update(node.child_chunk_ids)

    for leaf in sorted(leaf_ids):
        leaf_key = f"leaf_{_stable_hash(leaf)[:12]}"
        lines.append(f'  {leaf_key} [label="{dot_esc(leaf)}", shape=ellipse];')
        node_map[leaf] = leaf_key

    if tree.levels:
        top_level = min(tree.levels.keys())
        for node in tree.levels[top_level]:
            lines.append(f"  root -> {node_map[node.node_id]};")

    for level in sorted(tree.levels.keys()):
        for node in tree.levels[level]:
            for child_id in node.child_chunk_ids:
                child_key = node_map.get(child_id)
                if child_key:
                    lines.append(f"  {node_map[node.node_id]} -> {child_key};")

    lines.append("}")
    return "\n".join(lines) + "\n"
# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _select_gmm_components(embeddings: np.ndarray, max_k: int) -> int:
    """Picks the number of GMM components via BIC, the way the RAPTOR paper does.

    Tries k = 1..max_k, keeps the k with lowest BIC. Falls back to k=1 if
    fitting fails for every candidate (e.g. too few points / degenerate data).

    Uses spherical covariance rather than sklearn's default 'full'. With
    384-dim MiniLM embeddings and the chunk counts this system actually sees
    per document (tens to low hundreds), a full covariance matrix per
    component has vastly more free parameters than there are data points,
    so BIC's complexity penalty always dominates and collapses to k=1
    regardless of how well-separated the real clusters are. Spherical
    covariance (one variance scalar per component) keeps the parameter count
    sane for this regime and lets BIC actually discriminate between k values.

    The candidate range is additionally capped so that average cluster size
    never drops below RAPTOR_MIN_POINTS_PER_CLUSTER. Without this, BIC with
    spherical covariance keeps improving as k approaches n: individual
    components shrink their variance to nearly fit 1-2 points each, driving
    log-likelihood toward infinity faster than the parameter-count penalty
    can compensate. That's overfitting, not real structure, so we never let
    the search range reach it in the first place.
    """
    n = embeddings.shape[0]
    max_k_by_density = max(1, n // RAPTOR_MIN_POINTS_PER_CLUSTER)
    upper = max(1, min(max_k, max_k_by_density, n - 1))
    if upper <= 1:
        return 1

    best_k, best_bic = 1, float("inf")
    for k in range(1, upper + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type="spherical", random_state=42, n_init=1)
            gmm.fit(embeddings)
            bic = gmm.bic(embeddings)
            if bic < best_bic:
                best_bic = bic
                best_k = k
        except Exception as e:
            logger.debug(f"RAPTOR: GMM fit failed for k={k}: {e}")
            continue
    return best_k


def _soft_cluster(embeddings: np.ndarray, threshold: float = RAPTOR_SOFT_ASSIGN_THRESHOLD) -> List[List[int]]:
    """Soft-clusters embeddings with a BIC-selected GMM.

    Returns a list of clusters, each a list of row-indices into `embeddings`.
    A point can appear in more than one cluster if its posterior probability
    for that cluster exceeds `threshold` (this is the "soft" part — RAPTOR
    allows ambiguous chunks to inform more than one summary).
    """
    n = embeddings.shape[0]
    if n < RAPTOR_MIN_NODES_TO_CLUSTER:
        # Too few nodes to meaningfully cluster — treat as a single cluster.
        return [list(range(n))]

    k = _select_gmm_components(embeddings, max_k=min(RAPTOR_MAX_COMPONENTS_CAP, n // 2))
    if k <= 1:
        return [list(range(n))]

    gmm = GaussianMixture(n_components=k, covariance_type="spherical", random_state=42, n_init=1)
    gmm.fit(embeddings)
    probs = gmm.predict_proba(embeddings)  # (n, k)

    clusters: List[List[int]] = [[] for _ in range(k)]
    for i in range(n):
        joined_any = False
        for c in range(k):
            if probs[i, c] >= threshold:
                clusters[c].append(i)
                joined_any = True
        if not joined_any:
            # Numerical edge case: assign to the single most likely cluster
            # so no point is silently dropped from the tree.
            clusters[int(np.argmax(probs[i]))].append(i)

    # Drop empty clusters (can happen if GMM concentrates mass elsewhere).
    return [c for c in clusters if c]


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

class RaptorSummarizer:
    """Summarizes a cluster of chunk/node texts into one abstractive node via a
    local Ollama model.

    The ``model`` parameter (default: ``RAPTOR_OLLAMA_MODEL``) should match
    whatever model the rest of the pipeline already has loaded in Ollama so
    the GPU doesn't need to juggle two sets of weights simultaneously.  Pass
    the same value as ``OLLAMA_MODEL`` / ``--ollama-model`` to guarantee that.
    """

    def __init__(self, model: str = RAPTOR_OLLAMA_MODEL, host: str = RAPTOR_OLLAMA_HOST):
        self.model = model
        self.host = host
        self._client = None
        try:
            from ollama import Client
            # NOTE: ollama.Client (this version) has no per-call timeout
            # override, so one value covers both the reachability check below
            # and every later chat() call. The library's own default is
            # timeout=None (httpx waits forever, no error, no log line) —
            # that's what caused tree builds to hang silently for minutes.
            self._client = Client(host=self.host, timeout=RAPTOR_OLLAMA_TIMEOUT)
            # Cheap reachability check so we fail soft (log + fallback) instead
            # of throwing mid-tree-build if the local Ollama server is down or
            # the model hasn't been pulled yet.
            self._client.list()
        except ImportError:
            logger.warning("ollama package not installed — RAPTOR summarization disabled. Install with: pip install ollama")
            self._client = None
        except Exception as e:
            logger.warning(f"Could not reach Ollama at {self.host} — RAPTOR summarization disabled: {e}")
            self._client = None

    def summarize(self, texts: List[str], level: int) -> str:
        """Produces one abstractive summary covering all of `texts`.

        Falls back to simple concatenation + truncation if Ollama is
        unavailable or the call fails, so tree-building degrades gracefully
        instead of throwing.
        """
        combined = "\n\n---\n\n".join(t.strip() for t in texts if t and t.strip())
        if not combined:
            return ""

        if self._client is None:
            return combined[:RAPTOR_SUMMARY_MAX_CHARS]

        # Cap the number of input passages to prevent oversized context windows.
        # Clusters larger than RAPTOR_MAX_CLUSTER_INPUT_CHUNKS would push the
        # prompt past the num_ctx budget and spike VRAM; we take the first N
        # rather than truncating per-text so each passage stays coherent.
        if len(texts) > RAPTOR_MAX_CLUSTER_INPUT_CHUNKS:
            logger.info(
                f"RAPTOR: cluster has {len(texts)} texts, capping to "
                f"{RAPTOR_MAX_CLUSTER_INPUT_CHUNKS} for summarization (VRAM safety)."
            )
            texts = texts[:RAPTOR_MAX_CLUSTER_INPUT_CHUNKS]
            combined = "\n\n---\n\n".join(t.strip() for t in texts if t and t.strip())

        prompt = (
            "You are summarizing a cluster of related passages from the same document "
            "so the summary can stand in for all of them in a retrieval index.\n\n"
            "Write a dense, factual summary that:\n"
            "- Preserves specific names, numbers, dates, and technical terms from the "
            "passages (do not generalize them away)\n"
            "- Captures what the passages have in common AND notes any important "
            "differences between them\n"
            "- Reads as a standalone paragraph, not a list of bullet points\n"
            f"- Stays under {RAPTOR_SUMMARY_MAX_CHARS // 5} words\n\n"
            # 4 chunks × ~2 400 chars = ~9 600 chars, but we need to fit
            # everything inside num_ctx=2048.  ~3 chars/token → 2048 tokens
            # of passages budget after removing ~200 tokens of boilerplate
            # = (2048 - 200) × 3 ≈ 5 500 chars.  Use 5 000 to be safe.
            f"Passages:\n{combined[:5000]}"
        )
        try:
            t0 = time.perf_counter()
            logger.info(f"RAPTOR: summarizing cluster of {len(texts)} texts at level {level} via Ollama ({self.model})...")
            resp = self._chat_with_oom_retry(prompt)
            elapsed = time.perf_counter() - t0
            logger.info(f"RAPTOR: cluster summarized in {elapsed:.1f}s at level {level}")
            # resp is a typed ollama.ChatResponse (pydantic model); use attribute
            # access rather than .get(), which only works by coincidence here
            # because ChatResponse/Message happen to also support dict-style
            # access via SubscriptableBaseModel.
            summary = (resp.message.content or "").strip()
            if not summary:
                return combined[:RAPTOR_SUMMARY_MAX_CHARS]
            return summary[:RAPTOR_SUMMARY_MAX_CHARS]
        except Exception as e:
            logger.error(f"RAPTOR summarization failed at level {level} (after {time.perf_counter() - t0:.1f}s), falling back to truncated concat: {e}")
            return combined[:RAPTOR_SUMMARY_MAX_CHARS]

    def _chat_with_oom_retry(self, prompt: str):
        """Single retry, after a cooldown, specifically for the
        'llama-server process has terminated ... cudaMalloc failed: out of
        memory' failure. That error means a second model instance briefly
        existed on the GPU (e.g. a previous Ollama call with a different
        request config still tearing down) -- not that this prompt is too
        big. Retrying immediately after a short sleep usually succeeds once
        the dying process has actually freed its VRAM. Without this, one
        transient OOM mid-tree-build silently degrades every remaining
        cluster in the tree to truncated-concat instead of a real summary.
        """
        try:
            return self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You write dense, factual, standalone summaries for a search index. Output only the summary text, no preamble."},
                    {"role": "user", "content": prompt},
                ],
                # num_ctx: 2048 is the cap for this 4 GB GPU — see
                #   RAPTOR_OLLAMA_NUM_CTX comment above for the budget math.
                # num_predict: cap output tokens so Ollama doesn't try to fill
                #   the remaining context with padding.
                options={"temperature": 0.1, "num_predict": 600, "num_ctx": RAPTOR_OLLAMA_NUM_CTX},
            )
        except Exception as e:
            if "cudaMalloc" not in str(e) and "out of memory" not in str(e):
                raise
            logger.warning(f"RAPTOR: transient CUDA OOM, waiting 5s for VRAM to free and retrying once: {e}")
            time.sleep(5.0)
            return self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You write dense, factual, standalone summaries for a search index. Output only the summary text, no preamble."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.1, "num_predict": 600, "num_ctx": RAPTOR_OLLAMA_NUM_CTX},
            )




# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------

class RaptorTreeBuilder:
    """Builds a RAPTOR tree on top of an already-chunked, already-embedded document.

    This runs as a post-ingestion step: call `build` after the document's
    level-0 chunks are already in VectorStore (so we can reuse their dense
    embeddings instead of re-embedding from scratch).
    """

    def __init__(self, embedding_model, summarizer: Optional[RaptorSummarizer] = None):
        """
        embedding_model: anything exposing .encode(List[str]) -> np.ndarray,
                          intended to be VectorStore.embedding_model so level-1+
                          summary embeddings live in the exact same vector space
                          as the level-0 chunks already in the index.
        """
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for RAPTOR clustering. "
                "Install with: pip install scikit-learn"
            )
        self.embedding_model = embedding_model
        self.summarizer = summarizer or RaptorSummarizer()

    def build(
        self,
        document_id: str,
        chunk_ids: List[str],
        chunk_texts: List[str],
        chunk_embeddings: Optional[np.ndarray] = None,
        source_file: str = "",
        max_levels: int = RAPTOR_MAX_LEVELS,
    ) -> RaptorTree:
        """Build the tree for one document's level-0 chunks.

        Returns a RaptorTree whose `.levels[1], .levels[2], ...` contain the
        new summary nodes (level 0 is intentionally NOT duplicated into the
        tree — callers already have the raw chunks from chunking.py).
        """
        t0 = time.perf_counter()
        tree = RaptorTree(document_id=document_id)

        if len(chunk_ids) < RAPTOR_MIN_NODES_TO_CLUSTER:
            logger.info(
                f"RAPTOR: document {document_id} has only {len(chunk_ids)} chunks "
                f"(< {RAPTOR_MIN_NODES_TO_CLUSTER}); skipping tree build."
            )
            tree.stats = {"skipped": True, "reason": "too_few_chunks", "chunk_count": len(chunk_ids)}
            return tree

        # Level 0 working set: ids, texts, embeddings (computed if not supplied).
        cur_ids = list(chunk_ids)
        cur_texts = list(chunk_texts)
        cur_embs = (
            chunk_embeddings if chunk_embeddings is not None
            else np.asarray(self.embedding_model.encode(cur_texts))
        )

        level_stats = []
        level = 1
        while level <= max_levels:
            if len(cur_ids) < RAPTOR_MIN_NODES_TO_CLUSTER:
                logger.info(f"RAPTOR: level {level} has only {len(cur_ids)} nodes; stopping recursion.")
                break

            clusters = _soft_cluster(cur_embs)

            if len(clusters) <= 1:
                logger.info(f"RAPTOR: level {level} collapsed to a single cluster; this is the top of the tree.")
                # Still summarize it once, as the document's single top-level node,
                # then stop — there's nothing left to recurse on.
                node = self._summarize_cluster(document_id, level, cur_ids, cur_texts, source_file)
                tree.levels[level] = [node]
                level_stats.append({"level": level, "clusters": 1, "input_nodes": len(cur_ids)})
                break

            new_nodes: List[RaptorNode] = []
            n_clusters = len(clusters)
            logger.info(f"RAPTOR: level {level} split into {n_clusters} clusters from {len(cur_ids)} input nodes; summarizing each...")
            for ci, cluster_indices in enumerate(clusters, start=1):
                cluster_ids = [cur_ids[i] for i in cluster_indices]
                cluster_texts = [cur_texts[i] for i in cluster_indices]
                logger.info(f"RAPTOR: level {level}, cluster {ci}/{n_clusters} ({len(cluster_indices)} members)...")
                node = self._summarize_cluster(document_id, level, cluster_ids, cluster_texts, source_file)
                new_nodes.append(node)

            tree.levels[level] = new_nodes
            level_stats.append({"level": level, "clusters": len(clusters), "input_nodes": len(cur_ids)})

            # Promote this level's summaries to be next level's input.
            cur_ids = [n.node_id for n in new_nodes]
            cur_texts = [n.text for n in new_nodes]
            cur_embs = np.asarray(self.embedding_model.encode(cur_texts))

            if len(new_nodes) <= 1:
                break

            level += 1

        tree.stats = {
            "skipped": False,
            "levels_built": len(tree.levels),
            "total_summary_nodes": sum(len(v) for v in tree.levels.values()),
            "level_detail": level_stats,
            "build_time_ms": (time.perf_counter() - t0) * 1000,
        }
        logger.info(
            f"RAPTOR: built {tree.stats['levels_built']} levels "
            f"({tree.stats['total_summary_nodes']} summary nodes) for document {document_id} "
            f"in {tree.stats['build_time_ms']:.0f}ms"
        )
        return tree

    def _summarize_cluster(
        self, document_id: str, level: int, member_ids: List[str], member_texts: List[str], source_file: str
    ) -> RaptorNode:
        summary_text = self.summarizer.summarize(member_texts, level=level)
        cluster_signature = _stable_hash(document_id, level, sorted(member_ids))[:10]
        node_id = f"{document_id}_raptor_l{level}_{cluster_signature}"
        return RaptorNode(
            node_id=node_id,
            level=level,
            text=summary_text,
            document_id=document_id,
            child_chunk_ids=member_ids,
            source_file=source_file,
        )


# ---------------------------------------------------------------------------
# Convenience entry point used by ingestion / eval harness
# ---------------------------------------------------------------------------

def build_and_store_raptor_tree(
    document_id: str,
    chunks: List[Dict[str, Any]],
    vector_store,
    bm25_store=None,
    source_file: str = "",
    raptor_model: str = RAPTOR_OLLAMA_MODEL,
    cache_dir: str = RAPTOR_TREE_CACHE_DIR,
    reuse_cached: bool = True,
    export_visualization: bool = True,
) -> RaptorTree:
    """End-to-end helper: build a RAPTOR tree for `chunks` (level-0 output of
    DocumentChunker that's already been added to vector_store) and write the
    resulting summary nodes back into vector_store (and bm25_store if given)
    using their normal add_chunks() contract.

    If an identical chunk set was seen before, the cached tree is reused from
    disk so the system can resume from the prior RAPTOR tree instead of
    rebuilding it.

    ``raptor_model`` selects the Ollama model used for cluster summarization.
    Pass the same model name as the rest of the pipeline (e.g. ``llama3.2``)
    to ensure Ollama only needs to hold a single model in VRAM at a time.

    Safe to call with zero or very few chunks — it will just skip tree
    building and return an empty tree rather than raising.
    """
    chunk_ids = [c["chunk_id"] for c in chunks]
    chunk_texts = [c.get("text", "") for c in chunks]
    fingerprint = compute_raptor_fingerprint(chunk_ids, chunk_texts)

    tree = None
    if reuse_cached:
        tree = load_cached_raptor_tree(document_id, fingerprint=fingerprint, cache_dir=cache_dir)
        if tree is not None:
            logger.info(
                f"RAPTOR: loaded cached tree for document {document_id} "
                f"({fingerprint}, {sum(len(v) for v in tree.levels.values())} summary nodes)."
            )

    if tree is None:
        summarizer = RaptorSummarizer(model=raptor_model)
        builder = RaptorTreeBuilder(embedding_model=vector_store.embedding_model, summarizer=summarizer)
        tree = builder.build(
            document_id=document_id,
            chunk_ids=chunk_ids,
            chunk_texts=chunk_texts,
            source_file=source_file,
        )
        tree.stats["fingerprint"] = fingerprint
        tree.stats["source_file"] = source_file
        tree.stats["cache_version"] = RAPTOR_TREE_CACHE_VERSION
        if export_visualization and tree and not tree.is_empty():
            artifacts = save_raptor_tree_artifacts(
                tree,
                fingerprint=fingerprint,
                source_file=source_file,
                cache_dir=cache_dir,
            )
            tree.stats["artifacts"] = artifacts
            logger.info(
                f"RAPTOR: saved tree artifacts for {document_id} -> {artifacts['tree_json']}"
            )

    if tree.all_nodes():
        summary_chunk_dicts = [n.to_chunk_dict() for n in tree.all_nodes()]
        vector_store.add_chunks(summary_chunk_dicts)
        if bm25_store is not None:
            bm25_store.add_chunks(summary_chunk_dicts)
        logger.info(
            f"RAPTOR: indexed {len(summary_chunk_dicts)} summary nodes for "
            f"document {document_id} into vector_store"
            + (" and bm25_store" if bm25_store is not None else "")
        )

    return tree
