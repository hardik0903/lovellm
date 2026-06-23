"""
metrics.py
==========

Pure, dependency-free metric functions used by rag_eval.py. Kept separate so
they can be unit-tested without chromadb/sentence-transformers/groq installed.
"""

from typing import List


def hit_rate(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    return 1.0 if any(r in relevant_ids for r in retrieved_ids) else 0.0


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def context_precision(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for r in retrieved_ids if r in relevant_ids)
    return hits / len(retrieved_ids)


def context_precision_at_r(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Precision computed over only the top len(relevant_ids) retrieved items
    (precision@R, where R = number of gold-relevant chunks for this query).

    This avoids the structural ceiling that plain precision@k hits when k is
    fixed but the gold set has fewer relevant chunks than k -- e.g. with
    k=5 and 1 relevant chunk, precision@5 is capped at 0.2 even for a
    perfect retriever. precision@R = 1.0 in that case if the single relevant
    chunk is ranked first.
    """
    r = len(relevant_ids)
    if r == 0:
        return 1.0
    top_r = retrieved_ids[:r]
    if not top_r:
        return 0.0
    hits = sum(1 for x in top_r if x in relevant_ids)
    return hits / len(top_r)


def context_recall(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    if not relevant_ids:
        return 1.0
    hits = sum(1 for rid in relevant_ids if rid in retrieved_ids)
    return hits / len(relevant_ids)


def chunk_id_for(doc: dict) -> str:
    """Resolve a comparable chunk id. Naive retriever results come straight
    from chroma metadata; hybrid results carry chunk_id at the top level."""
    if "chunk_id" in doc:
        return doc["chunk_id"]
    return doc.get("metadata", {}).get("chunk_id", "")