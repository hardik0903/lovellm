from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from logger import logger

class Reranker:
    def __init__(self):
        # Model is loaded lazily on first rerank call (avoid blocking startup).
        # Pinned to CPU: the cross-encoder (~85 MB) is fast enough on CPU for
        # the small candidate sets it scores (typically 10-20 pairs), and
        # keeping it off-GPU preserves VRAM for Ollama's KV-cache.
        # On an RTX 3050 4 GB the total fixed VRAM footprint is already
        # llama3.2 ≈ 2000 MB + OS/driver ≈ 200 MB, leaving ~1896 MB for the
        # KV-cache. Every MB saved here is margin against OOM.
        self.model = None
        self._device = "cpu"

    def _load_model(self):
        if self.model is None:
            logger.info("Loading cross-encoder model for reranking (CPU)...")
            self.model = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                device=self._device,
            )
            logger.info("Cross-encoder model loaded on CPU.")

    def should_rerank(self, query: str, candidates: List[Dict[str, Any]]) -> bool:
        """
        Conditional Trigger: Runs ONLY when the top candidates are close in score,
        the query is comparative, or lexical and semantic retrieval disagree.
        For simplicity, we check if we have multiple candidates and if the query is comparative.
        In a full implementation, you'd analyze the score distribution and BM25 vs Dense overlap.
        """
        if len(candidates) <= 1:
            return False
            
        query_lower = query.lower()
        if "compare" in query_lower or "difference" in query_lower:
            return True
            
        # Check if top 2 scores are very close (assuming scores are normalized 0-1)
        if len(candidates) >= 2:
            score_diff = abs(candidates[0].get("score", 0) - candidates[1].get("score", 0))
            if score_diff < 0.1:  # Threshold for "close" scores
                return True
                
        # We can also always rerank if it's a hybrid query, but we'll stick to the strict rules.
        return False

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []
            
        if not self.should_rerank(query, candidates):
            logger.info("Reranking condition not met. Skipping reranker.")
            return candidates[:top_k]

        logger.info(f"Reranking {len(candidates)} candidates for query: {query}")
        
        # E-6 FIX: use context_text (the parent-expanded surface the LLM
        # actually receives) instead of the raw child text field.  The
        # cross-encoder was previously scoring against text but the generator
        # sees context_text — so the ranking signal was optimised for the
        # wrong content, quietly degrading reranker effectiveness for all
        # parent-resolved chunks.
        pairs = [[query, doc.get("context_text") or doc.get("text", "")] for doc in candidates]

        
        # Predict scores
        self._load_model()
        scores = self.model.predict(pairs)
        
        # Add new scores and sort
        for i, doc in enumerate(candidates):
            doc["rerank_score"] = float(scores[i])
            
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]