from typing import List, Dict, Any
from vector_store import VectorStore
from bm25_store import BM25Store
from router import QueryRouter
from reranker import Reranker
from logger import logger

class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_store: BM25Store):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.router = QueryRouter()
        self.reranker = Reranker()

    def _reciprocal_rank_fusion(self, results_lists: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        """Fuses multiple ranked lists using RRF."""
        fused_scores = {}
        doc_map = {}
        
        for results in results_lists:
            for rank, doc in enumerate(results):
                doc_id = doc["chunk_id"]
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0.0
                    doc_map[doc_id] = doc
                fused_scores[doc_id] += 1.0 / (k + rank + 1)
                
        # Sort by fused score
        reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for doc_id, score in reranked:
            doc = doc_map[doc_id].copy()
            doc["score"] = score
            final_results.append(doc)
            
        return final_results

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        route = self.router.route(query)
        logger.info(f"Retrieval route selected: {route}")
        
        candidates = []
        
        if route == "bm25":
            candidates = self.bm25_store.search(query, top_k=top_k * 2)
        elif route == "dense":
            candidates = self.vector_store.search(query, top_k=top_k * 2)
        else: # hybrid
            bm25_results = self.bm25_store.search(query, top_k=top_k * 2)
            dense_results = self.vector_store.search(query, top_k=top_k * 2)
            candidates = self._reciprocal_rank_fusion([bm25_results, dense_results])
            
        # Optional Reranking
        final_results = self.reranker.rerank(query, candidates, top_k=top_k)
        
        # Parent resolution: Instead of returning the child chunk's text,
        # we provide the parent_text to the generator to give broader context,
        # but keep the child metadata for precise citations.
        for doc in final_results:
            if "parent_text" in doc.get("metadata", {}):
                doc["context_text"] = doc["metadata"]["parent_text"]
            else:
                doc["context_text"] = doc["text"]
                
        return final_results
