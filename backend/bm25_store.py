import os
import pickle
import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from logger import logger

class BM25Store:
    def __init__(self, persist_dir: str = "./data/bm25"):
        self.persist_dir = persist_dir
        self.index_path = os.path.join(persist_dir, "bm25_index.pkl")
        self.docs_path = os.path.join(persist_dir, "bm25_docs.pkl")
        os.makedirs(persist_dir, exist_ok=True)
        
        self.bm25 = None
        self.documents: List[Dict[str, Any]] = []
        self._load()

    def _tokenize(self, text: str) -> List[str]:
        # Simple lowercase and alphanumeric tokenization
        return re.findall(r'\b\w+\b', text.lower())

    def _load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
            try:
                with open(self.index_path, 'rb') as f:
                    self.bm25 = pickle.load(f)
                with open(self.docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"Loaded BM25 index with {len(self.documents)} documents.")
            except Exception as e:
                logger.error(f"Failed to load BM25 index: {e}")

    def _save(self):
        try:
            with open(self.index_path, 'wb') as f:
                pickle.dump(self.bm25, f)
            with open(self.docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            logger.info("Saved BM25 index to disk.")
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return
            
        logger.info(f"Adding {len(chunks)} chunks to BM25 Store.")
        self.documents.extend(chunks)
        
        tokenized_corpus = [self._tokenize(doc["text"]) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self._save()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        logger.info(f"BM25 search for query: {query}")
        if not self.bm25 or not self.documents:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        output = []
        for idx in top_indices:
            score = scores[idx]
            if score > 0:
                doc = self.documents[idx].copy()
                output.append({
                    "chunk_id": doc["chunk_id"],
                    "text": doc["text"],
                    "score": score,
                    "metadata": {k: v for k, v in doc.items() if k != "text"}
                })
        
        return output
