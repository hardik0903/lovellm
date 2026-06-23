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
        self.tokenized_corpus = []
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
                self.tokenized_corpus = [self._tokenize(doc["text"]) for doc in self.documents]
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

    def delete_by_document_id(self, document_id: str):
        """Removes all chunks belonging to a document_id and rebuilds the BM25 index.
        Used by the reindexing/repair script to clear stale entries before re-ingesting
        with the current chunking schema."""
        before = len(self.documents)
        self.documents = [doc for doc in self.documents if doc.get("document_id") != document_id]
        removed = before - len(self.documents)
        if removed == 0:
            logger.info(f"No existing chunks found for document_id={document_id} in BM25 Store.")
            return

        logger.info(f"Removed {removed} chunks for document_id={document_id} from BM25 Store.")
        self.tokenized_corpus = [self._tokenize(doc["text"]) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
        self._save()

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return
            
        existing_ids = {doc["chunk_id"] for doc in self.documents}
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
        
        if not new_chunks:
            logger.info("All chunks already exist in BM25 Store. Skipping.")
            return
            
        logger.info(f"Adding {len(new_chunks)} new chunks to BM25 Store.")
        self.documents.extend(new_chunks)
        
        new_tokenized = [self._tokenize(doc["text"]) for doc in new_chunks]
        self.tokenized_corpus.extend(new_tokenized)
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._save()

    def search(self, query: str, top_k: int = 10, node_types: List[str] = None) -> List[Dict[str, Any]]:
        """BM25 search. `node_types`, if given, restricts results to documents whose
        `node_type` field is in that list (e.g. ["summary"] for RAPTOR nodes only).
        None preserves the original unfiltered behavior."""
        logger.info(f"BM25 search for query: {query}")
        if not self.bm25 or not self.documents:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        if node_types:
            allowed = set(node_types)
            candidate_indices = [i for i, doc in enumerate(self.documents) if doc.get("node_type") in allowed]
        else:
            candidate_indices = list(range(len(self.documents)))

        top_indices = sorted(candidate_indices, key=lambda i: scores[i], reverse=True)[:top_k]

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