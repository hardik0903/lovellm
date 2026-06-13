import os
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
from logger import logger
import diskcache

class VectorStore:
    def __init__(self, persist_dir: str = "./data/chroma"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="documents")
        logger.info("Loading sentence transformer model for dense embeddings...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Dense embedding model loaded.")
        self.parent_cache = diskcache.Cache(os.path.join(persist_dir, "parents_cache"))

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return
            
        logger.info(f"Adding {len(chunks)} chunks to Vector Store.")
        ids = [chunk["chunk_id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        
        # Serialize metadata to ensure ChromaDB compatibility (no dicts inside dicts)
        metadatas = []
        for chunk in chunks:
            if "parent_id" in chunk and "parent_text" in chunk:
                self.parent_cache[chunk["parent_id"]] = chunk["parent_text"]
            meta = {k: str(v) if not isinstance(v, (int, float, str, bool)) else v for k, v in chunk.items() if k not in ["text", "parent_text"]}
            metadatas.append(meta)

        embeddings = self.embedding_model.encode(texts).tolist()
        
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        logger.info(f"Dense search for query: {query}")
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        output = []
        if not results["ids"]:
            return output
            
        for i in range(len(results["ids"][0])):
            # Distance in Chroma is typically cosine distance or L2. Lower is better.
            # We invert it for score fusion so higher is better.
            raw_distance = results["distances"][0][i]
            score = 1.0 / (1.0 + raw_distance)
            
            meta_with_parent = dict(results["metadatas"][0][i])
            if "parent_id" in meta_with_parent:
                parent_text = self.parent_cache.get(meta_with_parent["parent_id"])
                if parent_text:
                    meta_with_parent["parent_text"] = parent_text

            output.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "score": score,
                "metadata": meta_with_parent
            })
            
        return output
