import os
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
from logger import logger
import diskcache
import json

# ---------------------------------------------------------------------------
# Monkey-patch: ChromaDB 0.5.0's _decode_seq_id crashes with
#   TypeError: object of type 'int' has no len()
# when the persisted SQLite DB stores seq_ids as plain ints instead of
# 8-byte blobs.  This patch makes the decoder accept both representations.
# ---------------------------------------------------------------------------
try:
    from chromadb.segment.impl.metadata import sqlite as _sqlite_mod
    _original_decode = _sqlite_mod._decode_seq_id

    def _patched_decode_seq_id(seq_id_bytes):  # noqa: N802
        if isinstance(seq_id_bytes, int):
            return seq_id_bytes
        return _original_decode(seq_id_bytes)

    _sqlite_mod._decode_seq_id = _patched_decode_seq_id
    logger.debug("Applied ChromaDB _decode_seq_id monkey-patch.")
except Exception as _patch_err:
    logger.warning(f"Could not apply ChromaDB seq_id patch: {_patch_err}")

class VectorStore:
    def __init__(self, persist_dir: str = "./data/chroma"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="documents")
        logger.info("Loading sentence transformer model for dense embeddings...")
        # Force CPU: the embedding model (~90 MB) is fast enough on CPU for
        # inference-time encode() calls and keeping it off-GPU leaves the full
        # 4 GB VRAM budget for Ollama (llama3.2 ≈ 2 GB + KV-cache ≈ 450 MB).
        # If you later move to a GPU with >6 GB VRAM, remove device="cpu" here.
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        logger.info("Dense embedding model loaded.")
        self.parent_cache_path = os.path.join(persist_dir, "parent_cache.json")
        self.parent_cache = {}
        if os.path.exists(self.parent_cache_path):
            try:
                with open(self.parent_cache_path, "r", encoding="utf-8") as f:
                    self.parent_cache = json.load(f)
                logger.info(f"Loaded parent cache with {len(self.parent_cache)} entries.")
            except Exception as e:
                logger.error(f"Failed to load parent cache: {e}")
        # If the cache has fewer entries than expected (e.g. documents were ingested
        # before parent_cache.json was introduced, or the file was wiped while chroma
        # retained the data), repair it from chroma's stored chunk metadata.
        collection_count = self.collection.count()
        if collection_count > 0 and len(self.parent_cache) < collection_count // 3:
            self._warm_parent_cache_from_chroma()
    def delete_by_document_id(self, document_id: str):
        """Removes all chunks (and orphaned parent_cache entries) belonging to a document_id.
        Used by the reindexing/repair script to clear stale entries before re-ingesting
        with the current chunking schema."""
        existing = self.collection.get(where={"document_id": document_id}, include=["metadatas"])
        ids_to_delete = existing.get("ids", [])
        if not ids_to_delete:
            logger.info(f"No existing chunks found for document_id={document_id}.")
            return

        parent_ids = set()
        for meta in existing.get("metadatas", []):
            if meta and "parent_id" in meta:
                parent_ids.add(meta["parent_id"])

        self.collection.delete(ids=ids_to_delete)
        logger.info(f"Deleted {len(ids_to_delete)} chunks for document_id={document_id} from Vector Store.")

        changed = False
        for pid in parent_ids:
            if pid in self.parent_cache:
                del self.parent_cache[pid]
                changed = True
        if changed:
            try:
                with open(self.parent_cache_path, "w", encoding="utf-8") as f:
                    json.dump(self.parent_cache, f)
                logger.info("Updated parent cache after deletion.")
            except Exception as e:
                logger.error(f"Failed to save parent cache after deletion: {e}")

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return
            
        logger.info(f"Adding {len(chunks)} chunks to Vector Store.")
        ids = [chunk["chunk_id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        
        # Serialize metadata to ensure ChromaDB compatibility (no dicts inside dicts)
        metadatas = []
        for chunk in chunks:
            # Parent-level chunks carry "parent_id": None (they have no parent
            # of their own -- see chunking.py). Guard with a truthiness check
            # here, not just key presence, so we never cache a bogus entry
            # under the key None/"None".
            if chunk.get("parent_id") and "parent_text" in chunk:
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

        try:
            with open(self.parent_cache_path, "w", encoding="utf-8") as f:
                json.dump(self.parent_cache, f)
            logger.info("Saved parent cache to disk.")
        except Exception as e:
            logger.error(f"Failed to save parent cache: {e}")
    def _warm_parent_cache_from_chroma(self):
        """Back-fills parent_cache by reconstructing it directly from Chroma.

        This covers the case where documents were ingested before
        parent_cache.json existed, or where the JSON file was wiped/went
        stale while chroma retained the data (including the legacy-schema
        case reindex_fixtures.py was written for, where an older parent_id
        format is no longer present in the current cache).

        Note: this does NOT read a "parent_text" metadata field -- add_chunks()
        deliberately excludes parent_text from chunk metadata (storing the
        full parent text on every child chunk's metadata would multiply
        storage several-fold). Instead we use the fact that, by construction
        (see chunking.py), a parent-level chunk's own chunk_id is exactly the
        parent_id every one of its child chunks points to, and that parent
        chunk's stored `documents` text *is* the parent text. So we build a
        chunk_id -> text map from everything in the collection in one pass,
        then resolve each referenced parent_id against that map.
        """
        try:
            existing = self.collection.get(include=["metadatas", "documents"])
            ids = existing.get("ids", [])
            documents = existing.get("documents", [])
            metadatas = existing.get("metadatas", [])

            text_by_id = {cid: text for cid, text in zip(ids, documents) if text}

            added = 0
            for meta in metadatas:
                if not meta:
                    continue
                pid = meta.get("parent_id")
                if not pid or pid == "None" or pid in self.parent_cache:
                    continue
                parent_text = text_by_id.get(pid)
                if parent_text:
                    self.parent_cache[pid] = parent_text
                    added += 1

            if added:
                with open(self.parent_cache_path, "w", encoding="utf-8") as f:
                    json.dump(self.parent_cache, f)
                logger.info(f"Warmed parent cache with {added} entries from chroma.")
        except Exception as e:
            logger.warning(f"Could not warm parent cache from chroma: {e}")

    def _fetch_parent_text_from_chroma(self, parent_id: str) -> str | None:
        """Looks up a single parent chunk's own text directly from Chroma by id.

        Used as a per-query self-heal path for a parent_id that's missing from
        parent_cache (e.g. cache went stale between startup and this query).
        Relies on the same chunk_id == parent_id identity used by
        _warm_parent_cache_from_chroma; see that method's docstring.
        """
        try:
            result = self.collection.get(ids=[parent_id], include=["documents"])
            docs = result.get("documents") or []
            if docs and docs[0]:
                return docs[0]
        except Exception as e:
            logger.warning(f"Could not fetch parent text for {parent_id} from chroma: {e}")
        return None

    def search(self, query: str, top_k: int = 10, node_types: List[str] = None) -> List[Dict[str, Any]]:
        """Dense search. `node_types`, if given, restricts results to chunks whose
        `node_type` metadata field is in that list (e.g. ["summary"] to search only
        RAPTOR summary nodes, or ["raw"] / None for the original chunk-only behavior
        when no RAPTOR tree exists for a document). Leaving it as None preserves the
        exact pre-RAPTOR behavior of searching every indexed chunk."""
        logger.info(f"Dense search for query: {query}")
        query_embedding = self.embedding_model.encode([query]).tolist()

        where_clause = None
        if node_types:
            where_clause = {"node_type": {"$in": node_types}} if len(node_types) > 1 else {"node_type": node_types[0]}

        query_kwargs = dict(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if where_clause:
            query_kwargs["where"] = where_clause

        results = self.collection.query(**query_kwargs)
        
        output = []
        if not results["ids"]:
            return output
            
        for i in range(len(results["ids"][0])):
            raw_distance = results["distances"][0][i]
            score = 1.0 / (1.0 + raw_distance)
            
            meta_with_parent = dict(results["metadatas"][0][i])
            pid = meta_with_parent.get("parent_id")
            # Parent-level chunks legitimately have no parent_id (serialized
            # as the string "None" by add_chunks' metadata sanitizer, since
            # raw None isn't a valid Chroma metadata value). Only chase a
            # parent lookup for chunks that actually reference a parent.
            if pid not in (None, "None", ""):
                # 1. Try the JSON cache (fast path, already in memory)
                parent_text = self.parent_cache.get(pid)
                # 2. Self-heal: look the parent chunk's own text up directly
                #    in chroma (covers entries missing from parent_cache.json,
                #    e.g. legacy/stale-schema chunks -- see
                #    _fetch_parent_text_from_chroma's docstring).
                if not parent_text:
                    parent_text = self._fetch_parent_text_from_chroma(pid)
                    if parent_text:
                        # Heal the cache in memory so sibling chunks hit fast path
                        self.parent_cache[pid] = parent_text
                if parent_text:
                    meta_with_parent["parent_text"] = parent_text
                else:
                    logger.warning(f"Parent text missing in cache for {pid}")

            output.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "score": score,
                "metadata": meta_with_parent
            })
            
        return output