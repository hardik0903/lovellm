"""
reindex_fixtures.py
====================

One-time repair script for index/schema drift.

Problem it fixes:
- `dummy.txt` and `semantic_fixture.txt` were ingested with an older version of
  chunking.py whose `parent_id` did not include the `_page_{N}_` segment
  (e.g. "dummy.txt_parent_0"), while the current chunking.py and
  data/chroma/parent_cache.json use "dummy.txt_page_1_parent_0". This causes
  every retrieval of these chunks to log "Parent text missing in cache" and
  fall back to the short child-chunk text instead of the richer parent text.
- The PDF "1. Language Fundamentals (1).pdf" was added to the vector store
  but never to the BM25 store, so bm25/hybrid-routed queries can never
  retrieve it -- the two indexes are out of sync.

What this script does:
1. Deletes all chunks for dummy.txt and semantic_fixture.txt from both the
   vector store and BM25 store (clearing stale parent_id metadata and
   orphaned parent_cache entries).
2. Re-ingests both files with the current DocumentChunker, writing correct
   page-qualified parent_ids to both stores and parent_cache.json.
3. Re-ingests the PDF into BM25 (idempotent -- add_chunks skips chunk_ids
   that already exist), so BM25 and the vector store cover the same corpus.

Usage:
    python -m eval.reindex_fixtures
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from logger import logger
from vector_store import VectorStore
from bm25_store import BM25Store
from ingestion import DocumentIngestor


FIXTURES = {
    "dummy.txt": "tests/fixtures/dummy.txt",
    "semantic_fixture.txt": "tests/fixtures/semantic_fixture.txt",
}

PDF_DOC_ID = "1._language_fundamentals_(1).pdf"
PDF_PATH = "data/uploads/1. Language Fundamentals (1).pdf"


def main():
    vector_store = VectorStore(persist_dir="./data/chroma")
    bm25_store = BM25Store(persist_dir="./data/bm25")
    ingestor = DocumentIngestor()

    # --- Step 1 & 2: clear and re-ingest the stale fixtures ---
    for document_id, rel_path in FIXTURES.items():
        if not os.path.exists(rel_path):
            logger.warning(f"Fixture not found, skipping: {rel_path}")
            continue

        logger.info(f"Re-indexing {document_id} from {rel_path}")
        vector_store.delete_by_document_id(document_id)
        bm25_store.delete_by_document_id(document_id)

        chunks = ingestor.parse_txt(rel_path, document_id)
        vector_store.add_chunks(chunks)
        bm25_store.add_chunks(chunks)
        logger.info(f"Re-ingested {len(chunks)} chunks for {document_id}.")

    # --- Step 3: sync the PDF into BM25 (idempotent) ---
    if os.path.exists(PDF_PATH):
        logger.info(f"Syncing {PDF_DOC_ID} into BM25 (idempotent)")
        pdf_chunks = ingestor.parse_pdf(PDF_PATH, PDF_DOC_ID)
        bm25_store.add_chunks(pdf_chunks)
    else:
        logger.warning(f"PDF not found at {PDF_PATH}; BM25/dense corpora may remain out of sync.")

    logger.info("Reindexing complete.")
    logger.info(f"Vector store parent_cache now has {len(vector_store.parent_cache)} entries.")
    logger.info(f"BM25 store now has {len(bm25_store.documents)} documents.")


if __name__ == "__main__":
    main()