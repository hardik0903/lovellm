from logger import logger
from vector_store import VectorStore
from bm25_store import BM25Store
from retriever import HybridRetriever

def run_evaluations():
    logger.info("Starting retrieval evaluations...")
    # This is a skeleton. In a real system, you'd have a test set of questions and expected chunk IDs.
    
    test_queries = [
        "What is the main topic of the document?",
        "Compare feature A and feature B."
    ]
    
    vector_store = VectorStore(persist_dir="./data/chroma")
    bm25_store = BM25Store(persist_dir="./data/bm25")
    retriever = HybridRetriever(vector_store, bm25_store)
    
    for query in test_queries:
        logger.info(f"Evaluating query: {query}")
        results = retriever.retrieve(query, top_k=3)
        logger.info(f"Found {len(results)} chunks.")
        
    logger.info("Evaluations completed.")

if __name__ == "__main__":
    run_evaluations()
