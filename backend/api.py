import os
import shutil
import asyncio
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

from logger import logger
from ingestion import DocumentIngestor
from vector_store import VectorStore
from bm25_store import BM25Store
from retriever import HybridRetriever
from generator import AnswerGenerator
from pipeline import PipelineOrchestrator
from raptor import list_cached_raptor_trees, load_cached_raptor_tree, render_raptor_tree_mermaid, render_raptor_tree_dot

app = FastAPI(title="Document Q&A Service")

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global components
os.makedirs("./data/uploads", exist_ok=True)
vector_store = VectorStore(persist_dir="./data/chroma")
bm25_store = BM25Store(persist_dir="./data/bm25")
retriever = HybridRetriever(vector_store, bm25_store)
generator = AnswerGenerator()
ingestor = DocumentIngestor()
orchestrator = PipelineOrchestrator(generator, retriever)

 
from telemetry_auth import auth_router
from telemetry_api  import router as telemetry_router
 
# Mount both — auth_router first (provides /telemetry/auth/* endpoints)
app.include_router(auth_router,      prefix="/telemetry")
app.include_router(telemetry_router, prefix="/telemetry")
 
# ─ What this gives you ─────────────────────────────────────────────────────
# Public  (no token needed):
#   POST /telemetry/auth/login    { "password": "..." } → { "token": "..." }
#   GET  /telemetry/auth/verify
#
# Protected (Bearer token required):
#   GET  /telemetry/summary
#   GET  /telemetry/routing/timeseries
#   GET  /telemetry/routing/confidence_distribution
#   GET  /telemetry/routing/score_heatmap
#   GET  /telemetry/agents/latency
#   GET  /telemetry/agents/error_rate
#   GET  /telemetry/recent_decisions
#   GET  /telemetry/health
 
# @app.on_event("startup")
# async def startup_event():
#     logger.info("Pre-warming reranker model...")
#     await asyncio.to_thread(retriever.reranker._load_model)

# backend/api.py
@app.on_event("startup")
async def startup_event():
    logger.info("Pre-warming reranker model...")

    warmup = getattr(retriever, "_load_reranker", None)
    if callable(warmup):
        await asyncio.to_thread(warmup)
        return

    reranker = getattr(retriever, "reranker", None)
    if reranker is not None and hasattr(reranker, "_load_model"):
        await asyncio.to_thread(reranker._load_model)

class ChatRequest(BaseModel):
    query: str
    mode: str = "auto"

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    logger.info(f"Received upload request for file: {file.filename}")
    
    file_path = os.path.join("./data/uploads", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    doc_id = file.filename.replace(" ", "_").lower()
    
    try:
        if file.filename.endswith(".pdf"):
            chunks = ingestor.parse_pdf(file_path, doc_id)
        elif file.filename.endswith(".txt"):
            chunks = ingestor.parse_txt(file_path, doc_id)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF or TXT.")
            
        if not chunks:
            raise HTTPException(status_code=400, detail="No extractable text found in document.")
            
        # Index in both stores
        vector_store.add_chunks(chunks)
        await asyncio.to_thread(bm25_store.add_chunks, chunks)

        # FIX (#5): the in-process query cache has no awareness that the
        # corpus just changed. Without this, a query answered before this
        # upload (e.g. "I could not find that in the documents") could keep
        # being served from cache indefinitely within this worker's
        # lifetime, even though the newly-ingested document now answers it.
        orchestrator.invalidate_cache()

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError as e:
                logger.error(f"Error removing uploaded file {file_path}: {e}")
                
        return {"message": "Document ingested successfully", "chunks_processed": len(chunks), "doc_id": doc_id}
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    query = request.query
    mode = request.mode
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    logger.info(f"Received chat query: {query} with mode: {mode}")
    
    try:
        has_docs = vector_store.collection.count() > 0
    except Exception as e:
        logger.error(f"Failed to read vector store collection count: {e}")
        has_docs = False

    return EventSourceResponse(orchestrator.execute(query, mode=mode, has_documents=has_docs))

@app.get("/health")
def health_check():
    health_status = {
        "status": "ok",
        "groq_key_configured": bool(os.getenv("GROQ_API_KEY")),
        "vector_store_reachable": vector_store.collection.count() >= 0,
        "bm25_index_loaded": len(bm25_store.documents) >= 0
    }
    return health_status


@app.get("/raptor/trees/{document_id}")
def list_raptor_trees(document_id: str):
    """List all cached RAPTOR trees for a document id."""
    return {"document_id": document_id, "trees": list_cached_raptor_trees(document_id)}


@app.get("/raptor/tree/{document_id}")
def get_raptor_tree(document_id: str, fingerprint: str | None = None):
    """Return the persisted RAPTOR tree JSON for a document.

    If fingerprint is omitted, the most recently modified cached version is used.
    """
    tree = load_cached_raptor_tree(document_id, fingerprint=fingerprint)
    if tree is None:
        raise HTTPException(status_code=404, detail=f"No cached RAPTOR tree found for {document_id}")
    return {
        "document_id": document_id,
        "fingerprint": tree.stats.get("fingerprint"),
        "tree": {
            "document_id": tree.document_id,
            "levels": {str(k): [n.to_chunk_dict() for n in v] for k, v in tree.levels.items()},
            "stats": tree.stats,
        },
    }


@app.get("/raptor/tree/{document_id}/mermaid")
def get_raptor_tree_mermaid(document_id: str, fingerprint: str | None = None):
    """Return a Mermaid representation of the cached RAPTOR tree."""
    tree = load_cached_raptor_tree(document_id, fingerprint=fingerprint)
    if tree is None:
        raise HTTPException(status_code=404, detail=f"No cached RAPTOR tree found for {document_id}")
    return {
        "document_id": document_id,
        "fingerprint": tree.stats.get("fingerprint"),
        "mermaid": render_raptor_tree_mermaid(tree),
    }


@app.get("/raptor/tree/{document_id}/dot")
def get_raptor_tree_dot(document_id: str, fingerprint: str | None = None):
    """Return a Graphviz DOT representation of the cached RAPTOR tree."""
    tree = load_cached_raptor_tree(document_id, fingerprint=fingerprint)
    if tree is None:
        raise HTTPException(status_code=404, detail=f"No cached RAPTOR tree found for {document_id}")
    return {
        "document_id": document_id,
        "fingerprint": tree.stats.get("fingerprint"),
        "dot": render_raptor_tree_dot(tree),
    }
