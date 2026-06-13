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

@app.on_event("startup")
async def startup_event():
    logger.info("Pre-warming reranker model...")
    await asyncio.to_thread(retriever.reranker._load_model)

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
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    logger.info(f"Received chat query: {query} with mode: {mode}")
    
    has_docs = vector_store.collection.count() > 0
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
