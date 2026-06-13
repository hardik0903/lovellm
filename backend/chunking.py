from typing import List, Dict, Any
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import logger

class DocumentChunker:
    def __init__(self):
        # We use a character text splitter as a proxy for tokens for simplicity,
        # but realistically you'd use TokenTextSplitter (e.g., tiktoken) if you need strict token bounds.
        # For Groq's llama-3.1, a rough approximation of 4 chars per token is fine.
        # Parent chunk: ~600 tokens -> 2400 chars. Overlap ~100 tokens -> 400 chars.
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2400,
            chunk_overlap=400,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Child chunk: ~150 tokens -> 600 chars. Overlap ~30 tokens -> 120 chars.
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_document(self, text: str, document_id: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunks the document into parent and child chunks, preserving metadata.
        Returns a flat list of child chunks, each carrying a reference to its parent chunk's text and ID.
        """
        if metadata is None:
            metadata = {}

        logger.info(f"Chunking document {document_id}")
        parent_docs = self.parent_splitter.split_text(text)
        
        all_child_chunks = []
        chunk_index = 0

        page_num = metadata.get("page_start", 0) if metadata else 0
        for p_index, parent_text in enumerate(parent_docs):
            parent_id = f"{document_id}_page_{page_num}_parent_{p_index}"
            
            child_docs = self.child_splitter.split_text(parent_text)
            
            for c_index, child_text in enumerate(child_docs):
                child_id = f"{parent_id}_child_{c_index}"
                
                chunk_data = {
                    "chunk_id": child_id,
                    "parent_id": parent_id,
                    "document_id": document_id,
                    "text": child_text,
                    "parent_text": parent_text, # Embed parent text directly for ease of access during generation
                    "chunk_index": chunk_index,
                    "chunk_type": "child",
                    **metadata
                }
                all_child_chunks.append(chunk_data)
                chunk_index += 1

        logger.info(f"Created {len(parent_docs)} parent chunks and {len(all_child_chunks)} child chunks for document {document_id}")
        return all_child_chunks
