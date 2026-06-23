import os
import re
from typing import List, Dict, Any
from pypdf import PdfReader
from chunking import DocumentChunker
from logger import logger

class DocumentIngestor:
    def __init__(self):
        self.chunker = DocumentChunker()

    def clean_text(self, text: str) -> str:
        """Removes excessive whitespace and standardizes newlines."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()

    def parse_pdf(self, file_path: str, document_id: str) -> List[Dict[str, Any]]:
        """Parses a PDF, extracting text page by page."""
        logger.info(f"Parsing PDF: {file_path}")
        chunks = []
        try:
            reader = PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue
                
                text = self.clean_text(text)
                if not text:
                    continue

                metadata = {
                    "page_start": i + 1,
                    "page_end": i + 1,
                    "source_file": os.path.basename(file_path)
                }
                
                # Chunk the page text
                page_chunks = self.chunker.chunk_document(text, document_id, metadata)
                chunks.extend(page_chunks)
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            raise e
        
        return chunks

    def parse_txt(self, file_path: str, document_id: str) -> List[Dict[str, Any]]:
        """Parses a raw text file."""
        logger.info(f"Parsing TXT: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        text = self.clean_text(text)
        metadata = {
            "page_start": 1,
            "page_end": 1,
            "source_file": os.path.basename(file_path)
        }
        return self.chunker.chunk_document(text, document_id, metadata)
