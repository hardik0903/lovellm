import os
import re
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from chunking import DocumentChunker
from logger import logger

# ---------------------------------------------------------------------------
# E-1 FIX: Logical page number extraction
# ---------------------------------------------------------------------------
# PdfReader enumerates pages by *physical* index (0-based → stored as i+1).
# But golden-QA datasets record the *logical* page number printed on the page.
# For any PDF with front-matter (cover, TOC, preface) these diverge by exactly
# the front-matter count, so `relevant_pages & retrieved_pages` evaluates to
# the empty set and hit-rate = 0 even when the correct chunk was retrieved.
#
# This heuristic scans the first and last 300 characters of the extracted
# text for a bare standalone integer (the most common footer/header style) or
# a "Page N" / "- N -" label, and uses it as the logical page_start when
# found. It falls back gracefully to i+1 when no printed number is detected.
# ---------------------------------------------------------------------------

# Patterns tried in order (first match wins):
#   1. Bare integer on its own line (trim whitespace) — catches most headers/footers
#   2. "Page N" or "Page N of M"
#   3. Dashes-wrapped: "- N -" or "— N —"
_PAGE_NUM_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(\d{1,4})\s*(?:\n|$)"),        # bare number on a line
    re.compile(r"\bpage\s+(\d{1,4})(?:\s+of\s+\d+)?\b", re.IGNORECASE),
    re.compile(r"[-–—]\s*(\d{1,4})\s*[-–—]"),              # - N - or — N —
]

def _extract_printed_page_number(text: str) -> Optional[int]:
    """Return the logical page number printed on the page, or None if unknown.

    Searches the first and last 300 chars (header / footer zones) so that
    body text numerals (e.g. numbered lists, years) don't produce false hits.
    """
    if not text:
        return None

    # Only inspect header / footer zones to avoid false hits in body text.
    head = text[:300]
    tail = text[-300:]
    zones = [head, tail]

    for zone in zones:
        for pat in _PAGE_NUM_PATTERNS:
            m = pat.search(zone)
            if m:
                try:
                    num = int(m.group(1))
                    # Sanity bounds: PDF page numbers are typically 1–9999.
                    # Numbers like years (1776, 2024) or section refs (999+) are
                    # still possible in footers, so we keep a wide upper bound
                    # and rely on the zone restriction to limit false positives.
                    if 1 <= num <= 9999:
                        return num
                except (ValueError, IndexError):
                    pass

    return None


class DocumentIngestor:
    def __init__(self):
        self.chunker = DocumentChunker()

    def clean_text(self, text: str) -> str:
        """Removes excessive whitespace and standardizes newlines."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()

    def parse_pdf(self, file_path: str, document_id: str) -> List[Dict[str, Any]]:
        """Parses a PDF, extracting text page by page.

        E-1 FIX: stores the logical (printed) page number in ``page_start``
        and ``page_end`` so that evaluation hit-rate metrics match the
        ``relevant_pages`` values in the golden-QA dataset (which are based on
        what the reader sees, not PdfReader's 0-based physical index).

        The raw physical index is preserved in ``physical_page`` for
        diagnostics and any code that needs the PdfReader-native position.
        """
        logger.info(f"Parsing PDF: {file_path}")
        chunks = []
        try:
            reader = PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                physical_page = i + 1          # PdfReader physical index (1-based)
                text = page.extract_text()
                if not text:
                    continue

                text = self.clean_text(text)
                if not text:
                    continue

                # Attempt to read the printed page number from the page text.
                # Falls back to the physical index when none is detected.
                logical_page = _extract_printed_page_number(text)
                if logical_page is None:
                    logical_page = physical_page
                    logger.debug(
                        f"No printed page number found on physical page {physical_page} "
                        f"of '{os.path.basename(file_path)}'; using physical index."
                    )
                else:
                    logger.debug(
                        f"Physical page {physical_page} → logical page {logical_page} "
                        f"in '{os.path.basename(file_path)}'"
                    )

                metadata = {
                    "page_start":    logical_page,
                    "page_end":      logical_page,
                    "physical_page": physical_page,
                    "source_file":   os.path.basename(file_path),
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
            "page_start":    1,
            "page_end":      1,
            "physical_page": 1,
            "source_file":   os.path.basename(file_path),
        }
        return self.chunker.chunk_document(text, document_id, metadata)

