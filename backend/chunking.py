import re
from typing import List, Dict, Any, Tuple
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import logger


# ---------------------------------------------------------------------------
# FIX (#3): structure-aware chunking
# ---------------------------------------------------------------------------
# The previous implementation used one fixed character-count splitter
# (chunk_size=2400/600, same separators) for every document regardless of
# type. A clause-dense legal document, an RFC with numbered sections, a
# research paper with equations, and casual prose all have very different
# "natural" units -- splitting them all the same way means child chunks can
# get cut mid-clause, mid-numbered-section, or mid-equation, and parent
# chunks become noisy grab-bags of unrelated content. This was the most
# likely cause of `very_large_dense` being the worst-performing document
# category across every retrieval architecture in eval_results_v3.json.
#
# This version keeps the exact same chunk_document() signature and output
# shape (so vector_store/bm25_store/retriever code needs no changes), but:
#   1. classifies the document into one of a few structural "shapes" using
#      cheap regex signals on the raw text,
#   2. pre-splits on structural boundaries that match that shape (e.g.
#      "SECTION 2", "Article I", "3.2.1 Title") BEFORE handing each
#      structural unit to the generic RecursiveCharacterTextSplitter, so a
#      2400-char parent window is never forced to span two unrelated
#      clauses/sections just because that's where the character count
#      happened to land.
#   3. still falls back to the original character-count behavior whenever a
#      document doesn't match any detected structure, or a structural unit
#      is itself still larger than the parent window.

# --- Document shape detection -------------------------------------------

# Requires ALL-CAPS "SECTION" (as used by the US Constitution's actual PDF
# text extraction, e.g. "SECTION. 2") or "Article <roman numeral>"/"AMENDMENT
# <n>". Deliberately case-sensitive on SECTION specifically: lowercase/
# titlecase "Section 7.4" is extremely common in RFCs and technical specs
# (confirmed against the eval corpus: rfc9112_http.pdf uses "Section N"
# dozens of times as a cross-reference but never emits all-caps "SECTION"),
# and matching it there mis-tags spec documents as legal_clause.
_LEGAL_HEADER_RE = re.compile(
    r"^\s*(SECTION\.?\s*\d+|Article\s+[IVXLC]+|AMENDMENT\s+[IVXLC0-9]+)\b",
    re.MULTILINE,
)

# Numbered section headers (RFC/spec/academic-paper style: "4. Status Line",
# "3.2.1 Scaled Dot-Product Attention", "5.1.\xa0\xa0Field Syntax"). Verified
# against real PDF-extracted text from rfc9112_http.pdf and attention_paper.pdf:
# - PDF extraction sometimes renders header spacing as non-breaking spaces
#   (\xa0) instead of regular spaces, and sometimes leaves trailing spaces
#   before the newline -- so this pattern only anchors the *start* of the
#   line (number + title-cased word), not the end, and uses \s+ (which
#   matches \xa0 in Python's regex engine) instead of a literal space.
# - A bare line consisting of just a number (page numbers, figure/table
#   index lines from PDF extraction artifacts like "5\nTable 1: ...") is
#   excluded by requiring a title-cased word immediately after the number.
_SPEC_HEADER_RE = re.compile(
    r"^\s{0,3}(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+[A-Z][A-Za-z]",
    re.MULTILINE,
)
_CODE_FENCE_RE = re.compile(r"```|^\s{4,}\S|^\t\S", re.MULTILINE)
# FIX: the original `\$[^$]+\$` matched arbitrary prose spanning hundreds of
# characters between two UNRELATED dollar amounts (e.g. "$6 ... costs $" in
# a textbook discussing book prices), which is not LaTeX math mode at all.
# Bound the span tightly and require it to look like actual inline math
# (short, contains a digit/operator/backslash-command, no sentence-ending
# punctuation) so two incidental price mentions don't get treated as a
# formula.
_EQUATION_HINT_RE = re.compile(
    r"[∑∫∏√≤≥≈±∞∂∇]|\\[a-zA-Z]+\{|\$[^$.!?\n]{1,40}[=+\-*/^_\\][^$.!?\n]{0,40}\$"
)


def _classify_document_shape(text: str) -> str:
    """Cheap, deterministic classification used only to pick separators /
    pre-split strategy. Never raises; unknown/ambiguous text falls back to
    "prose", which reproduces the previous one-size-fits-all behavior.

    Header-numbering style (legal_clause / numbered_spec) and content
    density (code_heavy / academic_dense) are independent signals -- e.g. an
    academic paper can have BOTH numbered sections ("3.2.1 ...") AND heavy
    equation notation. We pick the *pre-split strategy* from whichever
    header style is present (since that determines parent-chunk boundaries),
    but track content density separately so academic-dense papers with
    numbered sections still get the smaller, overlap-heavy splitter
    parameters even though they're being pre-split as numbered_spec.
    """
    if not text or not text.strip():
        return "empty"

    legal_hits = len(_LEGAL_HEADER_RE.findall(text))
    spec_hits = len(_SPEC_HEADER_RE.findall(text))
    has_code = bool(_CODE_FENCE_RE.search(text))
    # Require a few equation-hint matches, not just one or two. A guideline
    # document can legitimately use "≤150 minutes" once or twice without
    # being equation-dense in the sense that motivates smaller, more
    # overlap-heavy chunking (which trades more redundant chunks for a
    # lower chance of separating a formula from its context).
    equation_hits = len(_EQUATION_HINT_RE.findall(text))
    has_equations = equation_hits >= 4

    # Require a few hits, not just one, so a single stray match (e.g. "1.1"
    # appearing once in prose) doesn't misclassify an otherwise normal
    # document and start pre-splitting on phantom boundaries.
    if legal_hits >= 2:
        return "legal_clause"
    if spec_hits >= 3:
        # An academic paper with numbered sections AND heavy equation
        # notation should still get the equation-aware splitter sizing, not
        # the generic numbered_spec one (which uses the plain prose
        # parent/child splitters). Pre-split boundaries come from the
        # numbered headers either way -- see _parent_units_for_shape.
        return "numbered_spec_academic" if has_equations else "numbered_spec"
    if has_code:
        return "code_heavy"
    if has_equations:
        return "academic_dense"
    return "prose"


# --- Structural pre-splitting --------------------------------------------

def _presplit_legal_clause(text: str) -> List[str]:
    """Split on SECTION/Article/Amendment headers so a clause never gets
    merged with an unrelated clause just because they're adjacent in the
    character stream."""
    matches = list(_LEGAL_HEADER_RE.finditer(text))
    if not matches:
        return [text]
    units = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        unit = text[start:end].strip()
        if unit:
            units.append(unit)
    # Keep any preamble before the first header (e.g. the Constitution's
    # Preamble) as its own unit instead of silently dropping it.
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            units.insert(0, preamble)
    return units


def _presplit_numbered_spec(text: str) -> List[str]:
    """Split on numbered section headers (RFC/spec style: '3.2.1 Title')."""
    matches = list(_SPEC_HEADER_RE.finditer(text))
    if not matches:
        return [text]
    units = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        unit = text[start:end].strip()
        if unit:
            units.append(unit)
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            units.insert(0, preamble)
    return units


class DocumentChunker:
    def __init__(self):
        # Default (prose) splitters -- identical to the original behavior,
        # kept as the fallback for any document shape we don't have a
        # dedicated strategy for.
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2400,
            chunk_overlap=400,
            separators=["\n\n", "\n", " ", ""]
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Code/spec text: prefer splitting on blank lines and code fences
        # before falling back to raw characters, so a function body or a
        # fenced block is much less likely to be torn in half mid-line.
        self.parent_splitter_code = RecursiveCharacterTextSplitter(
            chunk_size=2400,
            chunk_overlap=300,
            separators=["\n```\n", "\n\n", "\n", " ", ""]
        )
        self.child_splitter_code = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n```\n", "\n\n", "\n", " ", ""]
        )

        # Academic/dense text (equations, citations): smaller overlap-heavy
        # windows so a formula and its surrounding explanatory sentence are
        # more likely to land in the same window, at the cost of slightly
        # more redundant chunks -- preferable here to silently truncating an
        # equation's context, which the flat 600/2400 split was prone to.
        self.parent_splitter_academic = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=500,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.child_splitter_academic = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    @staticmethod
    def _base_metadata(metadata: Dict[str, Any] | None) -> Dict[str, Any]:
        return dict(metadata or {})

    def _splitters_for_shape(self, shape: str) -> Tuple[RecursiveCharacterTextSplitter, RecursiveCharacterTextSplitter]:
        if shape == "code_heavy":
            return self.parent_splitter_code, self.child_splitter_code
        if shape in ("academic_dense", "numbered_spec_academic"):
            return self.parent_splitter_academic, self.child_splitter_academic
        # legal_clause, numbered_spec, and prose all use the standard
        # char-count splitter for the *within-unit* split -- the structural
        # win for those shapes comes from _presplit_* giving it better unit
        # boundaries to start from, not from different chunk_size/overlap.
        return self.parent_splitter, self.child_splitter

    def _parent_units_for_shape(self, text: str, shape: str) -> List[str]:
        """Returns the list of structural "parent candidate" strings before
        the character-count splitter runs. For shapes without a dedicated
        pre-split strategy, this is just [text] (i.e. behaves exactly like
        the original implementation, which ran the character splitter over
        the whole page text directly)."""
        if shape == "legal_clause":
            return _presplit_legal_clause(text)
        if shape in ("numbered_spec", "numbered_spec_academic"):
            return _presplit_numbered_spec(text)
        return [text]

    def chunk_document(self, text: str, document_id: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split a document into a hierarchy of retrieval units.

        Returns both:
          - parent chunks   (section-level, node_type="parent")
          - child chunks    (fact-level, node_type="raw")

        This keeps exact evidence available for fact lookup while giving
        synthesis questions a larger section-level unit to fall back to.

        The splitting strategy now adapts to the document's structural shape
        (legal/clause-numbered, RFC/spec-numbered, code-heavy, equation-heavy
        academic, or generic prose) instead of using one fixed character
        window for every document. See module docstring for rationale.
        """
        metadata = self._base_metadata(metadata)

        if not text or not text.strip():
            logger.info(f"Chunking document {document_id}: empty text, returning no chunks.")
            return []

        shape = _classify_document_shape(text)
        parent_splitter, child_splitter = self._splitters_for_shape(shape)
        structural_units = self._parent_units_for_shape(text, shape)

        logger.info(f"Chunking document {document_id} (detected shape={shape}, "
                    f"{len(structural_units)} structural unit(s) before char-splitting)")

        all_chunks: List[Dict[str, Any]] = []
        chunk_index = 0
        page_num = metadata.get("page_start", 0) if metadata else 0

        # Run the (shape-appropriate) character splitter over each
        # structural unit independently, rather than over the whole text in
        # one pass. For "prose"/shapes with no pre-split strategy this is
        # exactly equivalent to the original behavior, since
        # structural_units == [text].
        parent_docs: List[str] = []
        for unit in structural_units:
            parent_docs.extend(parent_splitter.split_text(unit))

        for p_index, parent_text in enumerate(parent_docs):
            parent_id = f"{document_id}_page_{page_num}_parent_{p_index}"

            parent_chunk = {
                "chunk_id": parent_id,
                "document_id": document_id,
                "text": parent_text,
                "parent_text": parent_text,
                "chunk_index": chunk_index,
                "chunk_type": "parent",
                "node_type": "parent",
                "hierarchy_level": 1,
                "parent_id": None,
                "parent_index": p_index,
                "doc_shape": shape,
                **metadata,
            }
            all_chunks.append(parent_chunk)
            chunk_index += 1

            child_docs = child_splitter.split_text(parent_text)
            for c_index, child_text in enumerate(child_docs):
                child_id = f"{parent_id}_child_{c_index}"
                chunk_data = {
                    "chunk_id": child_id,
                    "parent_id": parent_id,
                    "document_id": document_id,
                    "text": child_text,
                    "parent_text": parent_text,
                    "chunk_index": chunk_index,
                    "chunk_type": "child",
                    "node_type": "raw",
                    "hierarchy_level": 0,
                    "parent_index": p_index,
                    "child_index": c_index,
                    "doc_shape": shape,
                    **metadata,
                }
                all_chunks.append(chunk_data)
                chunk_index += 1

        logger.info(
            f"Created {len(parent_docs)} parent chunks and "
            f"{len(all_chunks) - len(parent_docs)} child chunks for document {document_id} (shape={shape})"
        )
        return all_chunks
