"""
vlm_retriever.py — VLM + MaxSim Document Retrieval Architecture
================================================================

Third retrieval architecture for RAG evaluation:

    Document path:  PDF Page → Image → Laplacian Filter → VLM (Qwen2.5-VL) → Description → Embeddings
    Query path:     Text Query → Multi-vector Embeddings
    Retrieval:      MaxSim(query_embeddings, doc_embeddings) → Ranked Results

Key optimisations:
    - Laplacian variance pre-filter: ~5 ms/image, cuts VLM calls 30-40 %
    - MaxSim late-interaction scoring for fine-grained multi-vector matching
    - Sentence-level embeddings for both documents and queries

Requirements:
    pip install opencv-python-headless PyMuPDF Pillow openai numpy sentence-transformers
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Optional heavy imports — guarded so the rest of the backend still loads
# even if these aren't installed.
# ---------------------------------------------------------------------------

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_VLM_MODEL = "qwen2.5vl:7b"

# Laplacian filter
DEFAULT_BLUR_THRESHOLD = 100.0   # variance below this ⇒ blurry / blank page
DEFAULT_MIN_PAGE_PIXELS = 10_000  # skip tiny rendered artefacts

# VLM
# Raised from 1024 -> 2048. At 1024, dense pages (tables, multi-column text,
# pages with >~700 words) were getting truncated mid-description by the VLM's
# own max_tokens cutoff, which silently drops trailing rows/paragraphs from the
# index rather than failing loudly (the response is still well-formed text, so
# the >= VLM_QUALITY_MIN_CHARS check doesn't catch it).
VLM_MAX_TOKENS = 2048
VLM_QUALITY_MIN_CHARS = 50  # reject VLM descriptions shorter than this

# MaxSim
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|(?<=\n)\s*')

logger = logging.getLogger("vlm_retriever")


# ===================================================================
# 1.  Laplacian Blur Filter
# ===================================================================

class LaplacianBlurFilter:
    """Pre-filters document page images before VLM processing.

    Computes Laplacian variance to measure image sharpness / information
    density.  Blank or very blurry pages are skipped, saving the ~200 ms
    VLM call.  The filter itself adds only ~5 ms per image.
    """

    def __init__(self, threshold: float = DEFAULT_BLUR_THRESHOLD):
        self.threshold = threshold
        if not HAS_CV2:
            logger.warning(
                "opencv-python-headless not installed — "
                "Laplacian blur filter disabled (all pages will be processed)."
            )

    def should_process(self, image: Image.Image) -> Tuple[bool, float]:
        """Return *(keep, variance)*.  ``keep`` is True when variance ≥ threshold."""
        if not HAS_CV2:
            return True, float("inf")

        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        if gray.size < DEFAULT_MIN_PAGE_PIXELS:
            return False, 0.0

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(laplacian.var())
        return variance >= self.threshold, variance


# ===================================================================
# 2.  PDF Page Renderer  (PyMuPDF / fitz)
# ===================================================================

class PDFPageRenderer:
    """Renders every page of a PDF to a PIL Image via PyMuPDF."""

    def __init__(self, dpi: int = 150):
        self.dpi = dpi
        if not HAS_FITZ:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF page rendering.  "
                "Install with:  pip install PyMuPDF"
            )

    def render_pages(self, pdf_path: str) -> List[Tuple[int, Image.Image]]:
        """Return a list of *(1-indexed page_num, PIL.Image)* tuples."""
        pages: List[Tuple[int, Image.Image]] = []
        doc = fitz.open(pdf_path)
        zoom = self.dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for idx in range(len(doc)):
            pix = doc[idx].get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append((idx + 1, img))

        doc.close()
        return pages

    def extract_text_layer(self, pdf_path: str) -> Dict[int, str]:
        """Return ``{1-indexed page_num: native_text}`` for every page that has
        an embedded text layer.

        Used as a fallback when the VLM call fails or returns a too-short
        description (e.g. VLM service unreachable, model returned a refusal,
        or a degenerate/garbled response). Pages with no embedded text (pure
        scans) simply won't appear in the returned dict, and the caller is
        expected to treat that as "no fallback available" rather than an
        error -- a scanned page legitimately has nothing for fitz to extract.
        """
        text_by_page: Dict[int, str] = {}
        try:
            doc = fitz.open(pdf_path)
            for idx in range(len(doc)):
                text = doc[idx].get_text("text") or ""
                text = text.strip()
                if text:
                    text_by_page[idx + 1] = text
            doc.close()
        except Exception as exc:
            logger.error("Failed to extract PDF text layer from %s: %s", pdf_path, exc)
        return text_by_page


# ===================================================================
# 3.  VLM Describer  (Qwen2.5-VL via Ollama)
# ===================================================================

class VLMDescriber:
    """Describes document page images using a Vision-Language Model.

    Uses Qwen2.5-VL-7B (or any compatible VLM) through Ollama's
    OpenAI-compatible ``/v1`` endpoint.  Target latency: ~200 ms / image.
    """

    PROMPT = (
        "You are a document analysis expert. Describe this document page thoroughly.\n\n"
        "Extract and describe ALL of the following:\n"
        "1. All visible text content (transcribe key passages verbatim)\n"
        "2. Tables: describe structure, headers, and data values\n"
        "3. Figures/charts: describe what they show, axes, labels, trends\n"
        "4. Layout structure: headings, sections, bullet points\n"
        "5. Key data points: numbers, dates, names, definitions\n"
        "6. Code snippets: transcribe exactly if present\n\n"
        "Be comprehensive. Include specific values, not just summaries.\n"
        "Output plain text only, no markdown formatting."
    )

    def __init__(
        self,
        model: str = DEFAULT_VLM_MODEL,
        base_url: str | None = None,
    ):
        self.model = model
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/") + "/v1"
        self.client = None
        self._init_client()

    # ---- internal helpers ------------------------------------------------

    def _init_client(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=self.base_url, api_key="ollama")
            logger.info("VLM describer ready: %s @ %s", self.model, self.base_url)
        except ImportError:
            logger.error("openai package not installed — VLM describer disabled.")

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ---- public API ------------------------------------------------------

    def describe(self, image: Image.Image, page_num: int = 0) -> Optional[str]:
        """Return a rich text description, or *None* on failure / low quality."""
        if self.client is None:
            return None

        b64 = self._image_to_base64(image)

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": self.PROMPT},
                    ],
                }],
                temperature=0.1,
                max_tokens=VLM_MAX_TOKENS,
            )
            desc = (resp.choices[0].message.content or "").strip()

            if len(desc) < VLM_QUALITY_MIN_CHARS:
                logger.warning(
                    "Page %d: VLM description too short (%d chars) — skipped.",
                    page_num, len(desc),
                )
                return None
            return desc

        except Exception as exc:
            logger.error("VLM failed on page %d: %s", page_num, exc)
            return None


# ===================================================================
# 4.  MaxSim Scorer
# ===================================================================

class MaxSimScorer:
    """Late-interaction MaxSim scoring (ColBERT-style).

    For query matrix *Q* (n_q × d) and document matrix *D* (n_d × d)::

        MaxSim(Q, D)  =  (1 / n_q)  ×  Σ_i  max_j  cos(Q_i, D_j)
    """

    def __init__(self, embedding_model=None):
        if embedding_model is not None:
            self.model = embedding_model
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

    # ---- text → multi-vector embedding -----------------------------------

    @staticmethod
    def _split_to_units(text: str) -> List[str]:
        """Split text into semantic units (sentences / phrases)."""
        units = SENTENCE_SPLIT_RE.split(text)
        units = [u.strip() for u in units if u.strip() and len(u.strip()) > 10]
        return units if units else ([text.strip()] if text.strip() else [])

    def embed_text(self, text: str) -> np.ndarray:
        """Embed *text* into multiple vectors (one per semantic unit).

        Returns ``np.ndarray`` of shape *(n_units, dim)*.
        """
        units = self._split_to_units(text)
        if not units:
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros((1, dim))
        return np.asarray(
            self.model.encode(units, normalize_embeddings=True)
        )

    # ---- scoring ---------------------------------------------------------

    @staticmethod
    def maxsim_score(
        query_embs: np.ndarray,
        doc_embs: np.ndarray,
    ) -> float:
        """Compute MaxSim between *query_embs* and *doc_embs*.

        Both arrays must be 2-D (n × d).  Embeddings are assumed to be
        L2-normalised, so cosine similarity = dot product.

        Returns a float in roughly [0, 1].
        """
        if query_embs.ndim == 1:
            query_embs = query_embs.reshape(1, -1)
        if doc_embs.ndim == 1:
            doc_embs = doc_embs.reshape(1, -1)

        sim = query_embs @ doc_embs.T          # (n_q, n_d)
        return float(sim.max(axis=1).mean())   # avg of per-query-token maxes


# ===================================================================
# 5.  Document Index  (in-memory, per-document)
# ===================================================================

@dataclass
class VLMPageEntry:
    """Stores VLM-processed data for a single PDF page."""
    page_num: int
    description: str
    embeddings: np.ndarray       # shape (n_units, dim)
    chunk_id: str
    laplacian_variance: float = 0.0
    # "vlm" (normal path) or "text_layer_fallback" (VLM failed/too-short and
    # we recovered using the PDF's embedded text instead of dropping the page).
    source: str = "vlm"


@dataclass
class VLMDocumentIndex:
    """In-memory index of VLM-processed pages for one document."""
    document_id: str
    pages: List[VLMPageEntry] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_page(self, entry: VLMPageEntry):
        self.pages.append(entry)

    def is_empty(self) -> bool:
        return len(self.pages) == 0


# ===================================================================
# 6.  VLM Retriever  (drop-in for rag_eval_v3)
# ===================================================================

class VLMRetriever:
    """VLM + MaxSim retriever for document-image-based retrieval.

    Architecture
    ------------
    Document : PDF → Page Images → Laplacian Filter → VLM → Descriptions → Embeddings
    Query    : Text → Multi-vector Embeddings
    Scoring  : MaxSim(query, doc) → Ranked pages

    Integrates with ``rag_eval_v3.py`` as a drop-in retriever alongside
    ``NaiveDenseRetriever`` and ``HybridRetrieverWrapper``.
    """

    def __init__(
        self,
        vlm_model: str = DEFAULT_VLM_MODEL,
        blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
        render_dpi: int = 150,
    ):
        self.blur_filter = LaplacianBlurFilter(threshold=blur_threshold)
        self.renderer = PDFPageRenderer(dpi=render_dpi)
        self.vlm = VLMDescriber(model=vlm_model)
        self.scorer = MaxSimScorer()
        self.index: Optional[VLMDocumentIndex] = None
        self.last_trace: Optional[Dict[str, Any]] = None

    # -----------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------

    def ingest_pdf(self, pdf_path: str, document_id: str) -> VLMDocumentIndex:
        """Run the full VLM pipeline on a PDF and build the page index.

        1. Render all pages to images  (fitz)
        2. Laplacian filter            (~5 ms / page)
        3. VLM describe survivors      (~200 ms / page)
        3b. Text-layer fallback        for pages where the VLM call failed
            or returned a too-short description -- recovers the page using
            its embedded PDF text instead of silently dropping it from the
            index.
        4. Embed descriptions           (sentence-transformers)

        Returns the populated :class:`VLMDocumentIndex`.
        """
        t0 = time.perf_counter()

        index = VLMDocumentIndex(document_id=document_id)
        stats: Dict[str, Any] = {
            "total_pages": 0,
            "filtered_by_laplacian": 0,
            "vlm_described": 0,
            "vlm_failed": 0,
            "text_layer_recovered": 0,
            "pages_dropped": 0,
            "total_time_ms": 0,
            "laplacian_time_ms": 0,
            "vlm_time_ms": 0,
            "embedding_time_ms": 0,
        }

        # ---- 1. render ---------------------------------------------------
        print("    [VLM] Rendering PDF pages...")
        pages = self.renderer.render_pages(pdf_path)
        stats["total_pages"] = len(pages)
        print(f"    [VLM] Rendered {len(pages)} pages")

        # ---- 2. Laplacian filter -----------------------------------------
        t_lap = time.perf_counter()
        passing: List[Tuple[int, Image.Image, float]] = []
        for page_num, img in pages:
            keep, var = self.blur_filter.should_process(img)
            if keep:
                passing.append((page_num, img, var))
            else:
                stats["filtered_by_laplacian"] += 1
        stats["laplacian_time_ms"] = (time.perf_counter() - t_lap) * 1000

        print(
            f"    [VLM] Laplacian filter: {len(passing)}/{len(pages)} pass  "
            f"({stats['filtered_by_laplacian']} filtered, "
            f"{stats['laplacian_time_ms']:.0f}ms)"
        )

        # ---- 3. VLM describe ---------------------------------------------
        t_vlm = time.perf_counter()
        described: List[Tuple[int, str, float, str]] = []  # (page_num, desc, var, source)
        failed_pages: List[Tuple[int, float]] = []  # (page_num, var) -- needs fallback
        for i, (page_num, img, var) in enumerate(passing):
            print(f"    [VLM] Processing page {page_num} ({i+1}/{len(passing)})...", flush=True)
            desc = self.vlm.describe(img, page_num=page_num)
            if desc:
                described.append((page_num, desc, var, "vlm"))
                stats["vlm_described"] += 1
            else:
                stats["vlm_failed"] += 1
                failed_pages.append((page_num, var))
        stats["vlm_time_ms"] = (time.perf_counter() - t_vlm) * 1000

        per_page_ms = stats["vlm_time_ms"] / max(stats["vlm_described"], 1)
        print(
            f"    [VLM] Described {stats['vlm_described']} pages  "
            f"({stats['vlm_failed']} failed, "
            f"{stats['vlm_time_ms']:.0f}ms total, "
            f"~{per_page_ms:.0f}ms/page)"
        )

        # ---- 3b. text-layer fallback for VLM failures ---------------------
        # Pages the VLM couldn't describe (service down, garbled/too-short
        # response, etc.) would previously just vanish from the index. If the
        # PDF has a native text layer for that page (it's a real digital
        # document, not a scan), use that instead of dropping it.
        if failed_pages:
            text_layer = self.renderer.extract_text_layer(pdf_path)
            for page_num, var in failed_pages:
                native_text = text_layer.get(page_num)
                if native_text and len(native_text) >= VLM_QUALITY_MIN_CHARS:
                    described.append((page_num, native_text, var, "text_layer_fallback"))
                    stats["text_layer_recovered"] += 1
                else:
                    stats["pages_dropped"] += 1

            if stats["text_layer_recovered"]:
                print(
                    f"    [VLM] Text-layer fallback recovered "
                    f"{stats['text_layer_recovered']}/{len(failed_pages)} failed pages"
                )
            if stats["pages_dropped"]:
                print(
                    f"    [VLM] {stats['pages_dropped']} pages dropped "
                    f"(VLM failed and no usable text layer -- likely a scan)"
                )

        # Keep page order stable regardless of which pass recovered each page.
        described.sort(key=lambda item: item[0])

        # ---- 4. embed descriptions ---------------------------------------
        t_emb = time.perf_counter()
        for page_num, desc, var, source in described:
            embs = self.scorer.embed_text(desc)
            index.add_page(VLMPageEntry(
                page_num=page_num,
                description=desc,
                embeddings=embs,
                chunk_id=f"{document_id}_vlm_page_{page_num}",
                laplacian_variance=var,
                source=source,
            ))
        stats["embedding_time_ms"] = (time.perf_counter() - t_emb) * 1000
        stats["total_time_ms"] = (time.perf_counter() - t0) * 1000

        print(
            f"    [VLM] Indexing complete: {len(index.pages)} pages indexed  "
            f"({stats['vlm_described']} via VLM, {stats['text_layer_recovered']} via text-layer fallback, "
            f"{stats['total_time_ms']:.0f}ms total)"
        )

        index.stats = stats
        self.index = index
        return index

    # -----------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k pages using MaxSim scoring.

        Returns results in the same dict format as the other retrievers so
        ``evaluate_document()`` in ``rag_eval_v3.py`` can consume them
        without any changes.
        """
        t0 = time.perf_counter() * 1000.0

        if self.index is None or self.index.is_empty():
            self.last_trace = self._empty_trace(query)
            return []

        # embed query
        t_e = time.perf_counter() * 1000.0
        q_embs = self.scorer.embed_text(query)
        embed_ms = time.perf_counter() * 1000.0 - t_e

        # score all pages
        t_s = time.perf_counter() * 1000.0
        scored: List[Tuple[float, VLMPageEntry]] = []
        for page in self.index.pages:
            s = self.scorer.maxsim_score(q_embs, page.embeddings)
            scored.append((s, page))
        scored.sort(key=lambda x: x[0], reverse=True)
        score_ms = time.perf_counter() * 1000.0 - t_s

        # build results
        all_cands: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []
        for sc, pg in scored:
            cand = {
                "chunk_id": pg.chunk_id,
                "text": pg.description[:300],
                "score": sc,
                "page": pg.page_num,
                "source": self.index.document_id,
                "laplacian_variance": pg.laplacian_variance,
            }
            all_cands.append(cand)
            if len(results) < top_k:
                results.append({
                    "chunk_id": pg.chunk_id,
                    "text": pg.description,
                    "context_text": pg.description,
                    "score": sc,
                    "metadata": {
                        "page_start": pg.page_num,
                        "page_end": pg.page_num,
                        "document_id": self.index.document_id,
                        "source_file": f"{self.index.document_id}.pdf",
                        "vlm_laplacian_variance": pg.laplacian_variance,
                        "retrieval_method": "vlm_maxsim",
                    },
                })

        total_ms = time.perf_counter() * 1000.0 - t0

        self.last_trace = {
            "route": "vlm_maxsim",
            "query_variants": [query],
            "query_features": {
                "method": "vlm_maxsim",
                "query_vectors": int(q_embs.shape[0]),
            },
            "embedding_candidates": all_cands,
            "bm25_candidates": [],
            "after_rrf": [],
            "after_rerank": all_cands[:top_k],
            "submitted_to_llm": all_cands[:top_k],
            "candidate_counts": {
                "total_indexed_pages": len(self.index.pages),
                "scored": len(scored),
                "final": len(results),
            },
            "notes": [
                f"VLM model: {self.vlm.model}",
                f"Pages indexed: {len(self.index.pages)}/{self.index.stats.get('total_pages', '?')}",
                f"Laplacian filtered: {self.index.stats.get('filtered_by_laplacian', 0)}",
            ],
            "stage_timings_ms": {
                "query_embedding": round(embed_ms, 2),
                "maxsim_scoring": round(score_ms, 2),
                "total": round(total_ms, 2),
            },
        }

        return results

    def route_for(self, query: str) -> str:
        """Fixed route label for the eval report."""
        return "vlm_maxsim"

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _empty_trace(query: str) -> Dict[str, Any]:
        return {
            "route": "vlm_maxsim",
            "query_variants": [query],
            "query_features": {},
            "embedding_candidates": [],
            "bm25_candidates": [],
            "after_rrf": [],
            "after_rerank": [],
            "submitted_to_llm": [],
            "candidate_counts": {
                "total_indexed_pages": 0,
                "scored": 0,
                "final": 0,
            },
            "notes": ["No VLM index available"],
            "stage_timings_ms": {"total": 0},
        }