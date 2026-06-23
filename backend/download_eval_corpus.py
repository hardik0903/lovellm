"""
download_eval_corpus.py
-----------------------
Downloads 7 diverse PDFs into backend/eval_corpus/ for RAG evaluation.

Usage:
    python download_eval_corpus.py          # downloads, skips existing
    python download_eval_corpus.py --force  # re-downloads everything
"""

import argparse
import os
import sys
import time
import urllib.request
import urllib.error
import ssl
from pathlib import Path
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Corpus definition
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(__file__).resolve().parent / "eval_corpus"

PDF_SOURCES: List[Dict] = [
    {
        "filename": "us_constitution.pdf",
        "description": "US Constitution (small, well-formatted, ~8 pages)",
        "category": "small_well_formatted",
        "urls": [
            "https://constitutioncenter.org/media/files/constitution.pdf",
            # fallback – GPO copy
            "https://www.govinfo.gov/content/pkg/CDOC-110hdoc50/pdf/CDOC-110hdoc50.pdf",
        ],
    },
    {
        "filename": "attention_paper.pdf",
        "description": "Attention Is All You Need – Transformer paper (academic, ~15 pages)",
        "category": "small_academic",
        "urls": [
            "https://arxiv.org/pdf/1706.03762",
            "https://arxiv.org/pdf/1706.03762v7",
        ],
    },
    {
        "filename": "rfc9112_http.pdf",
        "description": "RFC 9112 – HTTP/1.1 (medium, technical spec)",
        "category": "medium_technical_spec",
        "urls": [
            "https://www.rfc-editor.org/rfc/rfc9112.pdf",
            # fallback – RFC 2616 / RFC 7230
            "https://www.rfc-editor.org/rfc/rfc2616.pdf",
            "https://www.rfc-editor.org/rfc/rfc7230.pdf",
        ],
    },
    {
        "filename": "who_physical_activity.pdf",
        "description": "WHO 2020 Guidelines on Physical Activity and Sedentary Behaviour (medium, policy document)",
        "category": "medium_health_policy",
        "urls": [
            # Verified IRIS bitstream for the actual Physical Activity Guidelines doc.
            # (The previous primary URL pointed at the unrelated WHO Constitution PDF,
            # and the previous fallback pointed at a different US HHS document than the
            # one the golden QA set was authored against.)
            "https://iris.who.int/server/api/core/bitstreams/faa83413-d89e-4be9-bb01-b24671aef7ca/content",
            # fallback – US HHS Physical Activity Guidelines (different doc; only used if IRIS is unreachable)
            "https://health.gov/sites/default/files/2019-09/Physical_Activity_Guidelines_2nd_edition.pdf",
        ],
    },
    {
        "filename": "think_python.pdf",
        "description": "Think Python 2e (large educational textbook, ~240 pages)",
        "category": "large_educational",
        "urls": [
            "https://greenteapress.com/thinkpython2/thinkpython2.pdf",
        ],
    },
    {
        "filename": "newton_principia.pdf",
        "description": "Newton's Principia Mathematica (very large classic, ~600 pages)",
        "category": "very_large_classic",
        "urls": [
            "https://archive.org/download/newtonspmathema00newtrich/newtonspmathema00newtrich.pdf",
            # fallback – a smaller Principia scan
            "https://archive.org/download/philosophinaturi00newt/philosophinaturi00newt.pdf",
        ],
    },
    {
        "filename": "art_of_war.pdf",
        "description": "The Art of War by Sun Tzu (medium-large, archaic text, ~90 pages)",
        "category": "medium_large_archaic",
        "urls": [
            "https://archive.org/download/the-art-of-war_202101/The%20Art%20of%20War.pdf",
            "https://archive.org/download/artofwar0000sunt/artofwar0000sunt.pdf",
            # another public-domain edition
            "https://archive.org/download/TheArtOfWarBySunTzu/ArtOfWar.pdf",
        ],
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)
MAX_RETRIES = 3
RETRY_DELAY = 3  # seconds


def _sizeof_fmt(num_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:,.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:,.1f} TB"


def _progress_hook(block_num: int, block_size: int, total_size: int):
    """Print download progress on a single line."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(
            f"\r  [{bar}] {pct:5.1f}%  {_sizeof_fmt(downloaded)} / {_sizeof_fmt(total_size)}  "
        )
    else:
        sys.stdout.write(f"\r  Downloaded {_sizeof_fmt(downloaded)}...")
    sys.stdout.flush()


def _build_ssl_context() -> ssl.SSLContext:
    """Build an SSL context that works on most systems."""
    ctx = ssl.create_default_context()
    # Some archive.org mirrors have cert issues; fall back gracefully.
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


def _looks_like_pdf(path: Path) -> bool:
    """Check the file actually starts with a PDF magic header.

    Catches cases like a CDN/app-shell HTML error page being saved with a
    .pdf filename — these can be well over 1 KB, so a size-only check (the
    previous validation) silently accepts them as valid PDFs.
    """
    try:
        with open(path, "rb") as fp:
            return fp.read(5) == b"%PDF-"
    except Exception:
        return False


def download_file(url: str, dest: Path, retries: int = MAX_RETRIES) -> bool:
    """Download *url* to *dest* with retry logic. Returns True on success."""
    ssl_ctx = _build_ssl_context()
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            # Use urlretrieve-style loop for progress
            with urllib.request.urlopen(req, timeout=60, context=ssl_ctx) as resp:
                total = int(resp.headers.get("Content-Length", -1))
                block_size = 8192
                downloaded = 0
                with open(dest, "wb") as fp:
                    while True:
                        chunk = resp.read(block_size)
                        if not chunk:
                            break
                        fp.write(chunk)
                        downloaded += len(chunk)
                        _progress_hook(
                            downloaded // block_size, block_size, total
                        )
            print()  # newline after progress bar
            # Basic sanity: file should be > 1 KB
            if dest.stat().st_size < 1024:
                print(f"  [!] File too small ({dest.stat().st_size} bytes), retrying...")
                dest.unlink(missing_ok=True)
                continue
            # Magic-byte sanity: must actually be a PDF, not an HTML error/app-shell page
            # saved with a .pdf extension (this is how the WHO doc went bad previously).
            if not _looks_like_pdf(dest):
                print(f"  [!] File does not start with %PDF- magic bytes, retrying...")
                dest.unlink(missing_ok=True)
                continue
            return True
        except Exception as exc:
            print(f"\n  [X] Attempt {attempt}/{retries} failed: {exc}")
            dest.unlink(missing_ok=True)
            if attempt < retries:
                print(f"    Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
    return False


def download_with_fallbacks(entry: Dict, force: bool = False) -> str:
    """Try each URL for an entry until one succeeds.

    Returns one of: "downloaded", "skipped", "failed"
    """
    dest = CORPUS_DIR / entry["filename"]
    if dest.exists() and not force:
        if _looks_like_pdf(dest):
            print(f"[SKIP] {entry['filename']}  -- already exists, skipping")
            return "skipped"
        else:
            print(f"[!] {entry['filename']} exists but is not a valid PDF (corrupt/HTML page) -- re-downloading")

    print(f"\n[DL] {entry['filename']}")
    print(f"     {entry['description']}")

    for i, url in enumerate(entry["urls"]):
        label = f"(URL {i+1}/{len(entry['urls'])})" if len(entry['urls']) > 1 else ""
        print(f"  -> Trying {url} {label}")
        if download_file(url, dest):
            size = dest.stat().st_size
            print(f"  [OK] Saved {entry['filename']}  ({_sizeof_fmt(size)})")
            return "downloaded"
        print(f"  [X] Failed from this URL.")

    print(f"  [FAIL] All URLs failed for {entry['filename']}")
    return "failed"


# ---------------------------------------------------------------------------
# Public API – used by other modules
# ---------------------------------------------------------------------------

def get_corpus_files() -> List[Dict]:
    """Return metadata about each PDF in the eval corpus.

    Each dict has keys:
        filename, path (absolute), description, category,
        exists (bool), size_bytes (int or None)
    """
    results = []
    for entry in PDF_SOURCES:
        p = CORPUS_DIR / entry["filename"]
        exists = p.exists()
        results.append(
            {
                "filename": entry["filename"],
                "path": str(p),
                "description": entry["description"],
                "category": entry["category"],
                "exists": exists,
                "size_bytes": p.stat().st_size if exists else None,
            }
        )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download evaluation PDF corpus for RAG testing."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if files exist"
    )
    args, _ = parser.parse_known_args()  # ignore unknown args (e.g. --ollama-model from parent script)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Corpus directory: {CORPUS_DIR}\n")

    results = {"downloaded": [], "skipped": [], "failed": []}
    for entry in PDF_SOURCES:
        status = download_with_fallbacks(entry, force=args.force)
        results[status].append(entry["filename"])

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    if results["downloaded"]:
        print(f"\n[OK] Downloaded ({len(results['downloaded'])}):")
        for f in results["downloaded"]:
            p = CORPUS_DIR / f
            print(f"   - {f}  ({_sizeof_fmt(p.stat().st_size)})")
    if results["skipped"]:
        print(f"\n[SKIP] Skipped ({len(results['skipped'])}):")
        for f in results["skipped"]:
            p = CORPUS_DIR / f
            sz = _sizeof_fmt(p.stat().st_size) if p.exists() else "?"
            print(f"   - {f}  ({sz})")
    if results["failed"]:
        print(f"\n[FAIL] Failed ({len(results['failed'])}):")
        for f in results["failed"]:
            print(f"   - {f}")
    print()

    # Also show full corpus listing
    print("Full corpus listing:")
    for info in get_corpus_files():
        status = "+" if info["exists"] else "-"
        sz = _sizeof_fmt(info["size_bytes"]) if info["size_bytes"] else "N/A"
        print(f"  {status} {info['filename']:30s}  {sz:>12s}   {info['category']}")
    print()

    if results["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()