"""
generate_golden_qa.py
----------------------
Generates new golden-QA questions for the RAG eval harness, grounded directly
in real per-page PDF text -- so relevant_pages is set from the page the LLM
was actually shown, and cannot drift from the document the way the original
hand-authored golden QA set did (see eval_golden_qa.json fix notes / chat
history: several categories had ground-truth pages that were off by the
front-matter offset, or in one case described content that doesn't exist in
the document at all).

This script does NOT call out to the model on its own judgement of "what page
is X on" -- it always extracts the real page text first (using the exact same
PdfReader + page_start = i+1 logic as ingestion.py), shows that text to the
LLM, and tags the resulting question with the page(s) it was shown. That
structurally rules out the page-offset class of bug.

Usage:
    export GROQ_API_KEY=...
    export GROQ_API_KEY_2=...        # optional, for more throughput
    export GROQ_API_KEY_3=...
    export GROQ_API_KEY_4=...

    # Generate ~80 new questions for one category, write to a staging file:
    python generate_golden_qa.py --doc attention_paper --target 80

    # Generate for every category that's under its target, then merge
    # straight into eval_golden_qa.json with continued IDs:
    python generate_golden_qa.py --all --target 80 --merge

Requires: pip install groq pypdf
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _path_setup  # noqa: F401

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pypdf import PdfReader

try:
    from groq import AsyncGroq
except ImportError:
    print("Missing dependency: pip install groq", file=sys.stderr)
    raise

BACKEND_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BACKEND_DIR / "eval_corpus"
GOLDEN_QA_PATH = BACKEND_DIR / "eval_golden_qa.json"
STAGING_DIR = BACKEND_DIR / "generated_qa_staging"

MODEL = "llama-3.3-70b-versatile"  # stronger generation quality than the 8b answer model
ID_PREFIXES = {
    "us_constitution": "uc",
    "attention_paper": "ai",
    "rfc9112_http": "rfc",
    "who_physical_activity": "who",
    "think_python": "tp",
    "newton_principia": "np",
    "art_of_war": "aw",
}

QUESTION_TYPES = ["factual_lookup", "list_extraction", "comparison", "negation", "multi_hop"]
DIFFICULTIES = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are building a golden evaluation set for a RAG (retrieval-augmented generation) system.

You will be shown the exact text of one or more consecutive pages from a real document, \
labeled with their true page numbers. Your job is to write {n} question-answer pairs that:

1. Are answerable ONLY using the text shown below -- do not use outside knowledge, and do \
not invent facts, numbers, names, or section references that aren't in the text.
2. Have a ground_truth_answer that is fully supported by the shown text (a human checking \
the answer against the page text should be able to verify every claim in it).
3. Vary in question_type across: factual_lookup (a direct fact), list_extraction (asks for \
an enumerated list of items), comparison (compares two things mentioned in the text), \
negation (asks what is NOT true, or what is excluded/absent), multi_hop (requires combining \
two distinct facts from the shown text). Not every type needs to appear if the text doesn't \
support it -- skip a type rather than force a bad question.
4. Vary in difficulty across easy/medium/hard.
5. Set relevant_pages to EXACTLY the page number(s) shown to you below -- never a page you \
were not shown.

Return ONLY a JSON array (no preamble, no markdown fences) of objects with this exact shape:
[{{"query": "...", "ground_truth_answer": "...", "question_type": "...", "difficulty": "...", "relevant_pages": [<int>, ...]}}]

If the shown text is too sparse, fragmentary, or boilerplate (e.g. a mostly-blank page, a \
table of contents, a references list) to support {n} good grounded questions, return fewer \
items, or an empty array -- do not pad with low-quality questions.
"""


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def extract_pages(pdf_path: Path) -> List[str]:
    """Mirrors ingestion.py's DocumentIngestor.parse_pdf page extraction exactly,
    so page_start here means the same thing it means at retrieval time."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(clean_text(text))
    return pages


def make_windows(pages: List[str], window_size: int, max_windows: Optional[int]) -> List[Dict[str, Any]]:
    """Groups pages into windows of `window_size` consecutive pages (1-indexed),
    skipping windows whose combined text is too short to be useful. If the
    document has more candidate windows than max_windows, samples evenly
    across the document rather than just taking the first N (so a 600-page
    book doesn't only get questions from its first 50 pages)."""
    windows = []
    n = len(pages)
    for start in range(0, n, window_size):
        page_nums = list(range(start + 1, min(start + window_size, n) + 1))
        text = "\n\n".join(
            f"[Page {p}]\n{pages[p - 1]}" for p in page_nums if pages[p - 1].strip()
        )
        if len(text) < 200:  # skip near-blank / boilerplate windows
            continue
        windows.append({"pages": page_nums, "text": text})

    if max_windows and len(windows) > max_windows:
        # Even sampling across the whole document for good coverage.
        step = len(windows) / max_windows
        sampled = [windows[int(i * step)] for i in range(max_windows)]
        windows = sampled

    return windows


class KeyPool:
    """Round-robins across however many GROQ_API_KEY[, _2, _3, _4] env vars are set,
    with a per-key semaphore so we don't blow past each key's own rate limit."""

    def __init__(self, concurrency_per_key: int = 1):
        keys = []
        for name in ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEY_4"]:
            val = os.getenv(name)
            if val:
                keys.append(val)
        if not keys:
            raise ValueError("No GROQ_API_KEY[_2/_3/_4] environment variables set.")
        self.clients = [AsyncGroq(api_key=k) for k in keys]
        self.semaphores = [asyncio.Semaphore(concurrency_per_key) for _ in keys]
        self._next = 0
        print(f"[KeyPool] Using {len(keys)} Groq API key(s), concurrency {concurrency_per_key} each "
              f"({len(keys) * concurrency_per_key} max in-flight requests).")

    def lease(self):
        idx = self._next
        self._next = (self._next + 1) % len(self.clients)
        return idx, self.clients[idx], self.semaphores[idx]


async def generate_for_window(pool: KeyPool, window: Dict[str, Any], n: int, retries: int = 4) -> List[Dict[str, Any]]:
    idx, client, sem = pool.lease()
    page_label = ", ".join(str(p) for p in window["pages"])
    user_prompt = f"Pages shown: {page_label}\n\n{window['text'][:9000]}"

    for attempt in range(1, retries + 1):
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT.format(n=n)},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.4,
                    max_tokens=2000,
                )
                raw = resp.choices[0].message.content.strip()
                raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
                items = json.loads(raw)
                if not isinstance(items, list):
                    return []
                # Safety net: force relevant_pages to exactly what we showed,
                # in case the model drifts -- this is the structural guarantee.
                out = []
                for it in items:
                    if not isinstance(it, dict) or "query" not in it or "ground_truth_answer" not in it:
                        continue
                    it["relevant_pages"] = [p for p in window["pages"]]
                    it.setdefault("question_type", "factual_lookup")
                    it.setdefault("difficulty", "medium")
                    out.append(it)
                return out
            except json.JSONDecodeError as e:
                print(f"  [key {idx}] JSON parse failed (attempt {attempt}/{retries}) for pages {page_label}: {e}")
            except Exception as e:
                msg = str(e)
                wait = 2 ** attempt + random.random()
                print(f"  [key {idx}] error (attempt {attempt}/{retries}) for pages {page_label}: {msg} -- waiting {wait:.1f}s")
                await asyncio.sleep(wait)
    print(f"  [key {idx}] giving up on pages {page_label} after {retries} attempts")
    return []


async def generate_for_document(
    doc_id: str,
    target_new: int,
    pool: KeyPool,
    window_size: int,
    per_window: int,
    max_windows: Optional[int],
) -> List[Dict[str, Any]]:
    pdf_path = CORPUS_DIR / f"{doc_id}.pdf"
    if not pdf_path.exists():
        print(f"[{doc_id}] No PDF found at {pdf_path}, skipping.")
        return []

    print(f"[{doc_id}] Extracting page text from {pdf_path.name}...")
    pages = extract_pages(pdf_path)
    windows = make_windows(pages, window_size=window_size, max_windows=max_windows)
    random.shuffle(windows)  # avoid always hitting the same early pages first if we stop early
    print(f"[{doc_id}] {len(pages)} pages -> {len(windows)} candidate windows.")

    results: List[Dict[str, Any]] = []
    tasks = []
    needed_windows = max(1, -(-target_new // per_window))  # ceil
    for w in windows[: needed_windows * 2]:  # generous oversample; we'll trim/skip empties
        tasks.append(generate_for_window(pool, w, per_window))

    for coro in asyncio.as_completed(tasks):
        items = await coro
        results.extend(items)
        if len(results) >= target_new:
            break

    print(f"[{doc_id}] Generated {len(results)} new grounded question(s).")
    return results[:target_new] if len(results) > target_new else results


def assign_ids(doc_id: str, existing_ids: set, new_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prefix = ID_PREFIXES.get(doc_id, doc_id[:3])
    nums = [int(m.group(1)) for i in existing_ids if (m := re.search(r"(\d+)$", i))]
    next_num = (max(nums) + 1) if nums else 1
    out = []
    for item in new_items:
        item = dict(item)
        item["id"] = f"{prefix}_q{next_num}"
        next_num += 1
        out.append(item)
    return out


def load_golden_qa() -> Dict[str, Any]:
    with open(GOLDEN_QA_PATH) as f:
        return json.load(f)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", help="Single document id to generate for, e.g. attention_paper")
    ap.add_argument("--all", action="store_true", help="Generate for every category under --target")
    ap.add_argument("--target", type=int, default=80, help="Target total question count per category")
    ap.add_argument("--window-size", type=int, default=2, help="Pages per generation window")
    ap.add_argument("--per-window", type=int, default=3, help="Questions requested per window")
    ap.add_argument("--max-windows", type=int, default=60, help="Cap windows sampled per doc (for very long docs)")
    ap.add_argument("--concurrency-per-key", type=int, default=2)
    ap.add_argument("--merge", action="store_true", help="Write straight into eval_golden_qa.json instead of a staging file")
    args = ap.parse_args()

    if not args.doc and not args.all:
        ap.error("Pass --doc <id> or --all")

    pool = KeyPool(concurrency_per_key=args.concurrency_per_key)
    data = load_golden_qa()
    doc_by_id = {d["id"]: d for d in data["documents"]}

    targets = [args.doc] if args.doc else list(doc_by_id.keys())
    STAGING_DIR.mkdir(exist_ok=True)

    for doc_id in targets:
        doc = doc_by_id.get(doc_id)
        if doc is None:
            print(f"Unknown document id: {doc_id}")
            continue
        current = len(doc["questions"])
        need = args.target - current
        if need <= 0:
            print(f"[{doc_id}] Already at {current} >= target {args.target}, skipping.")
            continue

        print(f"[{doc_id}] Have {current}, need {need} more to reach {args.target}.")
        t0 = time.time()
        new_items = await generate_for_document(
            doc_id, need, pool,
            window_size=args.window_size,
            per_window=args.per_window,
            max_windows=args.max_windows,
        )
        existing_ids = {q["id"] for q in doc["questions"]}
        new_items = assign_ids(doc_id, existing_ids, new_items)
        print(f"[{doc_id}] Done in {time.time() - t0:.1f}s, {len(new_items)} items.")

        staging_path = STAGING_DIR / f"{doc_id}_new.json"
        with open(staging_path, "w") as f:
            json.dump(new_items, f, indent=2, ensure_ascii=False)
        print(f"[{doc_id}] Wrote staging file: {staging_path}")

        if args.merge:
            doc["questions"].extend(new_items)

    if args.merge:
        with open(GOLDEN_QA_PATH, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nMerged new questions directly into {GOLDEN_QA_PATH}")
    else:
        print(f"\nNot merged (no --merge flag). Review the staging files in {STAGING_DIR}/ "
              f"and merge manually once you've spot-checked a sample of them.")


if __name__ == "__main__":
    asyncio.run(main())