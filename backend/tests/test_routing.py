import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _path_setup  # noqa: F401

import asyncio
from pipeline import PipelineOrchestrator
from generator import AnswerGenerator

async def test_route(query: str):
    print(f"\n--- Testing: {query} ---")
    po = PipelineOrchestrator(AnswerGenerator(), None)
    async for e in po.execute(query):
        if e.get("event") == "final":
            print("FINAL EVENT:", e)
        elif e.get("event") == "delta":
            print("DELTA:", e.get("data"))

async def main():
    await test_route("According to this document, who is the project manager for Alpha?")

if __name__ == "__main__":
    asyncio.run(main())
