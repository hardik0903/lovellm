import asyncio
from pipeline import PipelineOrchestrator
from generator import AnswerGenerator

async def test_route(query: str):
    print(f"\n--- Testing: {query} ---")
    po = PipelineOrchestrator(AnswerGenerator(), None)
    async for e in po.execute(query):
        if e.get("event") == "final":
            print("FINAL EVENT:", e)

async def main():
    await test_route("what is eigenvector")
    await test_route("difference between eigenvector and eigenvalue")

if __name__ == "__main__":
    asyncio.run(main())
