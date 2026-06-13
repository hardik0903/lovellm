import os
import json
import asyncio
import time
import httpx
from datetime import datetime
from dotenv import load_dotenv

# Load backend imports for unit tests
from router import QueryRouter
from bm25_store import BM25Store
from vector_store import VectorStore
from reranker import Reranker
from groq import AsyncGroq

load_dotenv()

BASE_URL = "http://localhost:8000"

class TestHarness:
    def __init__(self):
        self.results = {
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "failing_modules": set(),
            "details": []
        }

    def _log_result(self, module_name, test_name, status, error_msg=None, doc_id=None):
        timestamp = datetime.utcnow().isoformat()
        detail = {
            "timestamp": timestamp,
            "module_name": module_name,
            "test_name": test_name,
            "status": status,
        }
        if error_msg:
            detail["error_message"] = str(error_msg)
        if doc_id:
            detail["document_id"] = doc_id

        self.results["details"].append(detail)
        
        if status == "PASS":
            self.results["passed"] += 1
            print(f"[\033[92mPASS\033[0m] {module_name} - {test_name}")
        elif status == "WARN":
            self.results["warnings"] += 1
            print(f"[\033[93mWARN\033[0m] {module_name} - {test_name}: {error_msg}")
        else:
            self.results["failed"] += 1
            self.results["failing_modules"].add(module_name)
            print(f"[\033[91mFAIL\033[0m] {module_name} - {test_name}: {error_msg}")

    async def check_health(self):
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{BASE_URL}/health")
                r.raise_for_status()
                data = r.json()
                assert data["status"] == "ok", "Status not ok"
                assert data["groq_key_configured"], "Groq key missing"
                self._log_result("API", "Health Check", "PASS")
        except Exception as e:
            self._log_result("API", "Health Check", "FAIL", e)

    async def check_groq_dryrun(self):
        try:
            client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
            r = await client.chat.completions.create(
                messages=[{"role": "user", "content": "Return exactly the word OK"}],
                model="llama-3.1-8b-instant",
                max_tokens=5
            )
            assert "ok" in r.choices[0].message.content.lower(), "Groq dry run did not return expected output"
            self._log_result("Generation", "Groq Dry Run", "PASS")
        except Exception as e:
            self._log_result("Generation", "Groq Dry Run", "FAIL", e)

    async def check_ingestion(self):
        try:
            async with httpx.AsyncClient() as client:
                # Test dummy.txt
                with open("tests/fixtures/dummy.txt", "rb") as f:
                    r1 = await client.post(f"{BASE_URL}/upload", files={"file": ("dummy.txt", f, "text/plain")})
                    r1.raise_for_status()
                    assert r1.json()["chunks_processed"] > 0
                    
                # Test semantic_fixture.txt
                with open("tests/fixtures/semantic_fixture.txt", "rb") as f:
                    r2 = await client.post(f"{BASE_URL}/upload", files={"file": ("semantic_fixture.txt", f, "text/plain")})
                    r2.raise_for_status()
                    assert r2.json()["chunks_processed"] > 0
                    
            self._log_result("Ingestion", "Upload Fixtures", "PASS")
        except Exception as e:
            self._log_result("Ingestion", "Upload Fixtures", "FAIL", e)

    def check_unit_retrieval(self):
        try:
            # Test Doc RAG Router
            router = QueryRouter()
            assert router.route("Project Alpha 2027-10-15") == "bm25", "Router failed BM25 rule"
            assert router.route("Explain how it differs from evolutionary algorithms") == "dense", "Router failed dense rule"
            self._log_result("Router", "Doc RAG Routing Logic", "PASS")
            
            # Since we hit /upload, the local DBs should have data
            bm25 = BM25Store(persist_dir="./data/bm25")
            vstore = VectorStore(persist_dir="./data/chroma")
            
            # BM25 test
            bm_res = bm25.search("Sarah Jenkins")
            assert len(bm_res) > 0, "BM25 failed to retrieve exact match"
            self._log_result("Retrieval", "BM25 Exact Match", "PASS")
            
            # Dense test
            dense_res = vstore.search("What is the advantage of using state superposition?")
            assert len(dense_res) > 0, "Dense failed to retrieve semantic match"
            self._log_result("Retrieval", "Dense Semantic Match", "PASS")
            
            # Reranker conditional trigger
            reranker = Reranker()
            assert reranker.should_rerank("compare QNAS and backpropagation", [{"score": 1.0}, {"score": 0.9}]) == True, "Reranker failed compare trigger"
            assert reranker.should_rerank("single query", [{"score": 1.0}]) == False, "Reranker triggered incorrectly on 1 candidate"
            self._log_result("Reranking", "Reranker Triggers", "PASS")
            
        except Exception as e:
            self._log_result("Retrieval", "Unit Checks", "FAIL", e)

    async def check_generation_and_streaming(self):
        try:
            query = "According to this document, who is the project manager for Alpha?"
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", f"{BASE_URL}/chat", json={"query": query}) as response:
                    response.raise_for_status()
                    
                    has_delta = False
                    final_json = None
                    async for line in response.aiter_lines():
                        if line.startswith("event: delta"):
                            has_delta = True
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str:
                                try:
                                    parsed = json.loads(data_str)
                                    if "answer" in parsed and "sources" in parsed and "mode" in parsed:
                                        final_json = parsed
                                except:
                                    pass
                                    
                    assert has_delta, "SSE did not emit delta events"
                    assert final_json is not None, "SSE did not emit final JSON"
                    assert "Sarah Jenkins" in final_json["answer"], "LLM hallucinated or failed to answer correctly"
                    assert len(final_json["sources"]) > 0, "Sources were empty"
                    assert final_json["mode"] == "doc_rag", f"Incorrect mode: {final_json['mode']}"
                    
            self._log_result("Streaming", "SSE Delta & Final JSON", "PASS", doc_id="query_1")
            
        except Exception as e:
            self._log_result("Streaming", "SSE Delta & Final JSON", "FAIL", e, doc_id="query_1")

    async def check_fallback(self):
        try:
            query = "What is the secret code for the nuclear launch?"
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", f"{BASE_URL}/chat", json={"query": query}) as response:
                    final_json = None
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str:
                                try:
                                    parsed = json.loads(data_str)
                                    if "answer" in parsed and "sources" in parsed:
                                        final_json = parsed
                                except:
                                    pass
                                    
                    assert final_json is not None
                    ans_lower = final_json["answer"].lower()
                    assert any(phrase in ans_lower for phrase in ["no information", "could not find support", "not support", "unable to answer", "cannot answer"]) or final_json["confidence"] == "low"
            self._log_result("Generation", "Fallback Behavior", "PASS")
        except Exception as e:
            self._log_result("Generation", "Fallback Behavior", "FAIL", e)

    async def check_web_modes(self):
        try:
            # direct_web test
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", f"{BASE_URL}/chat", json={"query": "What is the capital of France?"}) as response:
                    final_json = None
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str:
                                try:
                                    parsed = json.loads(data_str)
                                    if "answer" in parsed and "mode" in parsed:
                                        final_json = parsed
                                except Exception as err:
                                    print("Err parsing:", err, data_str)
                    assert final_json is not None, "final_json was None"
                    assert final_json["mode"] == "direct_web", f"Expected direct_web, got {final_json['mode']}"
                    if not final_json.get("sources"):
                        assert "evidence" in final_json["answer"].lower() or "web results" in final_json["answer"].lower(), f"Unexpected answer: {final_json['answer']}"
                    else:
                        assert final_json["sources"][0]["type"] == "web"
            self._log_result("Web Search", "Direct Web Routing & Extraction", "PASS")
            
            # web_rag test
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", f"{BASE_URL}/chat", json={"query": "Compare React and Vue in depth"}) as response:
                    final_json = None
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str:
                                try:
                                    parsed = json.loads(data_str)
                                    if "answer" in parsed and "mode" in parsed:
                                        final_json = parsed
                                except Exception as err:
                                    print("Err parsing:", err, data_str)
                    assert final_json is not None, "final_json was None in web_rag test"
                    assert final_json["mode"] == "web_rag", f"Expected web_rag, got {final_json['mode']}"
                    if not final_json.get("sources"):
                        assert "evidence" in final_json["answer"].lower() or "web results" in final_json["answer"].lower(), f"Unexpected answer: {final_json['answer']}"
                    else:
                        assert final_json["sources"][0]["type"] == "web"
            self._log_result("Web Search", "Web Modes E2E", "PASS")
            
        except Exception as e:
            self._log_result("Web Search", "Web Modes E2E", "FAIL", e)

    async def check_benchmark_definition(self):
        try:
            query = "What is eigenvector?"
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", f"{BASE_URL}/chat", json={"query": query}) as response:
                    final_json = None
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str:
                                try:
                                    parsed = json.loads(data_str)
                                    if "interpretation" in parsed:
                                        final_json = parsed
                                except: pass
                    assert final_json is not None
                    # Ensure deterministic mode and no LLM planning
                    assert final_json["interpretation"]["mode"] == "direct_web"
                    assert "eigenvector" in final_json["interpretation"]["concepts"][0].lower()
            self._log_result("Benchmark", "Definition Benchmark", "PASS")
        except Exception as e:
            self._log_result("Benchmark", "Definition Benchmark", "FAIL", e)

    async def check_benchmark_comparison(self):
        try:
            query = "Compare PCA and SVD"
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", f"{BASE_URL}/chat", json={"query": query}) as response:
                    final_json = None
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str:
                                try:
                                    parsed = json.loads(data_str)
                                    if "verification" in parsed:
                                        final_json = parsed
                                except Exception as err:
                                    print("Err parsing:", err, data_str)
                    assert final_json is not None, "final_json was None in comparison benchmark"
                    assert final_json["interpretation"]["intent"] == "comparison"
            self._log_result("Benchmark", "Comparison Benchmark", "PASS")
        except Exception as e:
            self._log_result("Benchmark", "Comparison Benchmark", "FAIL", e)

    async def run_all(self):
        print("=== Starting Test Harness ===")
        await self.check_health()
        await self.check_groq_dryrun()
        await self.check_ingestion()
        self.check_unit_retrieval()
        await self.check_generation_and_streaming()
        await self.check_fallback()
        await self.check_web_modes()
        await self.check_benchmark_definition()
        await self.check_benchmark_comparison()
        
        print("\n=== Human Readable Report ===")
        print(f"Total Passed: {self.results['passed']}")
        print(f"Total Failed: {self.results['failed']}")
        print(f"Warnings: {self.results['warnings']}")
        if self.results["failing_modules"]:
            print(f"Failing Modules: {', '.join(list(self.results['failing_modules']))}")
        
        # Machine Readable Summary
        with open("test_summary.json", "w") as f:
            # We must serialize sets
            output = self.results.copy()
            output["failing_modules"] = list(output["failing_modules"])
            json.dump(output, f, indent=2)
            
        print(f"\nMachine-readable summary saved to test_summary.json")

if __name__ == "__main__":
    harness = TestHarness()
    asyncio.run(harness.run_all())
