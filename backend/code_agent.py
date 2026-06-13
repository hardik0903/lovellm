from typing import Dict, Any, AsyncGenerator
from agent_base import BaseAgent
from code_detector import CodeDetector
from code_classifier import CodeClassifier
from code_solver import CodeSolver

class CodeAgent(BaseAgent):
    def __init__(self):
        self.detector = CodeDetector()
        self.classifier = CodeClassifier()
        self.solver = CodeSolver()

    def detect(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.detector.detect(query)

    async def classify(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return await self.classifier.classify(query)

    async def solve(self, query: str, context: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        classification = await self.classify(query)
        async for event in self.solver.solve(query, classification):
            yield event
