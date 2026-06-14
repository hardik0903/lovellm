from typing import AsyncGenerator, Dict, Any
import json
from math_detector import MathDetector
from math_classifier import MathClassifier
from math_solver import MathSolver
from math_grapher import MathGrapher
from logger import logger

class MathAgent:
    def __init__(self):
        self.detector = MathDetector()
        self.classifier = MathClassifier()
        self.solver = MathSolver()
        self.grapher = MathGrapher()

    def detect(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.detector.detect(query)

    async def solve(self, query: str, context: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        logger.info(f"MathAgent taking over query: {query}")
        
        classification = await self.classifier.classify(query)
        logger.info(f"Math classification: {classification}")
        
        final_solution = None
        async for event in self.solver.solve(query, classification):
            if event["event"] == "math_final":
                try:
                    final_solution = json.loads(event["data"])
                except Exception:
                    pass
            else:
                yield event
                
        if final_solution and classification.get("requires_graph"):
            logger.info("Generating graph data...")
            graph_data = self.grapher.generate_graph_data(classification, final_solution)
            final_solution["graph_data"] = graph_data
            
        if final_solution:
            display_envelope = {
                "type": "math_solution",
                **final_solution
            }

            answer_text = final_solution.get("solution", "")

            yield {
                "event": "delta",
                "data": json.dumps({"text": answer_text})
            }
            yield {
                "event": "final",
                "data": json.dumps({
                    "mode": "math",
                    "answer": answer_text,
                    "display": display_envelope,
                    "confidence": "high",
                    "sources": []
                })
            }