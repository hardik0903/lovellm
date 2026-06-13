import json
from typing import Dict, Any, AsyncGenerator
from logger import logger
from agent_base import BaseAgent
from data_detector import DataDetector
from data_profiler import DataProfiler
from data_analyst import DataAnalyst
from chart_generator import ChartGenerator

class DataAgent(BaseAgent):
    def __init__(self):
        self.detector = DataDetector()
        self.profiler = DataProfiler()
        self.analyst = DataAnalyst()
        self.chart_gen = ChartGenerator()

    def detect(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.detector.detect(query, context)

    async def classify(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        # Classifier for data is essentially just returning the intent
        return {"intent": "analyze"}

    async def solve(self, query: str, context: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        file_path = context.get("data_file_path") if context else None
        
        yield {
            "event": "data_thinking",
            "data": json.dumps({"status": "Profiling data..."})
        }
        
        if file_path and file_path.endswith('.csv'):
            profile = self.profiler.profile_csv(file_path)
        else:
            profile = {"columns": [], "preview": [], "error": "No valid CSV file found in context"}
            
        yield {
            "event": "data_thinking",
            "data": json.dumps({"status": "Analyzing data and generating insights..."})
        }
        
        analysis = await self.analyst.analyze(query, profile)
        chart = await self.chart_gen.generate_chart(query, profile)
        
        display_data = {
            "question": query,
            "answer": analysis.get("answer", "Analysis complete."),
            "insight_type": analysis.get("insight_type", "summary"),
            "chart": chart,
            "statistics": analysis.get("statistics", {}),
            "follow_up_questions": analysis.get("follow_up_questions", [])
        }
        
        yield {
            "event": "final",
            "data": json.dumps({
                "mode": "data",
                "display": display_data
            })
        }
