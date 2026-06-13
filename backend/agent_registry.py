from typing import Dict, Type
from agent_base import BaseAgent

from math_agent import MathAgent
from knowledge_agent import KnowledgeAgent
from code_agent import CodeAgent
from writing_agent import WritingAgent
from document_agent import DocumentAgent
from research_agent import ResearchAgent
from data_agent import DataAgent

class AgentRegistry:
    """
    Central registry of all specialized agents available in the system.
    """
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self.register("math", MathAgent())
        self.register("knowledge", KnowledgeAgent())
        self.register("code", CodeAgent())
        self.register("writing", WritingAgent())
        self.register("document", DocumentAgent())
        self.register("research", ResearchAgent())
        self.register("data", DataAgent())

    def register(self, name: str, agent: BaseAgent):
        self._agents[name] = agent

    def get_agent(self, name: str) -> BaseAgent:
        return self._agents.get(name)

    def get_all_agents(self) -> Dict[str, BaseAgent]:
        return self._agents
