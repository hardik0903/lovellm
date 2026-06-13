import abc
from typing import Dict, Any, AsyncGenerator

class BaseAgent(abc.ABC):
    """
    Abstract base class for all specialized agents in the LoveLLM ecosystem.
    Each agent must implement detection, classification, and solving logic.
    """

    @abc.abstractmethod
    def detect(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detects if this agent should handle the query.
        Returns a dictionary with at least 'is_match' (bool) and 'confidence' (float).
        """
        pass

    @abc.abstractmethod
    def classify(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Classifies the specific intent/sub-task within the agent's domain.
        Returns a dictionary representing the specific classification (e.g. intent, task_type).
        """
        pass

    @abc.abstractmethod
    async def solve(self, query: str, context: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Solves the query and yields SSE events.
        Events should follow the format: {"event": event_name, "data": json_string}
        The final event must have event="final".
        """
        pass
