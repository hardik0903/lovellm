import json
from typing import Dict, Any, List

class ConversationContext:
    """
    Knowledge graph of the current conversation to resolve entities and context.
    For simplicity, we store this in memory. In a real app, this might be backed by Redis or SQLite.
    """
    def __init__(self):
        self.entities: set = set()
        self.concepts: set = set()
        self.relationships: List[Dict] = []
        self.active_topic: str = ""
        self.history: List[str] = []
        self.history_objects: List[Dict[str, Any]] = []
        
    def update_context(self, query_plan: Dict[str, Any]):
        """
        Updates the graph based on the latest query plan.
        """
        concepts = query_plan.get("concepts", [])
        for concept in concepts:
            self.concepts.add(concept)
            self.active_topic = concept # most recent concept becomes active topic
            
        self.history.append(query_plan.get("original_query", ""))
        
    def resolve_references(self, query: str) -> str:
        """
        Resolves pronouns (it, they, them) to the active topic.
        """
        query_lower = query.lower()
        words = query_lower.split()
        
        # Simple heuristic: if query contains 'it', 'they', 'them', 'this', 'that'
        # and we have an active topic, append it.
        # A more advanced version would use an LLM or SpaCy coref resolution.
        pronouns = {"it", "they", "them", "this", "that", "these", "those"}
        if any(p in words for p in pronouns) and self.active_topic:
            return f"{query} (context: {self.active_topic})"
            
        return query
        
    def get_graph_summary(self) -> Dict[str, Any]:
        return {
            "entities": list(self.entities),
            "concepts": list(self.concepts),
            "active_topic": self.active_topic
        }
