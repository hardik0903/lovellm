from typing import List, Set, Dict

class RetrievalMemory:
    """
    State manager for a single conversation turn's retrieval process.
    Prevents infinite search loops and revisiting rejected evidence.
    """
    def __init__(self):
        self.searched_queries: List[str] = []
        self.rejected_urls: Set[str] = set()
        self.accepted_sources: List[Dict] = []
        self.search_pass_count: int = 0
        
    def add_query(self, query: str):
        self.searched_queries.append(query)
        self.search_pass_count += 1
        
    def is_query_searched(self, query: str) -> bool:
        return query in self.searched_queries
        
    def reject_source(self, url: str):
        self.rejected_urls.add(url)
        
    def is_source_rejected(self, url: str) -> bool:
        return url in self.rejected_urls
        
    def accept_source(self, source: Dict):
        self.accepted_sources.append(source)
        
    def get_accepted_sources(self) -> List[Dict]:
        return self.accepted_sources
