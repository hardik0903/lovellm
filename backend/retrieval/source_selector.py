from urllib.parse import urlparse
from typing import List, Dict

class SourceSelector:
    """
    Enforces source diversity to ensure the context window isn't dominated by a single domain.
    """
    def select_diverse_sources(self, ranked_results: List[Dict], max_sources: int) -> List[Dict]:
        selected = []
        seen_domains = set()
        
        for result in ranked_results:
            if len(selected) >= max_sources:
                break
                
            url = result.get("url", "")
            try:
                domain = urlparse(url).netloc
                # Remove www.
                if domain.startswith("www."):
                    domain = domain[4:]
            except:
                domain = url
                
            # We allow a domain to appear twice at most if it's high quality, otherwise just once.
            # For simplicity in this version, we restrict to 1 per domain to enforce strict diversity.
            if domain not in seen_domains:
                selected.append(result)
                seen_domains.add(domain)
                
        # If we couldn't hit max_sources due to diversity constraint, we can backfill
        if len(selected) < max_sources and len(ranked_results) > len(selected):
            for result in ranked_results:
                if len(selected) >= max_sources:
                    break
                if result not in selected:
                    selected.append(result)
                    
        return selected
