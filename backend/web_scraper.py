import aiohttp
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional
from logger import logger

class WebScraper:
    def __init__(self):
        self.timeout = aiohttp.ClientTimeout(total=5)

    async def fetch_page(self, url: str) -> Optional[str]:
        """Asynchronously fetches the HTML of a webpage."""
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        logger.warning(f"Failed to fetch {url}, status code: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def clean_html(self, html: str) -> str:
        """Extracts and cleans text from HTML."""
        if not html:
            return ""
            
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            script_or_style.decompose()
            
        text = soup.get_text(separator=' ')
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text

    async def scrape(self, url: str) -> Dict[str, Any]:
        """Fetches and cleans a single URL."""
        logger.info(f"Scraping URL: {url}")
        html = await self.fetch_page(url)
        text = self.clean_html(html) if html else ""
        return {
            "url": url,
            "text": text,
            "success": bool(text)
        }
