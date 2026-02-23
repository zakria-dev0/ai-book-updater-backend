"""
Tavily API wrapper — provides web search with source authority scoring.
"""
import asyncio
from typing import List, Optional
from tavily import AsyncTavilyClient
from app.core.config import settings
from app.core.logger import get_logger
from app.models.change import ResearchResult

logger = get_logger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2

# Domain authority tiers
GOVERNMENT_DOMAINS = [".gov", ".mil"]
ACADEMIC_DOMAINS = [".edu", ".ac."]
AUTHORITATIVE_INCLUDE = [
    "nasa.gov", "esa.int", "gov", "edu", "who.int",
    "un.org", "ieee.org", "arxiv.org", "nature.com",
    "science.org", "springer.com",
]


def _score_source(url: str, title: str) -> tuple[str, float]:
    """Return (source_type, relevance_score) based on domain authority."""
    url_lower = url.lower()

    if any(d in url_lower for d in GOVERNMENT_DOMAINS):
        return "government", 0.95
    if any(d in url_lower for d in ACADEMIC_DOMAINS):
        return "academic", 0.85
    if any(d in url_lower for d in ["arxiv.org", "ieee.org", "nature.com", "science.org"]):
        return "academic", 0.80
    if any(d in url_lower for d in ["reuters.com", "apnews.com", "bbc.com"]):
        return "news", 0.70
    return "commercial", 0.50


class TavilyResearchService:
    def __init__(self):
        self.client = AsyncTavilyClient(api_key=settings.TAVILY_API_KEY)
        self.max_results = settings.MAX_RESEARCH_RESULTS

    @property
    def is_configured(self) -> bool:
        return bool(settings.TAVILY_API_KEY)

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> List[ResearchResult]:
        """Run a Tavily search and return structured ResearchResult objects."""
        if not self.is_configured:
            logger.warning("Tavily API key not configured — skipping search")
            return []

        try:
            response = await self._search_with_retry(
                query=query,
                max_results=max_results or self.max_results,
                include_domains=include_domains or [],
                exclude_domains=exclude_domains or [],
            )
            if response is None:
                return []

            results: List[ResearchResult] = []
            for item in response.get("results", []):
                url = item.get("url", "")
                title = item.get("title", "")
                source_type, relevance = _score_source(url, title)

                results.append(ResearchResult(
                    source_url=url,
                    source_title=title,
                    source_type=source_type,
                    published_date=item.get("published_date"),
                    snippet=item.get("content", "")[:500],
                    relevance_score=relevance,
                ))

            # Sort by relevance (highest first)
            results.sort(key=lambda r: r.relevance_score, reverse=True)
            logger.info("Tavily search for '%s': %d results", query[:60], len(results))
            return results

        except Exception as e:
            logger.error("Tavily search failed for '%s': %s", query[:60], e)
            return []

    async def search_authoritative(self, query: str) -> List[ResearchResult]:
        """Search with preference for government, academic, and org domains."""
        return await self.search(
            query=query,
            include_domains=AUTHORITATIVE_INCLUDE,
        )

    async def _search_with_retry(self, **kwargs) -> dict | None:
        """Call Tavily with exponential backoff retry."""
        for attempt in range(MAX_RETRIES):
            try:
                return await self.client.search(
                    search_depth="advanced",
                    **kwargs,
                )
            except Exception as e:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Tavily search failed (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1, MAX_RETRIES, delay, e,
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(delay)
                else:
                    logger.error("Tavily search failed after %d retries", MAX_RETRIES)
                    return None
