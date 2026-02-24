"""
Tavily API wrapper — provides web search with source authority scoring.
Uses GPT-4o-mini to extract real author and published_date from raw_content.
"""
import json
import asyncio
from typing import List, Optional
from tavily import AsyncTavilyClient
from openai import AsyncOpenAI
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

METADATA_EXTRACTION_PROMPT = """\
Extract the author name and publication date from this webpage content.

Rules:
- "author": The actual person who wrote the article (e.g. "Flora Graham", "John Smith").
  Do NOT return the publisher/website name (e.g. "Nature", "BBC") as the author.
  If no individual author name is found, return null.
- "published_date": The date when this specific article/page was published.
  Look for patterns like "Published: 18 February 2020", dates near the byline,
  or metadata-style dates at the top of the article.
  Do NOT pick up random dates mentioned in the article body.
  If no clear publication date is found, return null.

Return ONLY a JSON object: {"author": "..." or null, "published_date": "..." or null}
No extra text, no markdown fences."""


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
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.max_results = settings.MAX_RESEARCH_RESULTS

    @property
    def is_configured(self) -> bool:
        return bool(settings.TAVILY_API_KEY)

    async def _extract_metadata_with_gpt(self, raw_content: str, url: str) -> dict:
        """Use GPT-4o-mini to extract real author and published_date from raw_content."""
        if not raw_content or not settings.OPENAI_API_KEY:
            return {"author": None, "published_date": None}

        # Send first 3000 chars — author/date are always near the top
        content_excerpt = raw_content[:3000]

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": METADATA_EXTRACTION_PROMPT},
                    {"role": "user", "content": f"URL: {url}\n\nWebpage content:\n{content_excerpt}"},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            data = json.loads(raw)
            author = data.get("author")
            pub_date = data.get("published_date")
            logger.debug(
                "GPT metadata extraction for %s: author=%s, date=%s",
                url[:60], author, pub_date,
            )
            return {"author": author, "published_date": pub_date}
        except Exception as e:
            logger.warning("GPT metadata extraction failed for %s: %s", url[:60], e)
            return {"author": None, "published_date": None}

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
                include_raw_content=True,
            )
            if response is None:
                return []

            tavily_results = response.get("results", [])

            # Extract metadata from all results in parallel using GPT-4o-mini
            metadata_tasks = [
                self._extract_metadata_with_gpt(
                    item.get("raw_content", ""),
                    item.get("url", ""),
                )
                for item in tavily_results
            ]
            all_metadata = await asyncio.gather(*metadata_tasks)

            results: List[ResearchResult] = []
            for item, metadata in zip(tavily_results, all_metadata):
                url = item.get("url", "")
                title = item.get("title", "")
                content = item.get("content", "")
                source_type, relevance = _score_source(url, title)

                # Use Tavily's published_date if available, otherwise GPT-extracted
                pub_date = item.get("published_date") or metadata.get("published_date")
                author = metadata.get("author")

                logger.info(
                    "Source '%s': author=%s, published_date=%s",
                    title[:50], author, pub_date,
                )

                results.append(ResearchResult(
                    source_url=url,
                    source_title=title,
                    source_type=source_type,
                    published_date=pub_date,
                    author=author,
                    snippet=content[:500],
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
