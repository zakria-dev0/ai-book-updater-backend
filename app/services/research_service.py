"""
Tavily API wrapper — provides web search with source authority scoring.
Uses GPT-4o-mini to extract real author and published_date from raw_content.
Falls back to direct HTTP fetch of HTML meta tags when raw_content is unavailable.

Enhanced source quality filtering:
- Tiered authority scoring (government > academic > technical > news > commercial)
- Filters out low-quality sources (Wikipedia, blogs, social media)
- Prefers aerospace-authoritative sources (NASA, ESA, JAXA, IEEE, AIAA)
"""
import json
import asyncio
import re
from typing import List, Optional
from tavily import AsyncTavilyClient
from openai import AsyncOpenAI
import httpx
from app.core.config import settings
from app.core.logger import get_logger
from app.models.change import ResearchResult

logger = get_logger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2

# ------------------------------------------------------------------ #
# Domain authority tiers (enhanced)                                    #
# ------------------------------------------------------------------ #

GOVERNMENT_DOMAINS = [
    ".gov", ".mil",
    "nasa.gov", "esa.int", "jaxa.jp", "isro.gov.in",
    "roscosmos.ru", "csa-asc.gc.ca", "space.gc.ca",
]

ACADEMIC_DOMAINS = [
    ".edu", ".ac.",
    "arxiv.org", "ieee.org", "nature.com", "science.org",
    "springer.com", "aiaa.org", "sciencedirect.com",
    "iopscience.iop.org", "journals.sagepub.com",
]

TECHNICAL_PUBLICATION_DOMAINS = [
    "spacenews.com", "space.com", "spacepolicyonline.com",
    "spectrum.ieee.org", "aviationweek.com",
    "nasaspaceflight.com", "spaceflightnow.com",
    "arstechnica.com/science",
]

INDUSTRY_OFFICIAL_DOMAINS = [
    "spacex.com", "blueorigin.com", "rocketlabusa.com",
    "boeing.com", "lockheedmartin.com", "northropgrumman.com",
    "oneweb.net", "amazon.com/kuiper", "starlink.com",
    "relativityspace.com", "ulalaunch.com",
]

NEWS_DOMAINS = [
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "nytimes.com", "washingtonpost.com", "theguardian.com",
]

# Domains to exclude — low-quality for academic textbooks
EXCLUDED_DOMAINS = [
    "wikipedia.org", "wikimedia.org", "wiktionary.org",
    "reddit.com", "twitter.com", "x.com", "facebook.com",
    "instagram.com", "tiktok.com", "youtube.com",
    "quora.com", "medium.com", "wordpress.com",
    "blogspot.com", "tumblr.com", "pinterest.com",
]

# Preferred domains for Tavily include_domains
AUTHORITATIVE_INCLUDE = [
    "nasa.gov", "esa.int", "jaxa.jp",
    "gov", "edu",
    "ieee.org", "arxiv.org", "nature.com", "science.org",
    "aiaa.org", "springer.com",
    "spacenews.com", "space.com", "spectrum.ieee.org",
    "spacex.com", "blueorigin.com",
    "reuters.com", "apnews.com", "bbc.com",
    "who.int", "un.org",
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

    # Tier 1: Government agencies (NASA, ESA, JAXA, etc.)
    if any(d in url_lower for d in GOVERNMENT_DOMAINS):
        return "government", 0.95

    # Tier 2: Academic journals & preprints
    if any(d in url_lower for d in ACADEMIC_DOMAINS):
        return "academic", 0.88

    # Tier 3: Industry official sources
    if any(d in url_lower for d in INDUSTRY_OFFICIAL_DOMAINS):
        return "industry", 0.82

    # Tier 4: Technical publications
    if any(d in url_lower for d in TECHNICAL_PUBLICATION_DOMAINS):
        return "technical", 0.75

    # Tier 5: Major news agencies
    if any(d in url_lower for d in NEWS_DOMAINS):
        return "news", 0.65

    return "commercial", 0.40


def _is_excluded_source(url: str) -> bool:
    """Check if a URL belongs to an excluded low-quality domain."""
    url_lower = url.lower()
    return any(d in url_lower for d in EXCLUDED_DOMAINS)


async def _fetch_html_meta_tags(url: str) -> dict:
    """
    Directly fetch a URL and extract author/date from HTML meta tags.
    This is the fallback when Tavily's raw_content is empty or GPT extraction fails.
    """
    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; BookUpdater/1.0)"},
        ) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                logger.debug("HTTP fallback: %s returned status %d", url[:60], resp.status_code)
                return {"author": None, "published_date": None}

            html = resp.text[:15000]  # Only need the <head> section

            # Extract author from meta tags
            author = None
            author_patterns = [
                r'<meta\s+name=["\']author["\']\s+content=["\'](.*?)["\']',
                r'<meta\s+content=["\'](.*?)["\']\s+name=["\']author["\']',
                r'<meta\s+property=["\']article:author["\']\s+content=["\'](.*?)["\']',
                r'<meta\s+content=["\'](.*?)["\']\s+property=["\']article:author["\']',
                r'<meta\s+name=["\']citation_author["\']\s+content=["\'](.*?)["\']',
                r'<meta\s+content=["\'](.*?)["\']\s+name=["\']citation_author["\']',
                r'<meta\s+name=["\']DC\.creator["\']\s+content=["\'](.*?)["\']',
                r'<meta\s+content=["\'](.*?)["\']\s+name=["\']DC\.creator["\']',
            ]
            for pattern in author_patterns:
                match = re.search(pattern, html, re.IGNORECASE)
                if match:
                    val = match.group(1).strip()
                    if val and len(val) < 100:
                        author = val
                        break

            # Extract published_date from meta tags
            pub_date = None
            date_patterns = [
                r'<meta\s+property=["\']article:published_time["\']\s+content=["\'](.*?)["\']',
                r'<meta\s+content=["\'](.*?)["\']\s+property=["\']article:published_time["\']',
                r'<meta\s+name=["\']publication_date["\']\s+content=["\'](.*?)["\']',
                r'<meta\s+content=["\'](.*?)["\']\s+name=["\']publication_date["\']',
                r'<meta\s+name=["\']citation_publication_date["\']\s+content=["\'](.*?)["\']',
                r'<meta\s+content=["\'](.*?)["\']\s+name=["\']citation_publication_date["\']',
                r'<meta\s+name=["\']DC\.date["\']\s+content=["\'](.*?)["\']',
                r'<meta\s+content=["\'](.*?)["\']\s+name=["\']DC\.date["\']',
                r'<meta\s+property=["\']og:updated_time["\']\s+content=["\'](.*?)["\']',
                r'<meta\s+content=["\'](.*?)["\']\s+property=["\']og:updated_time["\']',
            ]
            for pattern in date_patterns:
                match = re.search(pattern, html, re.IGNORECASE)
                if match:
                    val = match.group(1).strip()
                    if val:
                        pub_date = val
                        break

            if author or pub_date:
                logger.info(
                    "HTTP meta fallback for %s: author=%s, date=%s",
                    url[:60], author, pub_date,
                )
            return {"author": author, "published_date": pub_date}

    except Exception as e:
        logger.debug("HTTP meta fallback failed for %s: %s", url[:60], e)
        return {"author": None, "published_date": None}


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

    async def _extract_metadata(self, raw_content: str, url: str) -> dict:
        """
        Two-step metadata extraction:
        1. Try GPT-4o-mini on Tavily's raw_content
        2. If GPT returns null for both, fallback to direct HTTP meta tag fetch
        """
        raw_len = len(raw_content) if raw_content else 0
        logger.debug("raw_content length for %s: %d chars", url[:60], raw_len)

        # Step 1: Try GPT extraction from raw_content
        metadata = await self._extract_metadata_with_gpt(raw_content, url)

        # Step 2: If both are null, try direct HTTP meta tag fallback
        if metadata.get("author") is None and metadata.get("published_date") is None:
            logger.debug("GPT returned null for both fields, trying HTTP meta fallback for %s", url[:60])
            fallback = await _fetch_html_meta_tags(url)
            metadata["author"] = fallback.get("author")
            metadata["published_date"] = fallback.get("published_date")
        elif metadata.get("author") is None or metadata.get("published_date") is None:
            # One field is missing — try fallback to fill the gap
            fallback = await _fetch_html_meta_tags(url)
            if metadata.get("author") is None:
                metadata["author"] = fallback.get("author")
            if metadata.get("published_date") is None:
                metadata["published_date"] = fallback.get("published_date")

        return metadata

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> List[ResearchResult]:
        """Run a Tavily search and return structured ResearchResult objects.
        Filters out low-quality sources and prioritizes authoritative ones."""
        if not self.is_configured:
            logger.warning("Tavily API key not configured — skipping search")
            return []

        # Always exclude low-quality domains
        combined_excludes = list(EXCLUDED_DOMAINS)
        if exclude_domains:
            combined_excludes.extend(exclude_domains)

        try:
            response = await self._search_with_retry(
                query=query,
                max_results=(max_results or self.max_results) + 3,  # fetch extra to account for filtering
                include_domains=include_domains or [],
                exclude_domains=combined_excludes,
                include_raw_content=True,
            )
            if response is None:
                return []

            tavily_results = response.get("results", [])

            # Pre-filter: remove excluded sources that slipped through
            filtered_results = []
            for item in tavily_results:
                url = item.get("url", "")
                if _is_excluded_source(url):
                    logger.debug("Filtered out excluded source: %s", url[:80])
                    continue
                # Skip results with no meaningful content
                content = item.get("content", "")
                if not content or len(content.strip()) < 50:
                    logger.debug("Filtered out empty/short content source: %s", url[:80])
                    continue
                filtered_results.append(item)

            # Extract metadata from all results in parallel
            metadata_tasks = [
                self._extract_metadata(
                    item.get("raw_content", ""),
                    item.get("url", ""),
                )
                for item in filtered_results
            ]
            all_metadata = await asyncio.gather(*metadata_tasks)

            results: List[ResearchResult] = []
            for item, metadata in zip(filtered_results, all_metadata):
                url = item.get("url", "")
                title = item.get("title", "")
                content = item.get("content", "")
                source_type, relevance = _score_source(url, title)

                # Use Tavily's published_date if available, otherwise extracted
                pub_date = item.get("published_date") or metadata.get("published_date")
                author = metadata.get("author")

                logger.info(
                    "Source '%s' [%s, score=%.2f]: author=%s, date=%s",
                    title[:50], source_type, relevance, author, pub_date,
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

            # Trim to requested limit
            max_count = max_results or self.max_results
            results = results[:max_count]

            logger.info("Tavily search for '%s': %d results (after filtering)", query[:60], len(results))
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
