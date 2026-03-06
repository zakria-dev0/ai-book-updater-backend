"""
Tavily API wrapper — provides web search with source authority scoring.

Lightweight metadata extraction:
- Uses Tavily's built-in published_date when available
- Falls back to regex extraction from raw_content (fast, no API calls)
- GPT metadata extraction disabled by default for speed (saves ~10s per search)

Enhanced source quality filtering:
- Tiered authority scoring (government > academic > technical > news > commercial)
- Filters out low-quality sources (Wikipedia, blogs, social media)
- Prefers aerospace-authoritative sources (NASA, ESA, JAXA, IEEE, AIAA)
"""
import asyncio
import re
from typing import List, Optional
from tavily import AsyncTavilyClient
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


def _normalize_url(url: str) -> str:
    """Normalize a URL for deduplication: strip trailing slash, fragments, and lowercase."""
    url = url.split("#")[0]        # remove fragment
    url = url.rstrip("/")          # remove trailing slash
    return url.lower()


def _extract_query_key_terms(query: str) -> list[str]:
    """
    Extract meaningful key terms from a search query for content relevance matching.
    Returns lowercased terms that are specific enough to check against source content.
    """
    stop_words = {
        "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are",
        "was", "were", "be", "been", "has", "have", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can", "shall",
        "not", "no", "but", "if", "then", "than", "so", "as", "at", "by",
        "on", "up", "out", "with", "from", "into", "its", "it", "this", "that",
        "current", "status", "update", "latest", "news",
        "after", "before", "since", "about",
    }

    terms = []
    for word in query.lower().split():
        cleaned = word.strip(".,;:()\"'")
        if len(cleaned) >= 3 and cleaned not in stop_words:
            terms.append(cleaned)

    return terms


def _compute_content_relevance(
    query_terms: list[str], title: str, content: str,
) -> float:
    """
    Compute how relevant a source's content is to the query terms.
    Returns a score from 0.0 (no match) to 1.0 (strong match).
    """
    if not query_terms:
        return 0.5  # Can't assess, neutral score

    combined = f"{title} {content}".lower()
    matched = sum(1 for term in query_terms if term in combined)
    ratio = matched / len(query_terms)

    # Boost if multiple distinct key terms appear (not just one generic match)
    if matched >= 3:
        ratio = min(1.0, ratio * 1.2)

    return round(ratio, 3)


def _extract_date_from_content(raw_content: str) -> str | None:
    """
    Fast regex-based date extraction from raw_content.
    Looks for common date patterns near the top of the content (byline area).
    No API calls — pure string matching.
    """
    if not raw_content:
        return None

    # Only check the first 1500 chars (byline/header area)
    header = raw_content[:1500]

    # Common date patterns
    patterns = [
        # "Published: February 18, 2020" or "Updated: March 5, 2024"
        r'(?:Published|Updated|Posted|Date)[:\s]+(\w+ \d{1,2},?\s*\d{4})',
        # "2024-03-15" ISO format
        r'(\d{4}-\d{2}-\d{2})',
        # "March 15, 2024" standalone
        r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4})',
        # "15 March 2024" UK format
        r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
    ]

    for pattern in patterns:
        match = re.search(pattern, header, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def _extract_author_from_content(raw_content: str) -> str | None:
    """
    Fast regex-based author extraction from raw_content.
    Looks for common byline patterns. No API calls.
    """
    if not raw_content:
        return None

    # Only check the first 1500 chars
    header = raw_content[:1500]

    patterns = [
        # "By John Smith" or "by Jane Doe"
        r'[Bb]y\s+([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        # "Author: John Smith"
        r'[Aa]uthor[:\s]+([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
    ]

    for pattern in patterns:
        match = re.search(pattern, header)
        if match:
            val = match.group(1).strip()
            if len(val) < 60:  # Sanity check
                return val

    return None


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
        """Run a Tavily search and return structured ResearchResult objects.
        Filters out low-quality sources and prioritizes authoritative ones.
        Uses fast regex-based metadata extraction instead of GPT API calls."""
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
                include_raw_content=False,
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

            # Deduplicate by normalized URL (trailing slash, fragment, case)
            seen_urls: set[str] = set()
            deduped_results = []
            for item in filtered_results:
                norm = _normalize_url(item.get("url", ""))
                if norm in seen_urls:
                    logger.debug("Filtered duplicate URL: %s", item.get("url", "")[:80])
                    continue
                seen_urls.add(norm)
                deduped_results.append(item)
            filtered_results = deduped_results

            # Extract key search terms from the query for content relevance check
            query_terms = _extract_query_key_terms(query)

            results: List[ResearchResult] = []
            for item in filtered_results:
                url = item.get("url", "")
                title = item.get("title", "")
                content = item.get("content", "")
                raw_content = item.get("raw_content", "")
                source_type, base_relevance = _score_source(url, title)

                # ── Content relevance check ───────────────────────────
                content_relevance = _compute_content_relevance(
                    query_terms, title, content,
                )
                # Blend: domain authority (40%) + content relevance (60%)
                relevance = (base_relevance * 0.4) + (content_relevance * 0.6)

                # Hard penalty: if snippet has near-zero content match,
                # cap the score regardless of domain authority
                if content_relevance < 0.15:
                    relevance = min(relevance, 0.35)
                    logger.info(
                        "Hard-penalized low-relevance source (content=%.2f): '%s'",
                        content_relevance, title[:60],
                    )

                # ── Fast metadata extraction ──
                # Use Tavily's published_date first, then regex fallback on raw_content if available
                pub_date = item.get("published_date")
                if not pub_date and raw_content:
                    pub_date = _extract_date_from_content(raw_content)

                author = _extract_author_from_content(raw_content) if raw_content else None

                logger.info(
                    "Source '%s' [%s, score=%.2f (domain=%.2f, content=%.2f)]: author=%s, date=%s",
                    title[:50], source_type, relevance, base_relevance,
                    content_relevance, author, pub_date,
                )

                results.append(ResearchResult(
                    source_url=url,
                    source_title=title,
                    source_type=source_type,
                    published_date=pub_date,
                    author=author,
                    snippet=content[:500],
                    relevance_score=round(relevance, 3),
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
                    search_depth="basic",
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
