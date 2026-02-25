"""
Research Agent — takes flagged outdated claims and researches them
using Tavily web search to find updated information.
Supports parallel processing with concurrency limits.

Enhanced with:
- Better query building for constellation/mega-constellation topics
- Claim-type-aware search strategies
- More search results for comprehensive synthesis
"""
import asyncio
from typing import Dict, List
from app.core.logger import get_logger
from app.models.change import FactualClaim, ResearchResult
from app.services.research_service import TavilyResearchService

logger = get_logger(__name__)

# Max concurrent Tavily searches to avoid rate limits
MAX_CONCURRENT_SEARCHES = 3


class ResearchAgent:
    def __init__(self):
        self.tavily = TavilyResearchService()
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)

    async def research_claims(
        self, claims: List[FactualClaim]
    ) -> Dict[str, List[ResearchResult]]:
        """
        Research each outdated claim via Tavily using parallel async calls.
        Returns a dict mapping claim_id -> list of ResearchResult.
        """
        if not self.tavily.is_configured:
            logger.warning("Tavily not configured — skipping research")
            return {}

        outdated = [c for c in claims if c.is_outdated]
        if not outdated:
            logger.info("No outdated claims to research")
            return {}

        logger.info("Researching %d outdated claims (max %d concurrent)", len(outdated), MAX_CONCURRENT_SEARCHES)

        # Run searches in parallel with semaphore-based concurrency limit
        tasks = [self._research_single(claim, idx, len(outdated)) for idx, claim in enumerate(outdated)]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: Dict[str, List[ResearchResult]] = {}
        for claim, result in zip(outdated, task_results):
            if isinstance(result, Exception):
                logger.error("Research failed for claim %s: %s", claim.claim_id, result)
                continue
            if result:
                results[claim.claim_id] = result

        logger.info(
            "Research complete: %d/%d claims have results",
            len(results), len(outdated),
        )
        return results

    async def _research_single(
        self, claim: FactualClaim, idx: int, total: int
    ) -> List[ResearchResult]:
        """Research a single claim with semaphore-based concurrency control."""
        async with self._semaphore:
            logger.info("Researching claim %d/%d [%s]: %s", idx + 1, total, claim.claim_type, claim.text[:80])
            query = self._build_query(claim)
            logger.info("  Search query: %s", query)
            search_results = await self.tavily.search_authoritative(query)

            if search_results:
                logger.info(
                    "  -> %d results found (top source: %s, score=%.2f)",
                    len(search_results),
                    search_results[0].source_type,
                    search_results[0].relevance_score,
                )
            else:
                logger.info("  -> no results found")

            return search_results

    @staticmethod
    def _build_query(claim: FactualClaim) -> str:
        """Build an effective search query from claim text, entities, and type.
        Enhanced with specialized query patterns for constellations and missions."""
        parts = []

        # Use entities if available for a focused query
        if claim.entities:
            parts.append(" ".join(claim.entities[:3]))

        # Claim-type-specific query strategies
        type_context = {
            "statistic": "latest statistics data",
            "date": "current status update",
            "company_info": "latest company news update",
            "mission": "space mission current status update",
            "technology": "latest technology development update",
            "policy": "current policy regulation",
            "regulation": "current regulation update",
            "citation": "latest research findings",
            "constellation": "satellite constellation current size status update",
            "historical": "current status update",
            "business_philosophy": "industry trends commercial space update",
        }
        context = type_context.get(claim.claim_type, "latest update")
        parts.append(context)

        # Special handling for constellation claims — add specific sub-queries
        claim_lower = claim.text.lower()
        if claim.claim_type == "constellation" or any(kw in claim_lower for kw in [
            "constellation", "starlink", "oneweb", "kuiper", "mega-constellation",
            "satellite network", "orbital network"
        ]):
            # Add constellation-specific context
            if "starlink" in claim_lower or "spacex" in claim_lower:
                parts.append("Starlink constellation total satellites deployed")
            elif "oneweb" in claim_lower:
                parts.append("OneWeb constellation deployment status")
            elif "kuiper" in claim_lower or "amazon" in claim_lower:
                parts.append("Amazon Kuiper constellation progress")
            elif "gps" in claim_lower:
                parts.append("GPS constellation current satellite count")
            else:
                parts.append("mega-constellation satellite count Starlink OneWeb")

        # Add temporal context if available
        if claim.temporal_refs:
            parts.append(f"after {claim.temporal_refs[0]}")

        # Fallback: use first 100 chars of claim text
        if len(parts) <= 1:
            parts.append(claim.text[:100])

        query = " ".join(parts)
        return query[:200]  # Tavily query length limit
