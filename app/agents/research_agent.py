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

        Strategy: Lead with the specific subject (entities), then add a concise
        action phrase. Avoid generic filler words that dilute the query.
        """
        claim_lower = claim.text.lower()

        # ── Step 1: Extract the core subject ──────────────────────────
        # Entities are the most important part — they tell Tavily WHAT to search
        entity_str = ""
        if claim.entities:
            entity_str = " ".join(claim.entities[:3])

        # ── Step 2: Build a claim-specific query ──────────────────────
        # Instead of generic context phrases, build a query that captures
        # the specific claim's subject matter

        # Check for well-known subjects that need specific queries
        specific_query = ResearchAgent._get_specific_query(claim_lower, claim.entities)
        if specific_query:
            return specific_query[:200]

        # ── Step 3: Claim-type-aware action phrase (concise) ──────────
        type_suffix = {
            "statistic": "current data",
            "date": "current status",
            "company_info": "current status bankruptcy acquisition",
            "mission": "mission status outcome result",
            "technology": "latest version update",
            "policy": "current regulation",
            "regulation": "current regulation",
            "citation": "latest findings",
            "constellation": "constellation current status satellites",
            "historical": "current status",
            "business_philosophy": "current industry status",
        }
        suffix = type_suffix.get(claim.claim_type, "current status")

        # ── Step 4: Assemble the query ────────────────────────────────
        parts = []

        # Lead with entities (most specific)
        if entity_str:
            parts.append(entity_str)

        # Add a condensed version of the claim text (key noun phrases)
        # Extract first meaningful clause — up to 80 chars, cut at word boundary
        claim_snippet = claim.text[:80].rsplit(" ", 1)[0] if len(claim.text) > 80 else claim.text
        # Don't duplicate entity names already in the query
        if entity_str:
            for ent in claim.entities[:3]:
                claim_snippet = claim_snippet.replace(ent, "").strip()
        claim_snippet = " ".join(claim_snippet.split())  # collapse whitespace
        if claim_snippet and len(claim_snippet) > 10:
            parts.append(claim_snippet)

        # Add the action suffix
        parts.append(suffix)

        query = " ".join(parts)
        return query[:200]  # Tavily query length limit

    @staticmethod
    def _get_specific_query(claim_lower: str, entities: list) -> str | None:
        """Return a highly specific query for well-known subjects."""
        entities_lower = [e.lower() for e in entities] if entities else []

        # Mars One
        if "mars one" in claim_lower or "mars one" in " ".join(entities_lower):
            return "Mars One organization bankruptcy 2019 current status"

        # Inspiration Mars / Dennis Tito
        if "inspiration mars" in claim_lower or "dennis tito" in " ".join(entities_lower):
            return "Inspiration Mars Dennis Tito mission status cancelled"

        # Skybox Imaging
        if "skybox" in claim_lower or "skybox imaging" in " ".join(entities_lower):
            return "Skybox Imaging Google acquisition Terra Bella Planet Labs history"

        # DigitalGlobe
        if "digitalglobe" in claim_lower:
            return "DigitalGlobe Maxar acquisition current status"

        # JWST
        if "jwst" in claim_lower or "james webb" in claim_lower:
            return "James Webb Space Telescope JWST launch date deployment status"

        # Constellation-specific queries
        if "starlink" in claim_lower or "spacex" in claim_lower:
            return "Starlink constellation total satellites deployed current count"
        if "oneweb" in claim_lower:
            return "OneWeb constellation deployment status current satellites"
        if "kuiper" in claim_lower or "amazon" in claim_lower:
            return "Amazon Kuiper constellation launch progress"
        if "gps iii" in claim_lower or "gps 3" in claim_lower:
            return "GPS III satellite launch history operational status current"
        if "iridium next" in claim_lower or ("iridium" in claim_lower and ("next" in claim_lower or "replace" in claim_lower or "new" in claim_lower)):
            return "Iridium NEXT constellation launch completion status"
        if "gps" in claim_lower and "constellation" in claim_lower:
            return "GPS constellation current satellite count modernization"

        # OSIRIS-REx
        if "osiris" in claim_lower or "bennu" in claim_lower:
            return "OSIRIS-REx mission Bennu sample return status completed"

        # OCO-2
        if "oco-2" in claim_lower or "orbiting carbon observatory" in claim_lower:
            return "OCO-2 Orbiting Carbon Observatory launch A-Train status"

        # Mars 2020 / Perseverance
        if "mars 2020" in claim_lower or "perseverance" in claim_lower:
            return "Mars 2020 Perseverance rover launch landing status"

        # China space station
        if "china" in claim_lower and "space station" in claim_lower:
            return "China Tiangong space station completion status current"

        return None
