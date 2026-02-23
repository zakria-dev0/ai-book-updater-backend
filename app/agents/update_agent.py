"""
Update Agent — uses GPT-4o to generate change proposals by comparing
original text with research findings.
"""
import json
import uuid
import asyncio
from typing import Dict, List
from openai import AsyncOpenAI, RateLimitError, APITimeoutError
from app.core.config import settings
from app.core.logger import get_logger
from app.models.change import (
    FactualClaim,
    ResearchResult,
    ChangeProposal,
    ChangeType,
    ConfidenceLevel,
)

logger = get_logger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2
MAX_CONCURRENT_PROPOSALS = 3  # Limit concurrent GPT calls

SYSTEM_PROMPT = """\
You are an expert editor for academic and technical books.
Given a factual claim from a book and research findings with updated information,
generate a minimal, accurate update that preserves the original writing style.

For each update, return a JSON object with:
- "old_content": the original text that needs to change
- "new_content": the updated replacement text
- "change_type": one of "data_update", "tech_update", "mission_update", "company_update", "regulatory_update"
- "confidence": "high" if multiple authoritative sources agree, "medium" if one good source, "low" if uncertain
- "reasoning": brief explanation of why this change is needed

Return a JSON object with a "proposals" array. If no update is warranted, return {"proposals": []}.
Only return valid JSON — no markdown fences, no extra text.
"""


class UpdateAgent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.GPT_MODEL
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROPOSALS)

    async def generate_proposals(
        self,
        claims: List[FactualClaim],
        research: Dict[str, List[ResearchResult]],
        document_id: str,
    ) -> List[ChangeProposal]:
        """
        For each claim that has research results, ask GPT-4o to generate
        a change proposal. Uses parallel async calls with concurrency limits.
        """
        if not settings.OPENAI_API_KEY:
            logger.warning("OpenAI API key not configured — skipping proposal generation")
            return []

        claims_with_research = [c for c in claims if c.claim_id in research]
        if not claims_with_research:
            logger.info("No claims with research results — nothing to propose")
            return []

        logger.info(
            "Generating proposals for %d claims (max %d concurrent)",
            len(claims_with_research), MAX_CONCURRENT_PROPOSALS,
        )

        # Run proposal generation in parallel with semaphore control
        async def _generate_one(idx, claim):
            async with self._semaphore:
                logger.info(
                    "Generating proposal %d/%d for claim: %s",
                    idx + 1, len(claims_with_research), claim.text[:80],
                )
                sources = research[claim.claim_id]
                return await self._generate_for_claim(claim, sources, document_id)

        tasks = [_generate_one(i, c) for i, c in enumerate(claims_with_research)]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        proposals: List[ChangeProposal] = []
        for result in task_results:
            if isinstance(result, Exception):
                logger.error("Proposal generation failed: %s", result)
                continue
            proposals.extend(result)

        logger.info(
            "Generated %d change proposals total (tokens: prompt=%d, completion=%d)",
            len(proposals), self.total_prompt_tokens, self.total_completion_tokens,
        )
        return proposals

    async def _generate_for_claim(
        self,
        claim: FactualClaim,
        sources: List[ResearchResult],
        document_id: str,
    ) -> List[ChangeProposal]:
        """Generate change proposal(s) for a single claim."""
        # Build research context
        source_texts = []
        for i, s in enumerate(sources[:5]):
            source_texts.append(
                f"Source {i+1} [{s.source_type}]: {s.source_title}\n"
                f"  URL: {s.source_url}\n"
                f"  Published: {s.published_date or 'unknown'}\n"
                f"  Content: {s.snippet[:300]}"
            )

        user_prompt = (
            f"## Original Claim\n"
            f"Text: \"{claim.text}\"\n"
            f"Type: {claim.claim_type}\n"
            f"Entities: {', '.join(claim.entities) if claim.entities else 'none'}\n"
            f"Dates mentioned: {', '.join(claim.temporal_refs) if claim.temporal_refs else 'none'}\n\n"
            f"## Research Findings\n"
            + "\n\n".join(source_texts)
        )

        try:
            response = await self._call_with_retry(SYSTEM_PROMPT, user_prompt)
            if response is None:
                return []

            # Track token usage
            if response.usage:
                self.total_prompt_tokens += response.usage.prompt_tokens
                self.total_completion_tokens += response.usage.completion_tokens
                logger.debug(
                    "Token usage: prompt=%d, completion=%d (cumulative: %d/%d)",
                    response.usage.prompt_tokens, response.usage.completion_tokens,
                    self.total_prompt_tokens, self.total_completion_tokens,
                )

            raw = response.choices[0].message.content
            data = json.loads(raw)
            proposal_list = data.get("proposals", [])

            results = []
            for p in proposal_list:
                if not p.get("old_content") or not p.get("new_content"):
                    continue

                # Map confidence string to enum
                confidence_str = p.get("confidence", "medium").lower()
                confidence = {
                    "high": ConfidenceLevel.HIGH,
                    "medium": ConfidenceLevel.MEDIUM,
                    "low": ConfidenceLevel.LOW,
                }.get(confidence_str, ConfidenceLevel.MEDIUM)

                # Map change type
                change_type_str = p.get("change_type", "data_update").lower()
                change_type = {
                    "data_update": ChangeType.DATA_UPDATE,
                    "tech_update": ChangeType.TECH_UPDATE,
                    "mission_update": ChangeType.MISSION_UPDATE,
                    "company_update": ChangeType.COMPANY_UPDATE,
                    "regulatory_update": ChangeType.REGULATORY_UPDATE,
                    "image_update": ChangeType.IMAGE_UPDATE,
                }.get(change_type_str, ChangeType.DATA_UPDATE)

                results.append(ChangeProposal(
                    change_id=f"change_{uuid.uuid4().hex[:12]}",
                    document_id=document_id,
                    claim_id=claim.claim_id,
                    old_content=p["old_content"],
                    new_content=p["new_content"],
                    change_type=change_type,
                    confidence=confidence,
                    sources=sources,
                    paragraph_idx=claim.paragraph_idx,
                    page=claim.page,
                ))

            return results

        except json.JSONDecodeError as e:
            logger.error("Failed to parse GPT proposal response: %s", e)
            return []
        except Exception as e:
            logger.error("Proposal generation failed for claim %s: %s", claim.claim_id, e)
            return []

    async def _call_with_retry(self, system: str, user_content: str):
        """Call OpenAI with exponential backoff retry on rate limits."""
        for attempt in range(MAX_RETRIES):
            try:
                return await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
            except (RateLimitError, APITimeoutError) as e:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "OpenAI rate limit/timeout (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1, MAX_RETRIES, delay, e,
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(delay)
                else:
                    logger.error("OpenAI call failed after %d retries", MAX_RETRIES)
                    return None
