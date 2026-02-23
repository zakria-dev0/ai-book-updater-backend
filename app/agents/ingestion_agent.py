"""
Content Analysis Agent — uses GPT-4o to scan document text and identify
factual claims that may be outdated (dates, statistics, company info,
missions, technologies, policies, citations).
"""
import json
import uuid
import asyncio
from typing import List
from openai import AsyncOpenAI, RateLimitError, APITimeoutError
from app.core.config import settings
from app.core.logger import get_logger
from app.models.change import FactualClaim

logger = get_logger(__name__)

# Retry settings
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds

SYSTEM_PROMPT = """\
You are an expert fact-checker for academic and technical books.
Analyze the text below and identify **factual claims** that could become outdated.

For EACH claim you find, return a JSON object with these fields:
- "text": the exact sentence or phrase containing the claim
- "claim_type": one of "statistic", "date", "company_info", "mission", "technology", "policy", "citation", "regulation"
- "paragraph_idx": the paragraph index (provided with the text)
- "entities": list of named entities (people, orgs, places) mentioned
- "temporal_refs": list of dates or years mentioned (e.g. "2019", "March 2020")
- "is_outdated": true if the claim references information older than {staleness_years} years from today (2026), or if the claim uses phrases like "currently", "recently", "as of"

Return a JSON array of claim objects. If no factual claims are found, return [].
Only return valid JSON — no markdown fences, no extra text.
"""

CHUNK_SIZE = 8  # paragraphs per GPT call


class ContentAnalysisAgent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.GPT_MODEL
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    async def analyze_document(
        self,
        document_id: str,
        text_content: str,
        paragraphs: List[str] | None = None,
    ) -> List[FactualClaim]:
        """
        Analyze document text and return a list of FactualClaim objects.
        If paragraphs list is not provided, splits text_content by newlines.
        """
        if not settings.OPENAI_API_KEY:
            logger.warning("OpenAI API key not configured — skipping content analysis")
            return []

        if paragraphs is None:
            paragraphs = [p for p in text_content.split("\n") if p.strip()]

        all_claims: List[FactualClaim] = []

        # Process paragraphs in chunks
        for chunk_start in range(0, len(paragraphs), CHUNK_SIZE):
            chunk = paragraphs[chunk_start : chunk_start + CHUNK_SIZE]
            chunk_claims = await self._analyze_chunk(chunk, chunk_start)
            all_claims.extend(chunk_claims)
            logger.info(
                "Document %s — chunk %d–%d: found %d claims",
                document_id,
                chunk_start,
                chunk_start + len(chunk) - 1,
                len(chunk_claims),
            )

        logger.info(
            "Content analysis complete for %s: %d total claims, %d flagged outdated",
            document_id,
            len(all_claims),
            sum(1 for c in all_claims if c.is_outdated),
        )
        return all_claims

    async def _analyze_chunk(
        self, paragraphs: List[str], start_idx: int
    ) -> List[FactualClaim]:
        """Send a chunk of paragraphs to GPT-4o for claim extraction."""
        # Build numbered paragraph text
        numbered = "\n".join(
            f"[Paragraph {start_idx + i}]: {p}" for i, p in enumerate(paragraphs)
        )

        system = SYSTEM_PROMPT.format(
            staleness_years=settings.CONTENT_STALENESS_YEARS
        )

        try:
            response = await self._call_with_retry(system, numbered)
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

            # GPT may wrap in {"claims": [...]} or return a bare list
            if isinstance(data, dict):
                claims_list = data.get("claims", data.get("results", []))
            elif isinstance(data, list):
                claims_list = data
            else:
                claims_list = []

            return [
                FactualClaim(
                    claim_id=f"claim_{uuid.uuid4().hex[:12]}",
                    text=c.get("text", ""),
                    claim_type=c.get("claim_type", "unknown"),
                    paragraph_idx=c.get("paragraph_idx", start_idx),
                    entities=c.get("entities", []),
                    temporal_refs=c.get("temporal_refs", []),
                    is_outdated=c.get("is_outdated", False),
                )
                for c in claims_list
                if c.get("text")
            ]

        except json.JSONDecodeError as e:
            logger.error("Failed to parse GPT response as JSON: %s", e)
            return []
        except Exception as e:
            logger.error("Content analysis chunk failed: %s", e)
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
                    temperature=0.1,
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
