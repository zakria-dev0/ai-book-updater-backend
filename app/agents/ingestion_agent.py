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

Look for ANY of these types of claims:
- Statistics, numbers, percentages, measurements
- Dates, years, or time references (e.g. "in 2019", "currently", "recently", "as of")
- Company information (revenue, employees, headquarters, leadership, products)
- Mission statements, organizational goals, or strategic plans
- Technology descriptions (versions, capabilities, specifications, comparisons)
- Policy or regulation references (laws, standards, guidelines)
- Citations to research, studies, or reports
- Named entities with associated factual information (people, organizations, places)
- Descriptions of current state of any field, industry, or technology
- Any sentence with a verifiable fact that could change over time

For EACH claim you find, return a JSON object with these fields:
- "text": the exact sentence or phrase containing the claim
- "claim_type": one of "statistic", "date", "company_info", "mission", "technology", "policy", "citation", "regulation"
- "paragraph_idx": the paragraph index (provided with the text)
- "entities": list of named entities (people, orgs, places) mentioned
- "temporal_refs": list of dates or years mentioned (e.g. "2019", "March 2020")
- "is_outdated": whether the claim's TRUTH VALUE has likely changed (see rules below)

CRITICAL RULES for determining "is_outdated":

The key question is: "Would a reader be MISLED by this statement in {current_year}?"

Set is_outdated = FALSE for:
- Historical facts about past events that remain true (founding dates, release dates,
  when something happened, discontinuation dates, past milestones).
- Historical data anchored to a specific year (e.g. "As of [year]", "In [year]").
- Statements using past tense that record what occurred at a specific time.
- Broad trends or general truths that are still largely accurate, even if the exact
  situation has evolved gradually (e.g. general market trends, widely-used technologies).
- Statements that are still substantially correct, even if details have slightly shifted.

Set is_outdated = TRUE ONLY when:
- The claim makes a specific present-tense assertion that is CLEARLY WRONG today
  (e.g. someone "is" the CEO but has since stepped down, something "is the latest version"
  but newer versions have been released, something "is the newest" but has been replaced).
- The claim states a specific number, ranking, or position as current that is now
  demonstrably different (not just slightly evolved but fundamentally changed).
- The claim uses "currently", "now", "the latest", "the newest" and the underlying
  fact has been specifically superseded by a known successor or replacement.

When in doubt, set is_outdated = FALSE. Only flag claims where you are confident the
statement would actively mislead a reader today.

IMPORTANT: Return a JSON object with a "claims" key containing an array of claim objects.
Example: {{"claims": [{{"text": "...", "claim_type": "...", ...}}]}}
If no factual claims are found, return {{"claims": []}}.
Only return valid JSON — no markdown fences, no extra text.
Be thorough — even technical or academic text often contains verifiable facts.
"""

VERIFICATION_PROMPT = """\
You are a precise fact-checker. Review each claim below and determine if it is truly outdated
as of {current_year}. For each claim, decide if a reader would be MISLED by the statement today.

Rules:
- A claim is outdated ONLY if its core assertion is DEMONSTRABLY FALSE today.
- Broad generalizations that are still substantially true are NOT outdated.
- Historical facts using PAST TENSE anchored to dates are NOT outdated
  (e.g. "revenue was $X in 2019", "was founded in 1994", "was released in 2015").
- CRITICAL: Claims using PRESENT TENSE superlatives like "is the latest", "is the newest",
  "is the most popular" ARE outdated if a successor or replacement exists, EVEN IF the claim
  has a date prefix like "As of [year]". The "As of" prefix does NOT protect present-tense
  assertions — focus on whether the core verb uses "is/are" with a superlative.
- Only flag claims where a specific fact has been clearly superseded (new CEO, new version, etc.).

For each claim, return:
- "claim_id": the original claim_id
- "is_outdated": true or false (your corrected assessment)
- "reason": brief explanation of why it is or isn't outdated

Return a JSON object: {{"verified_claims": [...]}}
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

        # Verification pass: double-check outdated flags
        outdated_claims = [c for c in all_claims if c.is_outdated]
        if outdated_claims:
            logger.info(
                "Verification pass: re-checking %d claims flagged as outdated",
                len(outdated_claims),
            )
            all_claims = await self._verify_outdated_flags(all_claims)

        logger.info(
            "Content analysis complete for %s: %d total claims, %d flagged outdated",
            document_id,
            len(all_claims),
            sum(1 for c in all_claims if c.is_outdated),
        )
        return all_claims

    async def _verify_outdated_flags(
        self, claims: List[FactualClaim]
    ) -> List[FactualClaim]:
        """
        Verification pass: send all claims flagged as outdated to GPT for a
        second opinion. Corrects false positives where GPT was too aggressive.
        """
        outdated_claims = [c for c in claims if c.is_outdated]
        if not outdated_claims:
            return claims

        from datetime import datetime

        # Build the verification request
        claims_for_review = []
        for c in outdated_claims:
            claims_for_review.append({
                "claim_id": c.claim_id,
                "text": c.text,
                "claim_type": c.claim_type,
                "entities": c.entities,
                "temporal_refs": c.temporal_refs,
            })

        user_content = json.dumps({"claims_to_verify": claims_for_review}, indent=2)
        system = VERIFICATION_PROMPT.format(current_year=datetime.utcnow().year)

        try:
            response = await self._call_with_retry(system, user_content)
            if response is None:
                logger.warning("Verification call failed — keeping original flags")
                return claims

            if response.usage:
                self.total_prompt_tokens += response.usage.prompt_tokens
                self.total_completion_tokens += response.usage.completion_tokens
                logger.debug(
                    "Verification token usage: prompt=%d, completion=%d",
                    response.usage.prompt_tokens, response.usage.completion_tokens,
                )

            raw = response.choices[0].message.content
            logger.debug("Verification raw response: %s", raw[:1000])
            data = json.loads(raw)

            verified = data.get("verified_claims", [])

            # Build lookup: claim_id → verified is_outdated
            corrections = {}
            for v in verified:
                cid = v.get("claim_id")
                if cid:
                    corrections[cid] = v.get("is_outdated", True)
                    reason = v.get("reason", "")
                    logger.info(
                        "Verification [%s]: is_outdated=%s — %s",
                        cid, corrections[cid], reason,
                    )

            # Apply corrections
            corrected_count = 0
            for claim in claims:
                if claim.claim_id in corrections:
                    new_flag = corrections[claim.claim_id]
                    if new_flag != claim.is_outdated:
                        corrected_count += 1
                        logger.info(
                            "Corrected claim %s: outdated %s → %s",
                            claim.claim_id, claim.is_outdated, new_flag,
                        )
                    claim.is_outdated = new_flag

            logger.info("Verification complete: corrected %d/%d flags", corrected_count, len(outdated_claims))
            return claims

        except json.JSONDecodeError as e:
            logger.error("Failed to parse verification response: %s", e)
            return claims
        except Exception as e:
            logger.error("Verification step failed: %s", e)
            return claims

    async def _analyze_chunk(
        self, paragraphs: List[str], start_idx: int
    ) -> List[FactualClaim]:
        """Send a chunk of paragraphs to GPT-4o for claim extraction."""
        # Build numbered paragraph text
        numbered = "\n".join(
            f"[Paragraph {start_idx + i}]: {p}" for i, p in enumerate(paragraphs)
        )

        from datetime import datetime
        system = SYSTEM_PROMPT.format(
            staleness_years=settings.CONTENT_STALENESS_YEARS,
            current_year=datetime.utcnow().year,
        )

        if start_idx == 0:
            logger.info("First chunk content being sent to GPT:\n%s", numbered[:1000])

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
            logger.debug("GPT raw response (chunk %d): %s", start_idx, raw[:500])
            data = json.loads(raw)

            # GPT may wrap in {"claims": [...]} or return a bare list
            if isinstance(data, dict):
                # Try multiple possible keys GPT might use
                claims_list = (
                    data.get("claims")
                    or data.get("results")
                    or data.get("factual_claims")
                    or data.get("findings")
                    or []
                )
            elif isinstance(data, list):
                claims_list = data
            else:
                claims_list = []

            logger.debug("Parsed %d claims from chunk %d", len(claims_list), start_idx)

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
