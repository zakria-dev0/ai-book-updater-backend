"""
Content Analysis Agent — uses GPT-4o to scan document text and identify
factual claims that may be outdated (dates, statistics, company info,
missions, technologies, policies, citations, constellations).

Supports:
- Focus area filtering (only detect claims matching selected categories)
- Writing style analysis (determine document's grade level, tone, complexity)
- Enhanced constellation/mega-constellation detection patterns
"""
import json
import uuid
import asyncio
from typing import List, Optional
from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIError, APIConnectionError
from app.core.config import settings
from app.core.logger import get_logger
from app.models.change import (
    FactualClaim,
    FocusArea,
    CLAIM_TYPE_TO_FOCUS_AREA,
    StyleProfile,
)

logger = get_logger(__name__)

# Retry settings
MAX_RETRIES = 5
RETRY_BASE_DELAY = 3  # seconds

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
- Satellite constellations: constellation sizes, numbers of satellites, constellation purposes,
  mega-constellation references, orbital network descriptions. Pay special attention to statements
  about constellation sizes in "dozens" or "hundreds" — modern mega-constellations have thousands.

For EACH claim you find, return a JSON object with these fields:
- "text": the exact sentence or phrase containing the claim
- "claim_type": one of "statistic", "date", "company_info", "mission", "technology", "policy", "citation", "regulation", "constellation", "historical", "business_philosophy"
- "paragraph_idx": the paragraph index (provided with the text)
- "entities": list of named entities (people, orgs, places) mentioned
- "temporal_refs": list of dates or years mentioned (e.g. "2019", "March 2020")
- "is_outdated": whether the claim's TRUTH VALUE has likely changed (see rules below)

CLAIM TYPE GUIDANCE:
- "constellation": ANY mention of satellite constellations, constellation sizes, GPS/GLONASS/Galileo
  constellations, Starlink, OneWeb, Kuiper, mega-constellations, orbital networks, satellite counts
- "mission": Space missions, launches, mission names, rocket flights, exploration programs
- "technology": Hardware, software, systems, tech versions, spacecraft capabilities
- "statistic": Numerical data, percentages, rankings, metrics, budgets
- "company_info": Company details, leadership, ownership, mergers, acquisitions
- "business_philosophy": Industry trends, commercial space evolution, market dynamics
- "historical": Historical dates, events, milestones, founding dates
- "date": Time-sensitive references using "currently", "now", "as of", "recently"
- "policy"/"regulation": Laws, standards, guidelines, treaties, regulatory frameworks

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
- For constellations: statements about constellation sizes that are dramatically outdated
  (e.g. "constellations typically consist of 24-48 satellites" — now mega-constellations
  have thousands). Statements about constellation purposes limited to navigation only,
  when modern constellations serve broadband, IoT, imaging, etc.

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
- For constellation claims: statements about small constellation sizes (24-48 satellites as "typical")
  ARE outdated given modern mega-constellations (thousands of satellites).

For each claim, return:
- "claim_id": the original claim_id
- "is_outdated": true or false (your corrected assessment)
- "reason": brief explanation of why it is or isn't outdated

Return a JSON object: {{"verified_claims": [...]}}
Only return valid JSON — no markdown fences, no extra text.
"""

STYLE_ANALYSIS_PROMPT = """\
You are an expert linguistic analyst. Analyze the writing style of the following document excerpt
and produce a style profile.

Determine each of the following:
- "grade_level": one of "college_freshman", "college_junior", "college_senior", "graduate"
- "technical_depth": one of "introductory", "intermediate", "advanced"
- "tone": one of "conversational", "formal_academic", "authoritative"
- "sentence_complexity": one of "simple", "moderate", "complex"
- "terminology_level": one of "basic", "technical", "highly_technical"
- "avg_sentence_length": estimated average number of words per sentence (integer)
- "passive_voice_usage": one of "rare", "moderate", "frequent"

Consider:
- Vocabulary sophistication and domain-specific terminology
- Sentence structure (simple vs. compound-complex)
- Use of passive voice vs. active voice
- Formality level (contractions, colloquialisms vs. academic register)
- Technical jargon density
- Reference style (inline citations, footnotes, etc.)

Return ONLY a JSON object with these fields. No extra text, no markdown fences.
Example: {{"grade_level": "college_senior", "technical_depth": "advanced", "tone": "formal_academic", "sentence_complexity": "complex", "terminology_level": "highly_technical", "avg_sentence_length": 28, "passive_voice_usage": "moderate"}}
"""

CHUNK_SIZE = 8  # paragraphs per GPT call


class ContentAnalysisAgent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.GPT_MODEL
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    # ------------------------------------------------------------------ #
    # Writing Style Analysis                                               #
    # ------------------------------------------------------------------ #

    async def analyze_style(self, text_content: str) -> StyleProfile:
        """
        Analyze the first 3000 characters of a document to determine
        its writing style profile. Run once per document.
        """
        if not settings.OPENAI_API_KEY:
            logger.warning("OpenAI API key not configured — returning default style profile")
            return StyleProfile()

        excerpt = text_content[:3000]

        try:
            response = await self._call_with_retry(STYLE_ANALYSIS_PROMPT, excerpt)
            if response is None:
                logger.warning("Style analysis call failed — returning default profile")
                return StyleProfile()

            if response.usage:
                self.total_prompt_tokens += response.usage.prompt_tokens
                self.total_completion_tokens += response.usage.completion_tokens

            raw = response.choices[0].message.content
            logger.info("Style analysis raw response: %s", raw[:500])
            data = json.loads(raw)

            profile = StyleProfile(
                grade_level=data.get("grade_level", "college_senior"),
                technical_depth=data.get("technical_depth", "intermediate"),
                tone=data.get("tone", "formal_academic"),
                sentence_complexity=data.get("sentence_complexity", "moderate"),
                terminology_level=data.get("terminology_level", "technical"),
                avg_sentence_length=data.get("avg_sentence_length", 25),
                passive_voice_usage=data.get("passive_voice_usage", "moderate"),
            )
            logger.info(
                "Style profile: grade=%s, depth=%s, tone=%s, complexity=%s, terminology=%s",
                profile.grade_level, profile.technical_depth, profile.tone,
                profile.sentence_complexity, profile.terminology_level,
            )
            return profile

        except json.JSONDecodeError as e:
            logger.error("Failed to parse style analysis response: %s", e)
            return StyleProfile()
        except Exception as e:
            logger.error("Style analysis failed: %s", e)
            return StyleProfile()

    # ------------------------------------------------------------------ #
    # Claim Analysis with Focus Area Support                               #
    # ------------------------------------------------------------------ #

    async def analyze_document(
        self,
        document_id: str,
        text_content: str,
        paragraphs: List[str] | None = None,
        focus_areas: Optional[List[str]] = None,
    ) -> List[FactualClaim]:
        """
        Analyze document text and return a list of FactualClaim objects.
        If focus_areas is provided (not ["all"]), claims are filtered to only
        those matching the specified categories.
        """
        if not settings.OPENAI_API_KEY:
            logger.warning("OpenAI API key not configured — skipping content analysis")
            return []

        if paragraphs is None:
            paragraphs = [p for p in text_content.split("\n") if p.strip()]

        # Resolve focus areas
        active_focus = None
        if focus_areas and "all" not in focus_areas:
            active_focus = set(focus_areas)
            logger.info("Focus area filtering active: %s", active_focus)

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

        # Assign focus_area to each claim based on claim_type
        for claim in all_claims:
            mapped = CLAIM_TYPE_TO_FOCUS_AREA.get(claim.claim_type)
            claim.focus_area = mapped.value if mapped else "technology"

        # Filter by focus areas if specified
        if active_focus:
            before_count = len(all_claims)
            all_claims = [c for c in all_claims if c.focus_area in active_focus]
            logger.info(
                "Focus area filter: %d → %d claims (kept areas: %s)",
                before_count, len(all_claims), active_focus,
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
        """Call OpenAI with exponential backoff retry on transient errors."""
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
                    timeout=120.0,
                )
            except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as e:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                error_type = type(e).__name__
                status_code = getattr(e, "status_code", "N/A")
                logger.warning(
                    "OpenAI %s (status=%s, attempt %d/%d), retrying in %ds: %s",
                    error_type, status_code, attempt + 1, MAX_RETRIES, delay, str(e)[:200],
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "OpenAI call failed after %d retries. Last error: %s (status=%s)",
                        MAX_RETRIES, error_type, status_code,
                    )
                    return None
