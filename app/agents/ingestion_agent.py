"""
Content Analysis Agent — uses GPT-4o to scan document text and identify
factual claims that may be outdated.

Supports:
- Document age estimation (analyzes references to determine publication era)
- Age-aware claim detection (adjusts sensitivity based on document age)
- Focus area filtering (only detect claims matching selected categories)
- Writing style analysis (determine document's grade level, tone, complexity)
- Comprehensive detection: explicit facts, state-of-field descriptions,
  predictions, reference staleness, organizational changes, technology landscape
"""
import json
import re
import uuid
import asyncio
from typing import List, Optional, Dict
from datetime import datetime
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
RETRY_BASE_DELAY = 1  # seconds (shorter since we throttle proactively)

# Token-rate-aware throttling for OpenAI TPM limits
# With 30K TPM, we budget tokens across a sliding window
import time as _time

TPM_LIMIT = 30000
_token_log: list[tuple[float, int]] = []  # (timestamp, tokens_used)
_throttle_lock: asyncio.Lock | None = None


def _get_throttle_lock() -> asyncio.Lock:
    global _throttle_lock
    if _throttle_lock is None:
        _throttle_lock = asyncio.Lock()
    return _throttle_lock


async def _throttle_for_tpm(estimated_tokens: int):
    """Wait if necessary to stay under the TPM rate limit."""
    lock = _get_throttle_lock()
    async with lock:
        now = _time.monotonic()
        # Purge entries older than 60 seconds
        cutoff = now - 60.0
        while _token_log and _token_log[0][0] < cutoff:
            _token_log.pop(0)
        # Sum tokens used in the last 60 seconds
        used = sum(t for _, t in _token_log)
        if used + estimated_tokens > TPM_LIMIT * 0.85:  # 85% safety margin
            # Wait until enough budget frees up
            if _token_log:
                oldest_time = _token_log[0][0]
                wait = 60.0 - (now - oldest_time) + 0.5
                if wait > 0:
                    logger.info("TPM throttle: used %d/%d, waiting %.1fs", used, TPM_LIMIT, wait)
                    await asyncio.sleep(wait)
        # Record this request
        _token_log.append((_time.monotonic(), estimated_tokens))


# ------------------------------------------------------------------ #
# Document Age Estimation Prompt                                       #
# ------------------------------------------------------------------ #

DOCUMENT_AGE_PROMPT = """\
You are an expert publication date analyst for technical books.

Examine this text and estimate when it was written or last updated by analyzing:
1. Most recent reference/citation years
2. Technology or systems described as "new" or "emerging"
3. Forward-looking language about events that may have already occurred
4. Named programs, organizations, or products and their known active periods

Return ONLY a JSON object:
{{
  "estimated_publication_year": <integer>,
  "newest_reference_year": <integer or null>,
  "confidence": "high" | "medium" | "low",
  "document_age_years": <{current_year} minus estimated_publication_year>
}}
No markdown fences, no extra text.
"""


# ------------------------------------------------------------------ #
# Claim Detection Prompt (age-aware, multi-category)                   #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """\
You are an expert technical textbook update assistant.
Systematically identify EVERY statement that may need updating for a modern reader.

DOCUMENT CONTEXT:
- Estimated publication: ~{estimated_pub_year} ({document_age} years old)
- Current year: {current_year}

=======================================
DETECTION CATEGORIES -- check ALL of these
=======================================

A. EXPLICIT FACTS: Statistics, dates, named technologies, company/org info,
   policy/regulation references, program or mission names with status claims.

B. STATE-OF-FIELD DESCRIPTIONS: Any description of a field or technology as
   "emerging", "in its infancy", "nascent", "beginning to", "not yet mature",
   "only recently", "early stages". In a {document_age}-year-old document,
   these fields are likely now well-established.

C. PREDICTIONS & FORWARD-LOOKING CLAIMS: "will", "plans to", "is expected to",
   "we anticipate", "should become". If the timeframe has passed, it is outdated
   regardless of outcome -- the reader needs what ACTUALLY happened.

D. REFERENCE STALENESS: Citations older than 15 years in fast-evolving fields.
   Standards, guidelines, or reports that may have newer editions.

E. ORGANIZATIONAL CHANGES: Orgs that may have merged, dissolved, renamed, or
   been restructured. Programs that were cancelled, completed, or renamed.

F. TECHNOLOGY LANDSCAPE: Systems described as "new"/"advanced"/"cutting-edge"
   that may now be legacy. Cost figures, capability limits, or comparisons
   that may have fundamentally shifted.

G. IMPLICIT TIME-SENSITIVITY: Present-tense descriptions of dynamic situations.
   "There are now...", "Currently...", "Today's..." in a {document_age}-year-old text.

=======================================
STALENESS RULES
=======================================

Core question: "Would a student reading this in {current_year} form an INACCURATE
understanding of the current state of affairs?"

is_outdated = TRUE when:
- Present-tense claim about a dynamic situation in a document 10+ years old
- Prediction whose timeframe has passed
- Field described as "emerging"/"nascent" in a document 15+ years old
- Organization/company has fundamentally changed
- Cost, count, or market figures are likely >25% different
- Technology described as limited/experimental that is now mature
- Future-tense claims ("will launch", "plans to") where the date has passed
- Companies/orgs referenced in present tense that no longer exist or were acquired

is_outdated = FALSE only when:
- Permanent historical fact in past tense ("Apollo 11 landed on July 20, 1969")
- Physical law, mathematical formula, or stable definition
- Trivially minor changes that would not mislead a reader

AGE SENSITIVITY: This document is ~{document_age} years old. For 10+ year old documents,
MOST present-tense claims about technology, industry, or organizations ARE likely outdated.
Finding fewer than 10 claims in a full chapter likely means under-detection.

=======================================
OUTPUT FORMAT
=======================================

Return JSON:
{{"claims": [
    {{
      "text": "exact sentence or phrase from the text",
      "claim_type": "statistic"|"date"|"company_info"|"mission"|"technology"|
                   "policy"|"citation"|"regulation"|"constellation"|"historical"|
                   "business_philosophy"|"landscape"|"prediction"|"reference"|"methodology",
      "paragraph_idx": <integer>,
      "entities": ["named", "entities"],
      "temporal_refs": ["years", "or", "dates"],
      "is_outdated": true or false
    }}
]}}

Only valid JSON -- no markdown, no extra text. Be thorough.
"""


# ------------------------------------------------------------------ #
# Verification Prompt (age-aware, less conservative)                   #
# ------------------------------------------------------------------ #

VERIFICATION_PROMPT = """\
You are a senior technical editor verifying outdated-claim flags in a textbook.

CONTEXT: Document is ~{document_age} years old (est. published {estimated_pub_year}).
Current year: {current_year}.

RULES:

KEEP is_outdated = TRUE when:
- Specific factual claim about a dynamic situation (e.g., "X costs $Y", "there are N satellites")
- Prediction with a passed timeframe ("will launch by 2010", "plans to")
- Field described as "emerging"/"nascent" in a 15+ year old document
- Specific cost/market/count figures from {estimated_pub_year} or earlier
- Named organization or program that likely changed significantly
- Future-tense claims where the date has clearly passed
- Companies/orgs referenced in present tense that may no longer exist

FLIP to FALSE when:
- Permanent historical fact that cannot change
- Physical law or mathematical principle
- GENERAL TRUISMS that are STILL TRUE regardless of age. Examples:
  * "Methods may change as a result of evolving technology" → still true, not outdated
  * "We must find a way to reduce costs" → still true, not outdated
  * "The way we do X is continually evolving" → still true, not outdated
  * "End users receive and use the products" → definition, not outdated
  * "Operators control and maintain the space and ground assets" → role definition
- Descriptions of ROLES, PROCESSES, or GENERAL PRINCIPLES that haven't fundamentally changed
- Statements about the NATURE of engineering/design that are timeless
- You are CERTAIN the situation has NOT materially changed

KEY DISTINCTION: A claim that describes HOW THINGS WORK IN GENERAL (process, role, principle)
is different from a claim about A SPECIFIC STATE OF AFFAIRS (count, cost, status, capability).
General principles rarely become outdated. Specific states of affairs in a {document_age}-year-old
document usually ARE outdated.

For a {document_age}-year-old document, when in doubt about SPECIFIC FACTS -> KEEP the outdated flag.
When in doubt about GENERAL PRINCIPLES -> FLIP to FALSE.

For each claim return:
- "claim_id": original claim_id
- "is_outdated": true or false
- "reason": brief explanation

Return JSON: {{"verified_claims": [...]}}
No markdown fences, no extra text.
"""


# ------------------------------------------------------------------ #
# Style Analysis Prompt (unchanged)                                    #
# ------------------------------------------------------------------ #

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

CHUNK_SIZE = 12  # paragraphs per GPT call (smaller = more thorough per chunk)
MAX_CONCURRENT_CHUNKS = 3  # parallel GPT calls — limited by 30K TPM on OpenAI


class ContentAnalysisAgent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.GPT_MODEL
        # Use the full gpt-4o model for claim extraction — mini misses subtle staleness
        self.analysis_model = settings.GPT_MODEL
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._chunk_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)
        # Document age context — set by estimate_document_age() before analysis
        self._estimated_pub_year: int | None = None
        self._document_age: int | None = None

    # ------------------------------------------------------------------ #
    # Document Age Estimation                                              #
    # ------------------------------------------------------------------ #

    async def estimate_document_age(self, text_content: str) -> Dict:
        """
        Estimate when the document was published by analyzing references,
        language patterns, and technology descriptions.
        Returns dict with estimated_publication_year and document_age_years.
        """
        if not settings.OPENAI_API_KEY:
            logger.warning("OpenAI API key not configured — skipping age estimation")
            return self._fallback_age_estimation(text_content)

        # Send first 5000 chars + last 3000 chars (references are often at the end)
        excerpt = text_content[:5000]
        if len(text_content) > 8000:
            excerpt += "\n\n--- END OF DOCUMENT ---\n\n" + text_content[-3000:]

        current_year = datetime.utcnow().year
        system = DOCUMENT_AGE_PROMPT.format(current_year=current_year)

        try:
            response = await self._call_with_retry(system, excerpt)
            if response is None:
                logger.warning("Age estimation call failed — using fallback")
                return self._fallback_age_estimation(text_content)

            if response.usage:
                self.total_prompt_tokens += response.usage.prompt_tokens
                self.total_completion_tokens += response.usage.completion_tokens

            raw = response.choices[0].message.content
            logger.info("Document age estimation raw: %s", raw[:500])
            data = json.loads(raw)

            est_year = data.get("estimated_publication_year", current_year - 5)
            doc_age = current_year - est_year

            # Sanity check: age should be between 0 and 60
            if doc_age < 0:
                doc_age = 0
                est_year = current_year
            elif doc_age > 60:
                doc_age = 60
                est_year = current_year - 60

            self._estimated_pub_year = est_year
            self._document_age = doc_age

            logger.info(
                "Document age estimation: published ~%d (%d years old), confidence=%s",
                est_year, doc_age, data.get("confidence", "unknown"),
            )
            return {
                "estimated_publication_year": est_year,
                "document_age_years": doc_age,
                "confidence": data.get("confidence", "medium"),
                "newest_reference_year": data.get("newest_reference_year"),
            }

        except json.JSONDecodeError as e:
            logger.error("Failed to parse age estimation response: %s", e)
            return self._fallback_age_estimation(text_content)
        except Exception as e:
            logger.error("Age estimation failed: %s", e)
            return self._fallback_age_estimation(text_content)

    def _fallback_age_estimation(self, text_content: str) -> Dict:
        """
        Fallback: extract years from text using regex and estimate from the
        most recent reference year found.
        """
        current_year = datetime.utcnow().year
        # Find all 4-digit years in the text
        years = [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', text_content)]
        # Filter to reasonable range
        years = [y for y in years if 1950 <= y <= current_year]

        if years:
            newest = max(years)
            est_year = min(newest + 2, current_year)  # Publication usually ~2 years after newest ref
        else:
            est_year = current_year - 10  # Conservative default

        doc_age = current_year - est_year
        self._estimated_pub_year = est_year
        self._document_age = doc_age

        logger.info(
            "Fallback age estimation: newest ref year=%s, est. published ~%d (%d years old)",
            max(years) if years else "none", est_year, doc_age,
        )
        return {
            "estimated_publication_year": est_year,
            "document_age_years": doc_age,
            "confidence": "low",
            "newest_reference_year": max(years) if years else None,
        }

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

        # Ensure document age is estimated
        if self._estimated_pub_year is None:
            await self.estimate_document_age(text_content)

        # Resolve focus areas
        active_focus = None
        if focus_areas and "all" not in focus_areas:
            active_focus = set(focus_areas)
            logger.info("Focus area filtering active: %s", active_focus)

        all_claims: List[FactualClaim] = []

        # Process paragraphs in chunks — parallel with semaphore
        chunk_ranges = list(range(0, len(paragraphs), CHUNK_SIZE))
        total_chunks = len(chunk_ranges)
        logger.info(
            "Document %s: %d paragraphs -> %d chunks (max %d concurrent), doc age=%d years",
            document_id, len(paragraphs), total_chunks, MAX_CONCURRENT_CHUNKS,
            self._document_age or 0,
        )

        async def _process_chunk(chunk_start: int, chunk_idx: int) -> List[FactualClaim]:
            async with self._chunk_semaphore:
                chunk = paragraphs[chunk_start : chunk_start + CHUNK_SIZE]
                chunk_claims = await self._analyze_chunk(chunk, chunk_start)
                logger.info(
                    "Document %s -- chunk %d/%d (paras %d-%d): found %d claims",
                    document_id, chunk_idx + 1, total_chunks,
                    chunk_start, chunk_start + len(chunk) - 1,
                    len(chunk_claims),
                )
                await asyncio.sleep(0)  # Yield to event loop
                return chunk_claims

        tasks = [
            _process_chunk(cs, i) for i, cs in enumerate(chunk_ranges)
        ]
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in chunk_results:
            if isinstance(result, Exception):
                logger.error("Chunk analysis failed: %s", result)
                continue
            all_claims.extend(result)

        # Assign focus_area to each claim based on claim_type
        for claim in all_claims:
            mapped = CLAIM_TYPE_TO_FOCUS_AREA.get(claim.claim_type)
            claim.focus_area = mapped.value if mapped else "technology"

        # Filter by focus areas if specified
        if active_focus:
            before_count = len(all_claims)
            all_claims = [c for c in all_claims if c.focus_area in active_focus]
            logger.info(
                "Focus area filter: %d -> %d claims (kept areas: %s)",
                before_count, len(all_claims), active_focus,
            )

        # Verification pass: double-check outdated flags with age-aware prompt
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
        second opinion. Uses age-aware prompt that favors keeping outdated flags
        for older documents.
        """
        outdated_claims = [c for c in claims if c.is_outdated]
        if not outdated_claims:
            return claims

        current_year = datetime.utcnow().year
        est_pub_year = self._estimated_pub_year or (current_year - 10)
        doc_age = self._document_age or 10

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
        system = VERIFICATION_PROMPT.format(
            current_year=current_year,
            estimated_pub_year=est_pub_year,
            document_age=doc_age,
        )

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

            # Build lookup: claim_id -> verified is_outdated
            corrections = {}
            for v in verified:
                cid = v.get("claim_id")
                if cid:
                    corrections[cid] = v.get("is_outdated", True)
                    reason = v.get("reason", "")
                    logger.info(
                        "Verification [%s]: is_outdated=%s -- %s",
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
                            "Corrected claim %s: outdated %s -> %s",
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

        current_year = datetime.utcnow().year
        est_pub_year = self._estimated_pub_year or (current_year - 10)
        doc_age = self._document_age or 10

        system = SYSTEM_PROMPT.format(
            estimated_pub_year=est_pub_year,
            document_age=doc_age,
            current_year=current_year,
        )

        if start_idx == 0:
            logger.info("First chunk content being sent to GPT:\n%s", numbered[:1000])

        try:
            response = await self._call_with_retry(system, numbered, model=self.analysis_model)
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

    async def _call_with_retry(self, system: str, user_content: str, model: str = None):
        """Call OpenAI with TPM-aware throttling and retry on transient errors."""
        use_model = model or self.model
        # Estimate tokens (~4 chars per token) for throttling
        estimated_tokens = (len(system) + len(user_content)) // 4
        await _throttle_for_tpm(estimated_tokens)

        for attempt in range(MAX_RETRIES):
            try:
                return await self.client.chat.completions.create(
                    model=use_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    timeout=120.0,
                )
            except RateLimitError as e:
                # Parse "Please try again in Xs" from the error message
                msg = str(e)
                match = re.search(r'try again in (\d+(?:\.\d+)?)\s*s', msg)
                if match:
                    delay = float(match.group(1)) + 0.5
                else:
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt), 15)
                logger.warning(
                    "OpenAI RateLimitError (attempt %d/%d), waiting %.1fs",
                    attempt + 1, MAX_RETRIES, delay,
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(delay)
                else:
                    logger.error("OpenAI call failed after %d retries (rate limit)", MAX_RETRIES)
                    return None
            except (APITimeoutError, APIConnectionError, APIError) as e:
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
