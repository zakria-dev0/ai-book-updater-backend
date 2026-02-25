"""
Update Agent — uses GPT-4o to generate change proposals by comparing
original text with research findings.

Enhanced with:
- Writing style matching (uses document's style profile)
- Context-aware technical updates (aerospace-grade depth)
- Multi-source synthesis (combines 3-5 sources into comprehensive updates)
- Expanded change type support
- Better constellation/mega-constellation update generation
"""
import json
import uuid
import asyncio
from typing import Dict, List, Optional
from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIError, APIConnectionError
from app.core.config import settings
from app.core.logger import get_logger
from app.models.change import (
    FactualClaim,
    ResearchResult,
    ChangeProposal,
    ChangeType,
    ConfidenceLevel,
    StyleProfile,
)

logger = get_logger(__name__)

MAX_RETRIES = 5
RETRY_BASE_DELAY = 3
MAX_CONCURRENT_PROPOSALS = 3  # Limit concurrent GPT calls

# ------------------------------------------------------------------ #
# System prompt — style-aware, context-rich update generation          #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """\
You are an expert editor for academic and technical textbooks.
Given a factual claim from a book and research findings with updated information,
generate a **detailed, context-rich** update that matches the document's writing style.

## Document Writing Style Profile
{style_instructions}

## Update Generation Rules

1. **Match the writing style**: Your update MUST match the document's grade level, tone,
   sentence complexity, and terminology density described above. If the document uses formal
   academic language with complex sentences, your update should too. If it uses simpler language,
   match that.

2. **Provide context and significance**: Don't just swap facts. Explain WHY the change matters.
   Include implications, consequences, and engineering/scientific significance.

3. **Include specific technical details**:
   - Exact numbers, dates, and measurements
   - System names, mission designations, technical standards
   - Technical terminology appropriate to the field (e.g., LEO, GEO, orbital mechanics)
   - Version numbers, specification identifiers

4. **Synthesize multiple sources**: Use ALL provided research sources to build a comprehensive
   update. Cross-reference data points across sources. Don't just use the first source.

5. **For constellation/mega-constellation updates**: Include current constellation sizes (exact numbers),
   growth trajectory, technical architecture (LEO vs MEO/GEO), new applications enabled,
   engineering challenges (debris mitigation, spectrum allocation, inter-satellite links).

6. **Preserve the original's scope**: If the original text was one sentence, the update can be
   2-3 sentences if needed for technical accuracy. If the original was a paragraph, produce a
   comparable paragraph. Don't produce a tiny update for a paragraph-level claim or vice versa.

For each update, return a JSON object with:
- "old_content": the original text that needs to change
- "new_content": the updated replacement text (style-matched, detailed, technical)
- "change_type": one of "data_update", "tech_update", "mission_update", "company_update",
  "regulatory_update", "constellation_update", "statistics_update", "system_update",
  "regulation_update", "business_model_update", "historical_correction"
- "confidence": "high" if multiple authoritative sources agree, "medium" if one good source, "low" if uncertain
- "reasoning": brief explanation of why this change is needed and what sources support it

Return a JSON object with a "proposals" array. If no update is warranted, return {{"proposals": []}}.
Only return valid JSON — no markdown fences, no extra text.
"""

# Style instruction templates based on style profile values
STYLE_TEMPLATES = {
    "graduate_advanced": (
        "This is a GRADUATE-LEVEL textbook with ADVANCED technical depth.\n"
        "- Use highly technical terminology and domain-specific jargon freely\n"
        "- Write complex, multi-clause sentences (avg {avg_len} words/sentence)\n"
        "- Tone: {tone} — formal academic register, no colloquialisms\n"
        "- Include detailed technical specifications and engineering context\n"
        "- Use passive voice {passive} as is typical of academic writing\n"
        "- Reference specific standards, equations, and technical frameworks"
    ),
    "senior_advanced": (
        "This is a COLLEGE SENIOR-LEVEL textbook with ADVANCED technical depth.\n"
        "- Use technical terminology appropriate to upper-division courses\n"
        "- Write moderately complex sentences (avg {avg_len} words/sentence)\n"
        "- Tone: {tone}\n"
        "- Include specific numbers, dates, system names, and technical context\n"
        "- Use passive voice {passive}\n"
        "- Explain significance of technical changes for the field"
    ),
    "default": (
        "This is a {grade}-LEVEL textbook with {depth} technical depth.\n"
        "- Terminology level: {terminology}\n"
        "- Sentence complexity: {complexity} (avg {avg_len} words/sentence)\n"
        "- Tone: {tone}\n"
        "- Passive voice: {passive}\n"
        "- Match this style precisely in your updates"
    ),
}


def _build_style_instructions(style: Optional[StyleProfile]) -> str:
    """Build style instructions string from a StyleProfile."""
    if style is None:
        return (
            "No style profile available. Default to formal academic textbook style:\n"
            "- College senior level, advanced technical depth\n"
            "- Formal academic tone with complex sentences\n"
            "- Use technical terminology freely\n"
            "- Include specific numbers, dates, and technical context"
        )

    grade = style.grade_level
    depth = style.technical_depth

    if grade == "graduate" and depth == "advanced":
        template = STYLE_TEMPLATES["graduate_advanced"]
    elif grade in ("college_senior", "graduate") and depth in ("advanced", "intermediate"):
        template = STYLE_TEMPLATES["senior_advanced"]
    else:
        template = STYLE_TEMPLATES["default"]

    return template.format(
        grade=grade,
        depth=depth,
        tone=style.tone,
        complexity=style.sentence_complexity,
        terminology=style.terminology_level,
        avg_len=style.avg_sentence_length,
        passive=style.passive_voice_usage,
    )


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
        style_profile: Optional[StyleProfile] = None,
    ) -> List[ChangeProposal]:
        """
        For each claim that has research results, ask GPT-4o to generate
        a style-matched change proposal. Uses parallel async calls with concurrency limits.
        """
        if not settings.OPENAI_API_KEY:
            logger.warning("OpenAI API key not configured — skipping proposal generation")
            return []

        claims_with_research = [c for c in claims if c.claim_id in research]
        if not claims_with_research:
            logger.info("No claims with research results — nothing to propose")
            return []

        # Build style-aware system prompt
        style_instructions = _build_style_instructions(style_profile)
        system_prompt = SYSTEM_PROMPT.format(style_instructions=style_instructions)

        logger.info(
            "Generating proposals for %d claims (max %d concurrent), style: %s/%s",
            len(claims_with_research), MAX_CONCURRENT_PROPOSALS,
            style_profile.grade_level if style_profile else "default",
            style_profile.technical_depth if style_profile else "default",
        )

        # Run proposal generation in parallel with semaphore control
        async def _generate_one(idx, claim):
            async with self._semaphore:
                logger.info(
                    "Generating proposal %d/%d for claim: %s",
                    idx + 1, len(claims_with_research), claim.text[:80],
                )
                sources = research[claim.claim_id]
                return await self._generate_for_claim(claim, sources, document_id, system_prompt)

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
        system_prompt: str,
    ) -> List[ChangeProposal]:
        """Generate change proposal(s) for a single claim with full source synthesis."""
        # Build research context — use ALL sources (up to 5) for synthesis
        source_texts = []
        for i, s in enumerate(sources[:5]):
            source_texts.append(
                f"Source {i+1} [{s.source_type}, relevance={s.relevance_score:.2f}]: {s.source_title}\n"
                f"  URL: {s.source_url}\n"
                f"  Author: {s.author or 'unknown'}\n"
                f"  Published: {s.published_date or 'unknown'}\n"
                f"  Content: {s.snippet[:500]}"
            )

        user_prompt = (
            f"## Original Claim\n"
            f"Text: \"{claim.text}\"\n"
            f"Type: {claim.claim_type}\n"
            f"Focus area: {claim.focus_area or 'general'}\n"
            f"Entities: {', '.join(claim.entities) if claim.entities else 'none'}\n"
            f"Dates mentioned: {', '.join(claim.temporal_refs) if claim.temporal_refs else 'none'}\n\n"
            f"## Research Findings ({len(sources)} sources — synthesize ALL of them)\n"
            + "\n\n".join(source_texts)
            + "\n\n## Instructions\n"
            f"Generate a detailed, technically accurate update that synthesizes information "
            f"from ALL {len(sources)} sources above. The update should match the document's "
            f"writing style as described in the system prompt. Include specific numbers, dates, "
            f"and technical terminology from the sources."
        )

        try:
            response = await self._call_with_retry(system_prompt, user_prompt)
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

                # Map change type — expanded set
                change_type_str = p.get("change_type", "data_update").lower()
                change_type = {
                    "data_update": ChangeType.DATA_UPDATE,
                    "tech_update": ChangeType.TECH_UPDATE,
                    "mission_update": ChangeType.MISSION_UPDATE,
                    "company_update": ChangeType.COMPANY_UPDATE,
                    "regulatory_update": ChangeType.REGULATORY_UPDATE,
                    "image_update": ChangeType.IMAGE_UPDATE,
                    "constellation_update": ChangeType.CONSTELLATION_UPDATE,
                    "statistics_update": ChangeType.STATISTICS_UPDATE,
                    "system_update": ChangeType.SYSTEM_UPDATE,
                    "regulation_update": ChangeType.REGULATION_UPDATE,
                    "business_model_update": ChangeType.BUSINESS_MODEL_UPDATE,
                    "historical_correction": ChangeType.HISTORICAL_CORRECTION,
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
        """Call OpenAI with exponential backoff retry on transient errors (500, rate limit, timeout, connection)."""
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
