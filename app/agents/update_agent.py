"""
Update Agent — uses GPT-4o to generate change proposals by comparing
original text with research findings.

Hardcoded for "Understanding Space: An Introduction to Astronautics" (4th Edition).
- Book-specific system prompt with exact style rules and anti-patterns
- Context-aware: GPT sees surrounding paragraphs for style continuity
- Few-shot style examples (bad vs good) injected per change type
- Multi-source synthesis (combines up to 5 sources into comprehensive updates)
- core_claim_status tracking (false / outdated / incomplete / still_true)
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
)

logger = get_logger(__name__)

MAX_RETRIES = 5
RETRY_BASE_DELAY = 3
MAX_CONCURRENT_PROPOSALS = 3  # Limit concurrent GPT calls

# ------------------------------------------------------------------ #
# Reference: Document style profile metadata                          #
# ------------------------------------------------------------------ #

STYLE_PROFILE = {
    "book_title": "Understanding Space: An Introduction to Astronautics",
    "edition": "4th",
    "chapter": 2,
    "grade_level": "undergraduate_sophomore_junior",
    "technical_depth": "intermediate",
    "tone": "authoritative_accessible",
    "voice": "active_preferred",
    "passive_voice_ratio": 0.06,
    "avg_sentence_length_words": 20,
    "sentence_complexity": "mixed_short_and_long",
    "terminology_level": "technical_with_inline_definitions",
    "numbers_style": "specific_with_dual_units",
    "paragraph_length_sentences": "4_to_7",
    "narrative_style": "chronological_storytelling",
    "acronym_rule": "spell_out_on_first_use_then_abbreviate",
    "figure_reference_style": "(see Figure X-XX) or (Figure X-XX)",
}

# ------------------------------------------------------------------ #
# System prompt — hardcoded for "Understanding Space" 4th Edition      #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """\
You are an expert editor for "Understanding Space: An Introduction to Astronautics" (4th Edition),
a widely-used undergraduate aerospace engineering and astronautics textbook written for college
sophomore and junior level students in STEM programs.

Your task is to generate factually updated replacement text for outdated passages in this textbook.
The updated text must be INDISTINGUISHABLE from the original authors' writing style.

═══════════════════════════════════════════════════
DOCUMENT WRITING STYLE — FOLLOW THIS PRECISELY
═══════════════════════════════════════════════════

GRADE LEVEL & AUDIENCE:
- Undergraduate college level (sophomore/junior STEM students)
- Readers have basic physics and math background but are not yet specialists
- Assume the reader is intelligent but encountering many of these concepts for the first time

TONE:
- Authoritative yet accessible — confident without being condescending
- Educational and engaging — this is a textbook, not a journal paper
- Slightly narrative — events are told as a story with historical context
- Never use jargon without immediately defining it

SENTENCE STRUCTURE:
- Mix short sentences (10–15 words) with longer explanatory ones (25–35 words)
- Average sentence length: ~20 words
- ACTIVE VOICE strongly preferred — use passive voice sparingly (less than 10% of sentences)
- Use transitional phrases to connect ideas: "For example,", "More recently,", "Despite this,",
  "In contrast,", "As a result,", "Building on this,"

TECHNICAL TERMINOLOGY RULES (CRITICAL):
- Always spell out acronyms on first use: "Low Earth Orbit (LEO)", "International Space Station (ISS)"
- Define technical terms in parentheses on first use: "mega-constellations (large fleets of hundreds
  or thousands of satellites)", "pushbroom sensor (a detector array that scans swaths of terrain
  as the satellite passes overhead)"
- After defining, use the short form freely: "LEO", "the ISS", "mega-constellations"
- Include specific technical details: exact altitudes, masses, dates, counts, velocities

NUMBERS AND DATA:
- Always include specific numbers — never say "many" when you can say "5,400"
- Use dual units for physical measurements: "550 kilometers (342 miles)", "250 kg (550 lb)"
- Spell out numbers under ten; use numerals for 10 and above: "three satellites", "24 satellites"
- Include exact dates when known: "launched on September 27, 2021", "as of early 2024"
- Dollar figures with context: "$10 billion in annual revenue"

STRUCTURE OF AN UPDATE (MANDATORY PATTERN):
1. CORRECTION FIRST — If the original claim is now false, state the correction in sentence 1.
   Do NOT bury the correction in the middle of the paragraph.
2. CONTEXT — Explain what changed and briefly why (1–2 sentences)
3. SIGNIFICANCE — Explain what this means for the field or the reader (1 sentence)
4. TECHNICAL DETAIL — Include specific system names, dates, numbers (woven throughout)

PARAGRAPH LENGTH:
- Match the original's scope: if original was 1 sentence → write 2–4 sentences
- If original was a paragraph → write a comparable paragraph (4–7 sentences)
- Never produce a one-word or one-clause update for a multi-sentence original

WHAT THE ORIGINAL TEXT LOOKS LIKE (EXAMPLES FOR STYLE MATCHING):

Example 1 — Active voice, specific data, narrative:
"SpaceX developed the Falcon 9 (Figure 2-58) in part with financial and technical support from NASA
as part of a program called Commercial Orbital Transportation Services (COTS) to create systems to
transport cargo to and from the ISS. Under COTS, SpaceX developed the Falcon 9 rocket and Dragon
spacecraft, while Orbital Sciences Corporation developed the Antares rocket and Cygnus spacecraft."

Example 2 — Accessible technical depth with definition:
"Most communications services are provided by satellites in geosynchronous orbit, at an altitude of
approximately 36,000 kilometers, where orbital velocity matches the speed of the Earth's rotation,
making the satellite appear stationary as we'll explore in Chapters 4 and 5."

Example 3 — Historical narrative style:
"After the Space Shuttle Columbia was lost on re-entry on February 1, 2003, the U.S. decided to
retire the Space Shuttle after the ISS was complete; the last Shuttle mission, STS-135, flew in
July 2011."

Example 4 — Company/mission update with context:
"In the United States, a lack of commercial business caused Boeing and Lockheed Martin to merge
their Delta and Atlas rocket programs into a joint venture, United Launch Alliance (ULA), which
sells rockets primarily to U.S. government customers. More recently, SpaceX has entered the
commercial launch market with its Falcon 9 rocket, launching its first commercial communications
satellite in late 2013."

═══════════════════════════════════════════════════
UPDATE GENERATION RULES
═══════════════════════════════════════════════════

1. VERIFY THE CORE CLAIM FIRST:
   Before writing the update, determine: Is the original claim still true today?
   - If FALSE → Begin with the correction. Example: "Landsat-9, launched on September 27, 2021,
     has since joined the fleet as the most recent addition..."
   - If OUTDATED/INCOMPLETE → Expand with current data while correcting what's stale.
   - If STILL TRUE but needs detail → Add context and updated numbers.

2. SYNTHESIZE MULTIPLE SOURCES:
   - Use information from ALL provided research sources, not just the first one
   - Cross-reference data points — if two NASA sources agree on a number, use it confidently
   - Prioritize: NASA/ESA/government sources > academic/industry > general news

3. INCLUDE SPECIFIC TECHNICAL DETAILS:
   - Mission names and designations (Crew Dragon Endeavour, Soyuz MS-25)
   - Exact numbers (5,400 satellites, 550 km orbit, $2.7 billion contract)
   - Dates (launched November 2020, as of February 2024)
   - System specifications when relevant (pushbroom vs. whiskbroom, LEO vs. GEO)

4. FOR CONSTELLATION / MEGA-CONSTELLATION UPDATES:
   - State current constellation size with exact number and date
   - Contrast with what the book said (e.g., "dozens" → now "thousands")
   - Name the major players: Starlink, OneWeb, Amazon Kuiper, Chinese Guowang
   - Mention key engineering implications: orbital debris, spectrum coordination,
     inter-satellite laser links, LEO broadband coverage
   - Include growth trajectory if relevant

5. FOR COMPANY STATUS UPDATES:
   - If a company went bankrupt or shut down → STATE THIS IN SENTENCE 1
   - If leadership changed → mention who leads now and when change occurred
   - If a program was cancelled → say so explicitly, don't soften with "faced challenges"

6. FIGURE REFERENCES:
   - Preserve any existing figure references from the original: "(see Figure 2-41)"
   - If the original referenced a figure, keep that reference in your update

7. WHAT NOT TO DO:
   - Do NOT use passive voice as the primary voice
   - Do NOT start updates with "It is worth noting that..." or "It should be mentioned..."
   - Do NOT use hedging language like "may have" or "could potentially" when facts are known
   - Do NOT write in a news article style (no inverted pyramid, no "According to...")
   - Do NOT use bullet points — this is flowing prose
   - Do NOT plagiarize source snippets — write in the textbook's own voice
   - Do NOT bury the correction in paragraph 2 after context-setting
   - Do NOT open any sentence with "As of [year]," — rephrase to put the subject first
   - Split any sentence over 35 words into two sentences at a natural conjunction or em-dash

═══════════════════════════════════════════════════
FORBIDDEN ENDING PATTERNS (CRITICAL — YOUR OUTPUT WILL BE REJECTED IF VIOLATED)
═══════════════════════════════════════════════════

NEVER end your new_content with ANY of these editorial commentary patterns:
   - "This highlights..."
   - "This underscores..."
   - "This demonstrates..."
   - "This marks a significant..."
   - "This shift underscores..."
   - "...highlighting the challenges..."
   - "...highlighting the complexities..."
   - "...underscoring the importance..."

These are editorial opinions, NOT factual textbook writing. The textbook NEVER editorializes.

INSTEAD, end on a SPECIFIC FACT — a date, a number, a technical detail, or a concrete outcome.

FORBIDDEN ENDING EXAMPLE:
  "This highlights the challenges faced by private ventures in executing complex
   interplanetary missions."

CORRECT ENDING EXAMPLE:
  "The company's assets were liquidated in early 2019 after a Swiss court ruling."

FORBIDDEN ENDING EXAMPLE:
  "This shift underscores the challenges of private missions requiring substantial
   governmental resources."

CORRECT ENDING EXAMPLE:
  "Tito subsequently booked two seats on SpaceX's planned Starship lunar flyby mission,
   announced in October 2022."

═══════════════════════════════════════════════════
MINIMUM QUALITY REQUIREMENTS (CRITICAL — ALL UPDATES MUST MEET THESE)
═══════════════════════════════════════════════════

Regardless of topic simplicity, EVERY update must be written at college-senior aerospace textbook level.
Even for simple factual updates (company went bankrupt, mission was canceled), you MUST provide:

1. MINIMUM 3 SENTENCES — Never write fewer than 3 sentences for any update.
2. TECHNICAL CONTEXT — Include at least one technical detail (vehicle name, orbit type,
   mission architecture, propulsion system, payload capacity, etc.)
3. SPECIFIC NUMBERS — At least 2 specific data points (dates, dollar amounts, masses,
   distances, satellite counts, crew sizes, etc.)
4. FIELD IMPLICATIONS — One sentence explaining what happened next or what this means
   for the broader aerospace ecosystem (but state it as FACT, not editorial opinion).

NEVER write in news-brief style. A news brief says:
  "XCOR filed for bankruptcy in 2017, halting the Lynx project."

A TEXTBOOK says:
  "XCOR Aerospace ceased operations in 2017 after filing for bankruptcy, ending development
   of its Lynx suborbital spaceplane before the vehicle completed test flights. The company
   had made notable technical progress — including successfully testing a liquid-oxygen
   piston pump capable of supplying the Lynx main engines — but encountered financial
   difficulties following the departure of key co-founders in 2015. The Lynx's horizontal
   takeoff and landing design, intended to enable multiple flights per day at low operating
   cost, represented an innovative approach that other suborbital vehicle developers continue
   to explore."

═══════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════

Return a JSON object with a "proposals" array. Each proposal must include:
{{
  "proposals": [
    {{
      "old_content": "exact original text being replaced",
      "new_content": "updated replacement text matching book style — MUST end on a specific fact, NOT editorial commentary",
      "change_type": one of: "tech_update" | "mission_update" | "constellation_update" |
                    "statistics_update" | "company_update" | "historical_correction" |
                    "system_update" | "regulation_update" | "business_model_update",
      "confidence": "high" | "medium" | "low",
      "core_claim_status": "false" | "outdated" | "incomplete" | "still_true",
      "reasoning": "One sentence: what changed and which sources confirm it"
    }}
  ]
}}

If no update is warranted, return {{"proposals": []}}.
Return ONLY valid JSON — no markdown fences, no extra text outside the JSON.
"""

# ------------------------------------------------------------------ #
# Style examples — bad vs good, injected selectively per change type  #
# ------------------------------------------------------------------ #

STYLE_EXAMPLES = {
    "company_update": {
        "bad": (
            "XCOR Aerospace had been developing a rocket-powered spaceplane known as the Lynx, "
            "which was designed to carry a pilot and a passenger. The project faced challenges, "
            "including the departure of key co-founders in 2015, which impacted the company's trajectory."
        ),
        "good": (
            "XCOR Aerospace ultimately ceased operations in 2017 after filing for bankruptcy, "
            "ending development of its Lynx suborbital spaceplane before the vehicle completed "
            "test flights. The company had made notable technical progress — including successfully "
            "testing a liquid oxygen piston pump capable of supplying the Lynx main engines — but "
            "encountered financial difficulties following the departure of key co-founders in 2015. "
            "The Lynx's horizontal takeoff and landing design, intended to enable multiple flights "
            "per day at low operating cost, represented an innovative approach to the suborbital "
            "market that other companies continue to explore."
        ),
    },
    "tech_update": {
        "bad": (
            "The latest addition to the Landsat fleet is Landsat-8, which became operational "
            "following its launch in 2013. Landsat-8 features the Operational Land Imager (OLI) "
            "and the Thermal Infrared Sensor (TIRS)..."
        ),
        "good": (
            "Landsat-9, launched on September 27, 2021, has since joined the fleet as the most "
            "recent addition, carrying an updated Operational Land Imager 2 (OLI-2) and Thermal "
            "Infrared Sensor 2 (TIRS-2) that improve upon the sensors aboard Landsat-8. Together, "
            "Landsat-8 and Landsat-9 operate in the same orbit offset by 180 degrees, enabling "
            "the pair to image any point on Earth every eight days — twice as often as either "
            "satellite could achieve alone — substantially improving the temporal resolution "
            "available for land-cover monitoring and environmental studies."
        ),
    },
    "mission_update": {
        "bad": (
            "While the Russian Soyuz spacecraft continues to be a reliable vehicle for transporting "
            "astronauts to and from the International Space Station (ISS), it is no longer the sole "
            "means of crew transportation. Since 2020, NASA's Commercial Crew Program has enabled "
            "the use of SpaceX's Crew Dragon spacecraft for crewed missions to the ISS."
        ),
        "good": (
            "Since 2020, U.S. crews have had a domestic option for reaching the ISS through NASA's "
            "Commercial Crew Program. SpaceX's Crew Dragon spacecraft — developed under this "
            "program — began carrying astronauts to the station in May 2020 with the Demo-2 mission, "
            "ending a nine-year gap in U.S. human launch capability. The Soyuz spacecraft continues "
            "to ferry Russian cosmonauts and partner-nation astronauts to the ISS, and both vehicles "
            "now contribute to crew rotation, providing greater operational flexibility and redundancy "
            "for the station program."
        ),
    },
}

# Map change types to which style example to use
_EXAMPLE_TYPE_MAP = {
    "company_update": "company_update",
    "business_model_update": "company_update",
    "historical_correction": "company_update",
    "tech_update": "tech_update",
    "statistics_update": "tech_update",
    "system_update": "tech_update",
    "mission_update": "mission_update",
    "constellation_update": "mission_update",
    "regulation_update": "mission_update",
    "regulatory_update": "mission_update",
}


def _get_style_example(claim_type: str) -> str:
    """Return a bad/good style example relevant to the claim type."""
    example_key = _EXAMPLE_TYPE_MAP.get(claim_type, "tech_update")
    example = STYLE_EXAMPLES.get(example_key, STYLE_EXAMPLES["tech_update"])
    return (
        f"\n\nSTYLE EXAMPLE — study this before writing:\n"
        f"BAD (do NOT write like this):\n\"{example['bad']}\"\n\n"
        f"GOOD (write like this):\n\"{example['good']}\""
    )


# ------------------------------------------------------------------ #
# Post-processing: forbidden ending pattern removal                    #
# ------------------------------------------------------------------ #

# Patterns that GPT frequently uses despite instructions not to.
# Each regex matches a final sentence that starts with one of these editorial phrases.
import re as _re

_FORBIDDEN_ENDING_PATTERNS = [
    _re.compile(
        r'\.\s+'
        r'(?:This|These|Such|The shift|The change|The move|The transition)'
        r'\s+'
        r'(?:highlight|underscore|demonstrate|mark|illustrate|showcase|emphasize|signal|reflect|reveal)'
        r's?\s+'
        r'.{10,}$',
        _re.IGNORECASE,
    ),
    _re.compile(
        r'\.\s+'
        r'.{0,30}'
        r'(?:highlighting|underscoring|demonstrating|marking|illustrating|showcasing|emphasizing|signaling)'
        r'\s+(?:the\s+)?'
        r'(?:challenges|complexities|difficulties|importance|significance|immense|growing)'
        r'.{5,}$',
        _re.IGNORECASE,
    ),
]


def _fix_forbidden_endings(text: str) -> str:
    """
    Remove editorial commentary sentences from the end of generated text.
    These sentences add no factual value and violate the textbook style.
    """
    for pattern in _FORBIDDEN_ENDING_PATTERNS:
        match = pattern.search(text)
        if match:
            # Remove the forbidden sentence, keep everything before it
            cleaned = text[:match.start() + 1].strip()  # +1 to keep the period
            if len(cleaned) > 20:  # Safety: don't reduce to near-empty
                return cleaned
    return text


# ------------------------------------------------------------------ #
# Post-processing: minimum quality enforcement                        #
# ------------------------------------------------------------------ #

# Sentence-ending pattern: period/question-mark/exclamation followed by
# space+uppercase or end-of-string.  Avoids splitting on abbreviations
# like "U.S." or "Dr." by requiring the next char to be uppercase.
_SENTENCE_SPLIT = _re.compile(r'[.!?](?:\s+[A-Z]|$)')

MIN_SENTENCES = 3
MIN_CHAR_LENGTH = 150  # ~2–3 short sentences


def _count_sentences(text: str) -> int:
    """Count approximate number of sentences in text."""
    if not text:
        return 0
    # Count sentence-ending punctuation followed by space+uppercase or end
    parts = _SENTENCE_SPLIT.split(text)
    # The number of sentences = number of splits + 1 (if text ends with punctuation)
    count = len(_SENTENCE_SPLIT.findall(text))
    # If text ends with sentence-ending punctuation, add 1 for the last sentence
    if text.rstrip()[-1:] in '.!?':
        count += 1
    return max(count, 1)


def _passes_quality_check(new_content: str) -> bool:
    """
    Programmatic quality gate: ensures the generated content meets
    minimum textbook quality standards.
    Returns True if the content passes, False if it needs regeneration.
    """
    sentence_count = _count_sentences(new_content)
    char_count = len(new_content)

    if sentence_count < MIN_SENTENCES:
        return False
    if char_count < MIN_CHAR_LENGTH:
        return False
    return True


REGENERATION_PROMPT = """\
Your previous output was REJECTED because it was too short (news-brief style).

Your new_content had only {sentence_count} sentence(s) and {char_count} characters.
MINIMUM REQUIREMENTS: at least 3 sentences and 150 characters.

Here is your rejected output:
\"\"\"{rejected_content}\"\"\"

Rewrite this as a proper COLLEGE-LEVEL AEROSPACE TEXTBOOK paragraph:
1. Minimum 3 sentences
2. Include technical context (vehicle names, orbit types, mission architecture)
3. Include at least 2 specific data points (dates, numbers, measurements)
4. End on a specific fact, NOT editorial commentary

Return the SAME JSON format as before. Only update the "new_content" field — keep all other fields identical.
"""


# ------------------------------------------------------------------ #
# User prompt template                                                #
# ------------------------------------------------------------------ #

USER_PROMPT_TEMPLATE = """\
ORIGINAL TEXT FROM TEXTBOOK:
\"\"\"{old_content}\"\"\"

DOCUMENT CONTEXT (surrounding paragraph for reference):
\"\"\"{context}\"\"\"

FOCUS AREAS FOR THIS ANALYSIS: {focus_areas}

RESEARCH FINDINGS FROM AUTHORITATIVE SOURCES:
{research_results}

DOCUMENT STYLE PROFILE SUMMARY:
- Grade: Undergraduate STEM textbook (sophomore/junior level)
- Tone: Authoritative yet accessible, narrative, educational
- Voice: ACTIVE preferred, ~20 words/sentence average
- Terminology: Define acronyms and jargon on first use
- Numbers: Always specific — exact dates, dual units, real figures
{style_example}

TASK:
1. First, identify the CORE CLAIM in the original text (one sentence).
2. Determine if that core claim is: false / outdated / incomplete / still_true.
3. If false or outdated — begin your new_content with the correction.
4. Write updated replacement text in the EXACT style of the textbook.
5. Match the original's length (sentence for sentence, paragraph for paragraph).

Return valid JSON only.
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
        paragraphs: Optional[List[str]] = None,
    ) -> List[ChangeProposal]:
        """
        For each claim that has research results, ask GPT-4o to generate
        a style-matched change proposal. Uses parallel async calls with concurrency limits.

        Args:
            claims: List of factual claims from content analysis
            research: Dict mapping claim_id -> list of research results
            document_id: The document being analyzed
            paragraphs: Full list of document paragraphs for surrounding context
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
                return await self._generate_for_claim(
                    claim, sources, document_id, paragraphs or [],
                )

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
        paragraphs: List[str],
    ) -> List[ChangeProposal]:
        """Generate change proposal(s) for a single claim with full source synthesis."""
        # Build surrounding context (±2 paragraphs around the claim)
        context = self._build_context(claim.paragraph_idx, paragraphs)

        # Build research results text — use ALL sources (up to 5) for synthesis
        source_texts = []
        for i, s in enumerate(sources[:5]):
            source_texts.append(
                f"Source {i+1} [{s.source_type}, relevance={s.relevance_score:.2f}]: {s.source_title}\n"
                f"  URL: {s.source_url}\n"
                f"  Author: {s.author or 'unknown'}\n"
                f"  Published: {s.published_date or 'unknown'}\n"
                f"  Content: {s.snippet[:500]}"
            )

        # Get a relevant style example for this claim type
        style_example = _get_style_example(claim.claim_type)

        # Build user prompt from template
        user_prompt = USER_PROMPT_TEMPLATE.format(
            old_content=claim.text,
            context=context,
            focus_areas=claim.focus_area or "general",
            research_results="\n\n".join(source_texts),
            style_example=style_example,
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

                # ── Post-processing: fix forbidden ending patterns ────
                new_content = _fix_forbidden_endings(p["new_content"])
                if new_content != p["new_content"]:
                    logger.info(
                        "Fixed forbidden ending pattern in proposal for claim %s",
                        claim.claim_id,
                    )

                # ── Post-processing: quality gate (min sentences/length) ──
                if not _passes_quality_check(new_content):
                    sc = _count_sentences(new_content)
                    cc = len(new_content)
                    logger.warning(
                        "Quality check FAILED for claim %s (%d sentences, %d chars) — regenerating",
                        claim.claim_id, sc, cc,
                    )
                    regenerated = await self._regenerate_short_proposal(
                        p, new_content, user_prompt,
                    )
                    if regenerated:
                        new_content = regenerated
                    else:
                        logger.warning(
                            "Regeneration also failed quality check for claim %s — using best effort",
                            claim.claim_id,
                        )

                # Map confidence string to enum
                confidence_str = p.get("confidence", "medium").lower()
                confidence = {
                    "high": ConfidenceLevel.HIGH,
                    "medium": ConfidenceLevel.MEDIUM,
                    "low": ConfidenceLevel.LOW,
                }.get(confidence_str, ConfidenceLevel.MEDIUM)

                # Map change type — expanded set
                change_type_str = p.get("change_type", "tech_update").lower()
                change_type = {
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
                    "data_update": ChangeType.DATA_UPDATE,  # backward compat
                }.get(change_type_str, ChangeType.TECH_UPDATE)

                # Parse core_claim_status
                core_status = p.get("core_claim_status", "").lower()
                if core_status not in ("false", "outdated", "incomplete", "still_true"):
                    core_status = None

                results.append(ChangeProposal(
                    change_id=f"change_{uuid.uuid4().hex[:12]}",
                    document_id=document_id,
                    claim_id=claim.claim_id,
                    old_content=p["old_content"],
                    new_content=new_content,
                    change_type=change_type,
                    confidence=confidence,
                    core_claim_status=core_status,
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

    async def _regenerate_short_proposal(
        self, original_proposal: dict, rejected_content: str, original_user_prompt: str,
    ) -> str | None:
        """
        Re-call GPT with a regeneration prompt when the initial output
        was too short (news-brief style). Returns improved new_content
        or None if regeneration also fails quality check.
        """
        regen_prompt = REGENERATION_PROMPT.format(
            sentence_count=_count_sentences(rejected_content),
            char_count=len(rejected_content),
            rejected_content=rejected_content,
        )
        # Append regeneration instruction to the original user prompt
        combined_prompt = original_user_prompt + "\n\n" + regen_prompt

        try:
            response = await self._call_with_retry(SYSTEM_PROMPT, combined_prompt)
            if response is None:
                return None

            if response.usage:
                self.total_prompt_tokens += response.usage.prompt_tokens
                self.total_completion_tokens += response.usage.completion_tokens

            raw = response.choices[0].message.content
            data = json.loads(raw)
            proposals = data.get("proposals", [])
            if not proposals:
                return None

            new_content = proposals[0].get("new_content", "")
            new_content = _fix_forbidden_endings(new_content)

            if _passes_quality_check(new_content):
                logger.info(
                    "Regeneration succeeded: %d sentences, %d chars",
                    _count_sentences(new_content), len(new_content),
                )
                return new_content

            logger.warning(
                "Regeneration still too short: %d sentences, %d chars",
                _count_sentences(new_content), len(new_content),
            )
            return None

        except Exception as e:
            logger.error("Regeneration failed: %s", e)
            return None

    @staticmethod
    def _build_context(paragraph_idx: int, paragraphs: List[str], window: int = 2) -> str:
        """Extract surrounding paragraphs (±window) for context."""
        if not paragraphs:
            return "(no surrounding context available)"

        start = max(0, paragraph_idx - window)
        end = min(len(paragraphs), paragraph_idx + window + 1)
        context_parts = []
        for i in range(start, end):
            marker = " >>> " if i == paragraph_idx else "     "
            context_parts.append(f"{marker}[para {i}] {paragraphs[i][:300]}")
        return "\n".join(context_parts)

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
