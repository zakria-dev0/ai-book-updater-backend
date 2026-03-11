"""
Update Agent — uses GPT-4o to generate change proposals by comparing
original text with research findings.

Generic for any technical textbook:
- Dynamic style profile from document analysis (not hardcoded to any book)
- Context-aware: GPT sees surrounding paragraphs for style continuity
- Few-shot style examples (bad vs good) injected per change type
- Multi-source synthesis (combines up to 5 sources into comprehensive updates)
- core_claim_status tracking (false / outdated / incomplete / still_true)
"""
import json
import uuid
import asyncio
from datetime import datetime
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

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1
MAX_CONCURRENT_PROPOSALS = 3  # Limit concurrent GPT calls — constrained by 30K TPM

# ------------------------------------------------------------------ #
# System prompt — dynamic, adapts to any textbook via style profile    #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """\
You are an expert editor for a technical textbook. Your task is to generate factually
updated replacement text for outdated passages. The updated text must be INDISTINGUISHABLE
from the original authors' writing style.

=======================================
DOCUMENT WRITING STYLE — FOLLOW THIS PRECISELY
=======================================

{style_section}

=======================================
ABSOLUTE RULE: NO FABRICATION (CRITICAL)
=======================================

You MUST ONLY include facts, numbers, dates, names, and events that appear in the
provided RESEARCH SOURCES. This is the single most important rule.

NEVER:
- Invent successor systems, missions, or versions (e.g., do NOT invent "FireSat II")
- Fabricate dollar amounts, thresholds, or statistics not in sources
- Claim specific dates/numbers unless a source explicitly states them
- Assume outcomes for events your sources do not cover
- Fill gaps in source data with plausible-sounding but unverified claims

IF SOURCES ARE INSUFFICIENT:
- Write a narrower but accurate update using only what sources confirm
- Use phrases like "significant changes have occurred in this area since [year]"
- It is MUCH better to be vague-but-correct than specific-but-fabricated
- If sources are too weak to write a meaningful update, return {{"proposals": []}}

=======================================
SPECIAL CASES
=======================================

TEXTBOOK EXAMPLES & HYPOTHETICAL SCENARIOS:
- If the original text uses a hypothetical example for teaching (e.g., "FireSat",
  "ExampleSat"), do NOT invent real-world successors or claim the example changed.
- IMPORTANT: Even if a real-world entity shares the same name as the textbook example
  (e.g., Muon Space's "FireSat" constellation), do NOT connect them — the textbook's
  example is FICTIONAL and unrelated to any real company's product.
- Instead, note whether the UNDERLYING CONCEPTS the example illustrates have evolved.
- If the example itself is fine as-is for teaching, return {{"proposals": []}}.

GENERAL PRINCIPLES & TIMELESS STATEMENTS:
- If the original states a general truth that is STILL TRUE (e.g., "Space is expensive"),
  do NOT rewrite it just because the document is old.
- Only update if the core claim has materially changed.

=======================================
UPDATE STRUCTURE (CRITICAL)
=======================================

YOUR UPDATE MUST DESCRIBE THE CURRENT STATE OF AFFAIRS — NOT HISTORY.
The reader already knows the old information. They need to know WHAT IS TRUE NOW.

STRUCTURE:
1. CORRECTION FIRST — State what is different NOW in sentence 1.
   BAD: "Initiated by DARPA in the mid-1980s, the LightSat initiative aimed to..."
   GOOD: "Small satellites have become a mainstream capability, with over 2,000 launched annually."
2. SPECIFIC EVIDENCE — Include exact current data points FROM YOUR SOURCES (names, dates, numbers).
3. CONTEXT — Briefly explain what changed (1-2 sentences).

WRONG APPROACH (do NOT do this):
- Do NOT write a HISTORY of the topic — the update should be about THE PRESENT
- Do NOT explain how things developed over time — jump to the current state
- Do NOT start with "Initiated by..." or "Originally developed..." or "The development of..."

RIGHT APPROACH:
- Lead with WHAT IS TRUE TODAY
- Then optionally add 1-2 sentences of context about what changed

PARAGRAPH LENGTH:
- Match the original's scope: if original was 1 sentence, write 2-3 sentences
- If original was a paragraph, write a comparable paragraph (4-6 sentences)

=======================================
SOURCE USAGE RULES
=======================================

1. ONLY use facts from the provided research sources — never from your training data
2. Prioritize: government/official sources > academic/industry > general news
3. If a source is clearly about a DIFFERENT topic than the claim, IGNORE that source
4. Cross-reference: if only one low-quality source supports a fact, mark confidence "low"
5. Every specific claim in new_content must be traceable to a provided source
6. CHECK SOURCE DATES — if a source is from before 2020, its data may be outdated.
   Prefer facts from the most recent sources. If the only source for a number is
   from 2010, note the date in your update or skip that fact.
7. TENSE ACCURACY — if a source says something is "planned", "proposed", "expected",
   or "under development", do NOT present it as completed or achieved. Use the SAME
   tense as the source. Getting tense wrong is a FACTUAL ERROR.

=======================================
CONFIDENCE SCORING (STRICT)
=======================================

"high" — 2+ authoritative sources (government/academic) directly confirm the update
"medium" — 1 authoritative source OR 2+ news/commercial sources confirm the update
"low" — Only indirect/tangential sources available, or sources are about related but
         different topics. MOST updates should be "medium" or "low".

If you are unsure whether sources truly support your update, use "low".

=======================================
WHAT NOT TO DO
=======================================

- Do NOT use passive voice as the primary voice
- Do NOT start with "It is worth noting..." or "It should be mentioned..."
- Do NOT use hedging language when facts are clearly stated in sources
- Do NOT write in news article style — this is textbook prose
- Do NOT use bullet points — flowing prose only
- Do NOT open any sentence with "As of [year]," — put the subject first
- Do NOT end with editorial commentary ("This highlights...", "This underscores...")
- Do NOT use participial editorial phrases ANYWHERE:
  BANNED: "highlighting the...", "underscoring the...", "marking ... as a pivotal...",
  "thereby addressing...", "reflecting the...", "signaling the..."
- Do NOT use "significant", "crucial", "important", "pivotal", "notable", "assertive",
  "bold", "unprecedented", "sweeping" filler adjectives
- Do NOT use "thereby" — it is always editorial filler
- Do NOT use vague references like "this trend", "this shift", "this evolution" — name the SPECIFIC thing
- Do NOT use "exemplifies", "exemplified by", "epitomizes", "embodies" — just state the fact directly
- Do NOT use clichés: "paving the way", "ushering in a new era", "on the cusp of",
  "poised to", "a testament to", "a harbinger of", "at the forefront"
- Do NOT start sentences with "Such innovations..." or "Such advancements..." — name WHAT specifically
- Do NOT use "highlights the development of" — just describe the development directly
- Do NOT write about HISTORY — your update must describe WHAT IS TRUE NOW, not how we got here
- The LAST sentence must state a SPECIFIC FACT (name, date, number) — never an interpretation

=======================================
OUTPUT FORMAT
=======================================

Return JSON:
{{
  "proposals": [
    {{
      "old_content": "exact original text being replaced",
      "new_content": "updated text matching book style — every fact sourced",
      "change_type": "tech_update"|"mission_update"|"constellation_update"|
                    "statistics_update"|"company_update"|"historical_correction"|
                    "system_update"|"regulation_update"|"business_model_update"|
                    "reference_update"|"methodology_update"|"landscape_update"|"prediction_update",
      "confidence": "high"|"medium"|"low",
      "core_claim_status": "false"|"outdated"|"incomplete"|"still_true",
      "reasoning": "What changed + which specific source(s) confirm it"
    }}
  ]
}}

If no update is warranted or sources are insufficient, return {{"proposals": []}}.
Return ONLY valid JSON — no markdown fences, no extra text.
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
    "landscape_update": "tech_update",
    "reference_update": "tech_update",
    "methodology_update": "tech_update",
    "prediction_update": "mission_update",
    "mission_update": "mission_update",
    "constellation_update": "mission_update",
    "regulation_update": "mission_update",
    "regulatory_update": "mission_update",
}


def _build_style_section(style_profile: dict | None) -> str:
    """
    Build the DOCUMENT WRITING STYLE section dynamically from the style profile
    detected during analysis. Falls back to sensible defaults if no profile available.
    """
    if not style_profile:
        style_profile = {}

    grade = style_profile.get("grade_level", "college_senior")
    depth = style_profile.get("technical_depth", "intermediate")
    tone = style_profile.get("tone", "formal_academic")
    complexity = style_profile.get("sentence_complexity", "moderate")
    terminology = style_profile.get("terminology_level", "technical")
    avg_length = style_profile.get("avg_sentence_length", 25)
    passive = style_profile.get("passive_voice_usage", "moderate")

    # Map grade levels to audience descriptions
    grade_desc = {
        "college_freshman": "college freshman level — accessible with minimal prerequisites",
        "college_junior": "college sophomore/junior level — assumes foundational coursework",
        "college_senior": "college senior level — assumes substantial technical background",
        "graduate": "graduate level — assumes advanced domain knowledge",
    }.get(grade, "college level")

    # Map tone to writing guidance
    tone_desc = {
        "conversational": "Conversational and engaging — use first person, contractions are acceptable",
        "formal_academic": "Formal academic — no contractions, precise language, scholarly register",
        "authoritative": "Authoritative yet accessible — confident without being condescending",
    }.get(tone, "Formal and precise")

    # Map passive voice
    voice_guidance = {
        "rare": "ACTIVE VOICE strongly preferred — use passive voice sparingly (less than 10%)",
        "moderate": "Mix active and passive voice naturally — prefer active for clarity",
        "frequent": "Passive voice is acceptable per the document's existing style",
    }.get(passive, "Prefer active voice")

    return f"""GRADE LEVEL & AUDIENCE:
- {grade_desc}
- Technical depth: {depth}

TONE:
- {tone_desc}
- Never use jargon without defining it on first use

SENTENCE STRUCTURE:
- Sentence complexity: {complexity}
- Average sentence length: ~{avg_length} words
- {voice_guidance}
- Use transitional phrases: "For example,", "More recently,", "In contrast,"

TERMINOLOGY:
- Level: {terminology}
- Always spell out acronyms on first use
- Define technical terms in parentheses on first use
- Include specific technical details: exact numbers, dates, names

NUMBERS AND DATA:
- Always include specific numbers — never say "many" when you can say an exact count
- Include exact dates when known
- Use dual units for measurements when the original does so"""


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

# Editorial verbs (base forms) that GPT uses for commentary
_EDITORIAL_VERBS_BASE = (
    r'(?:highlight|underscore|demonstrate|mark|illustrate|showcase|emphasize|'
    r'signal|reflect|reveal|underscore|indicate|suggest|point\s+to|show)'
)
# -ing forms
_EDITORIAL_GERUNDS = (
    r'(?:highlighting|underscoring|demonstrating|marking|illustrating|showcasing|'
    r'emphasizing|signaling|reflecting|revealing|indicating|suggesting|addressing)'
)

_FORBIDDEN_ENDING_PATTERNS = [
    # Pattern 1: Last sentence starting with "This/These/Such" + editorial verb or filler
    # "This highlights...", "These changes underscore...", "Such innovations are paving..."
    _re.compile(
        r'\.\s+'
        r'(?:This|These|Such|The shift|The change|The move|The transition)'
        r'\s+(?:\w+\s+)?'  # optional word like "development" before verb
        r'(?:'
        + _EDITORIAL_VERBS_BASE +
        r's?'
        r'|(?:are|is|were|was)\s+(?:paving|ushering|driving|fueling|enabling|transforming|reshaping|revolutionizing)'
        r')\s+'
        r'.{10,}$',
        _re.IGNORECASE,
    ),
    # Pattern 2: Sentence containing editorial gerund + abstract noun
    # "...highlighting the challenges/importance/dynamics..."
    _re.compile(
        r'[.,]\s+'
        r'.{0,40}'
        + _EDITORIAL_GERUNDS +
        r'\s+(?:the\s+)?'
        r'(?:challenges?|complexities|difficulties|importance|significance|immense|'
        r'growing|urgency|dynamics|shift|need|evolution|transformation|trajectory|'
        r'pivotal|nature|scope|extent|breadth|depth|scale)'
        r'.{3,}$',
        _re.IGNORECASE,
    ),
    # Pattern 3: ", [gerund] the [adjective] [noun]..." anywhere near the end
    # "..., highlighting the urgent need for..."
    _re.compile(
        r',\s*'
        + _EDITORIAL_GERUNDS +
        r'\s+(?:the\s+)?'
        r'(?:urgent|growing|increasing|critical|immense|significant|pressing|'
        r'ongoing|continued|escalating|rapid|unprecedented|profound|broader|'
        r'evolving|changing|shifting|new|complex|mounting|stark|clear|evident)'
        r'\s+\w+'
        r'.{0,80}$',
        _re.IGNORECASE,
    ),
    # Pattern 4: "thereby [gerund]..." — always editorial
    # "..., thereby addressing these growing challenges"
    _re.compile(
        r',?\s*thereby\s+\w+ing\s+.{5,}$',
        _re.IGNORECASE,
    ),
    # Pattern 5: ", marking [X] as a [anything]..." — always editorial
    # Catches: "marking 2025 as a pivotal year", "marking 2025 as a year for structural changes"
    _re.compile(
        r',\s*marking\s+.{0,40}\s+as\s+.{5,}$',
        _re.IGNORECASE,
    ),
    # Pattern 6: Last sentence with editorial verb as main verb
    # "..., [subject] underscore/highlight/illustrate [the] [abstract noun]..."
    # Catches: "collaborations underscore the geopolitical dynamics...",
    #          "highlight ongoing efforts to establish..."
    _re.compile(
        r'\.\s+.{0,200}'
        + _EDITORIAL_VERBS_BASE +
        r's?\s+(?:the\s+)?'
        r'(?:geopolitical|broader|growing|shifting|evolving|changing|complex|'
        r'increasing|ongoing|underlying|profound|stark|critical|urgent)'
        r'\s+\w+'
        r'.{0,80}$',
        _re.IGNORECASE,
    ),
]


def _fix_forbidden_endings(text: str) -> str:
    """
    Remove editorial commentary sentences from the end of generated text.
    These sentences add no factual value and violate the textbook style.
    Handles both full-sentence patterns (remove from the preceding period)
    and mid-sentence comma patterns (remove from the comma).
    Applies iteratively — a single proposal may have multiple editorial phrases.
    """
    changed = True
    while changed:
        changed = False
        for pattern in _FORBIDDEN_ENDING_PATTERNS:
            match = pattern.search(text)
            if match:
                # Determine cut point: if match starts at a period, keep the period;
                # if it starts at a comma, cut at the comma and add a period instead
                match_char = text[match.start()]
                if match_char == ',':
                    cleaned = text[:match.start()].rstrip() + "."
                else:
                    cleaned = text[:match.start() + 1].strip()  # +1 to keep the period
                if len(cleaned) > 20:  # Safety: don't reduce to near-empty
                    text = cleaned
                    changed = True
                    break  # Restart pattern scan on shortened text
    return text


# ------------------------------------------------------------------ #
# Post-processing: mid-text editorial cleanup                          #
# ------------------------------------------------------------------ #

# Filler adjectives/adverbs to strip (GPT adds these despite instructions)
_FILLER_WORDS = _re.compile(
    r'\b(significantly|substantial|substantially|crucially|notably|remarkably|'
    r'dramatically|fundamentally|profoundly|critically|immensely|tremendously|'
    r'pivotal|pivotally|pioneering)\b\s*',
    _re.IGNORECASE,
)

# "significant" before a noun — replace with nothing or a neutral word
# "a significant growth phase" → "a growth phase"
_SIGNIFICANT_ADJ = _re.compile(
    r'\b(significant|crucial|critical|pivotal|notable|remarkable|dramatic|immense|tremendous|'
    r'assertive|aggressive|bold|unprecedented|sweeping|paradigm)\s+',
    _re.IGNORECASE,
)

# Mid-text editorial verb patterns: "underscores the urgent need", "reflects the growing..."
# These rewrite the clause to remove the editorial framing
_MIDTEXT_EDITORIAL = [
    # ", underscores/highlights [adj] need/efforts..." → remove that clause
    # Catches both "highlight the ongoing need" and "highlight ongoing efforts"
    _re.compile(
        r',?\s*'
        r'(?:underscores?|highlights?|reflects?|illustrates?|demonstrates?|showcases?|emphasizes?|reveals?)'
        r'\s+(?:the\s+)?'
        r'(?:urgent|growing|critical|pressing|increasing|escalating|continued|ongoing|clear|stark|evident)'
        r'\s+'
        r'(?:need|importance|challenge|threat|risk|demand|concern|requirement|imperative|efforts?|push|drive|trend)'
        r'(?:\s+(?:for|of|to|toward|towards|in)\s+[^.]{5,80})?'
        r'\.',
        _re.IGNORECASE,
    ),
    # ", reflecting/indicating a/the [noun phrase]" → replace with period
    _re.compile(
        r',\s*(?:reflecting|indicating|suggesting|signaling)\s+(?:a|the|an)\s+[^.]{10,80}\.',
        _re.IGNORECASE,
    ),
    # "exemplifies this trend" — vague editorial reference
    _re.compile(
        r'\s+(?:exemplifies?|epitomizes?|typifies?|embodies?)\s+this\s+'
        r'(?:trend|shift|transition|evolution|movement|development)\b',
        _re.IGNORECASE,
    ),
]

# Cliché phrases to remove entirely (replaced with nothing)
_CLICHE_PHRASES = _re.compile(
    r'(?:,?\s*paving the way for [^.]{5,60}\.?'
    r'|,?\s*ushering in (?:a |an )?[^.]{5,50}\.?'
    r'|,?\s*on the cusp of [^.]{5,40}\.?'
    r'|,?\s*poised to [^.]{5,60}\.?'
    r'|,?\s*a testament to [^.]{5,60}\.?'
    r'|,?\s*at the forefront of [^.]{5,50}\.?'
    r'|,?\s*a harbinger of [^.]{5,50}\.?)',
    _re.IGNORECASE,
)

# "Such innovations/advancements..." starter — replace with specific subject
_SUCH_STARTER = _re.compile(
    r'(?:Such|These)\s+(?:innovations?|advancements?|developments?|efforts?|initiatives?|achievements?)'
    r'\s+(?:are|is|have|has|were|was)\s+',
    _re.IGNORECASE,
)


def _clean_midtext_editorial(text: str) -> str:
    """
    Remove filler adjectives/adverbs and mid-text editorial phrases.
    Unlike _fix_forbidden_endings which trims from the end, this cleans
    editorial language embedded anywhere in the text.
    """
    # Step 1: Remove filler adverbs (significantly, substantially, etc.)
    text = _FILLER_WORDS.sub('', text)

    # Step 2: Remove filler adjectives before nouns (significant growth → growth)
    text = _SIGNIFICANT_ADJ.sub('', text)

    # Step 2b: Replace editorial verb + "the development/progress of" with neutral "includes"
    # "highlights the development of X" → "includes the development of X"
    text = _re.sub(
        r'\b(?:highlights?|underscores?|showcases?|illustrates?|demonstrates?)\s+'
        r'(the\s+(?:development|creation|emergence|evolution|rise|growth|expansion|'
        r'progress|advancement|deployment)\s+of\s+)',
        r'includes \1',
        text,
        flags=_re.IGNORECASE,
    )

    # Step 2c: Replace "exemplified by" with "such as" (neutral alternative)
    text = _re.sub(
        r'\b(?:exemplified|epitomized|typified|embodied)\s+by\b',
        'such as',
        text,
        flags=_re.IGNORECASE,
    )

    # Step 3: Remove mid-text editorial clauses
    for pattern in _MIDTEXT_EDITORIAL:
        match = pattern.search(text)
        if match:
            # Replace the editorial clause with just a period
            text = text[:match.start()] + "." + text[match.end():]

    # Step 3b: Remove cliché phrases ("paving the way", "ushering in", etc.)
    text = _CLICHE_PHRASES.sub('.', text)

    # Step 3c: Replace "highlights/demonstrates the potential for" with neutral phrasing
    text = _re.sub(
        r'\b(?:highlights?|demonstrates?|showcases?|underscores?)\s+the\s+potential\s+(?:for|of)\b',
        'shows the feasibility of',
        text,
        flags=_re.IGNORECASE,
    )

    # Step 4: Clean up double spaces and awkward punctuation
    text = _re.sub(r'\s+,', ',', text)     # extra space before comma ("progressed , particularly")
    text = _re.sub(r'\s+\.', '.', text)    # extra space before period
    text = _re.sub(r'\s{2,}', ' ', text)   # double spaces
    text = _re.sub(r'\.\s*\.', '.', text)  # double periods
    text = _re.sub(r',\s*\.', '.', text)   # comma before period

    # Step 5: Fix a/an grammar (removing adjectives can leave "a" before vowels)
    text = _re.sub(r'\ba\s+([aeiouAEIOU])', r'an \1', text)
    # Also fix "an" before consonants if somehow introduced
    text = _re.sub(r'\ban\s+([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])', r'a \1', text)
    text = text.strip()

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
    # Count sentence boundaries: punctuation followed by space+uppercase or end-of-string
    # The $-alternative already counts the final sentence, so do NOT add +1 again.
    count = len(_SENTENCE_SPLIT.findall(text))
    # Only add 1 if text does NOT end with punctuation (trailing sentence without period)
    if text.rstrip() and text.rstrip()[-1:] not in '.!?':
        count += 1
    return max(count, 1)


def _fix_truncated_start(new_content: str) -> str:
    """
    Fix proposals that start with lowercase (truncated/malformed GPT output).
    If the first character is lowercase, capitalize it.
    """
    if new_content and new_content[0].islower():
        new_content = new_content[0].upper() + new_content[1:]
    return new_content


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
TODAY'S DATE: {today_date}

ORIGINAL TEXT FROM TEXTBOOK:
\"\"\"{old_content}\"\"\"

DOCUMENT CONTEXT (surrounding paragraph for reference):
\"\"\"{context}\"\"\"

FOCUS AREAS FOR THIS ANALYSIS: {focus_areas}

RESEARCH FINDINGS FROM AUTHORITATIVE SOURCES:
{research_results}

{source_quality_note}

DOCUMENT STYLE (from analysis):
{style_summary}
{style_example}

TASK:
1. Identify the CORE CLAIM in the original text.
2. Determine: is it false / outdated / incomplete / still_true?
3. Check if this is a textbook EXAMPLE or HYPOTHETICAL — if so, only update if the
   underlying concept has changed. Do NOT invent successors to fictional examples.
4. Review each source — does it ACTUALLY relate to this specific claim? Ignore irrelevant sources.
5. Write the update describing WHAT IS TRUE NOW — not a history lesson.
   WRONG: "The LightSat concept was initiated by DARPA in the 1980s and led to..."
   RIGHT: "Small satellites now represent a mainstream capability in the space industry,
           with commercial operators launching thousands of satellites annually."
6. Use ONLY facts from the relevant sources. Do NOT add facts from memory.
7. If sources don't provide enough information to write a meaningful update, return empty proposals.

CRITICAL WRITING CHECK — before finalizing your new_content, verify:
- Does sentence 1 describe the CURRENT STATE? (Not history, not background)
- Does every sentence contain at least one specific fact from sources?
- Is the last sentence a SPECIFIC FACT (name, date, number)?
- Did you avoid ALL banned phrases (check the WHAT NOT TO DO list)?

MANDATORY SOURCE CHECK:
Go through every sentence in your new_content and ask:
- "Which source explicitly states this fact?" If you cannot point to a source → DELETE that sentence.
- "Did I add any adjective, number, date, or name not in the sources?" If yes → REMOVE it.
- "Is this claim from my training data rather than the provided sources?" If yes → REMOVE it.
If after removing unsourced sentences you have fewer than 2 sentences, return empty proposals.

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
        style_profile: Optional[dict] = None,
        document_age: Optional[int] = None,
    ) -> List[ChangeProposal]:
        """
        For each claim that has research results, ask GPT-4o to generate
        a style-matched change proposal. Uses parallel async calls with concurrency limits.

        Args:
            claims: List of factual claims from content analysis
            research: Dict mapping claim_id -> list of research results
            document_id: The document being analyzed
            paragraphs: Full list of document paragraphs for surrounding context
            style_profile: Document style profile from analysis (grade, tone, etc.)
            document_age: Estimated document age in years (for context in prompts)
        """
        self._style_profile = style_profile
        self._document_age = document_age
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
                result = await self._generate_for_claim(
                    claim, sources, document_id, paragraphs or [],
                )
                await asyncio.sleep(0)  # Yield to event loop
                return result

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

        # ── Pre-filter sources: remove irrelevant ones before sending to GPT ──
        # Only keep sources with relevance >= 0.30 (already scored by research_service)
        MIN_SOURCE_RELEVANCE = 0.30
        filtered_sources = [s for s in sources if s.relevance_score >= MIN_SOURCE_RELEVANCE]
        if len(filtered_sources) < len(sources):
            dropped = len(sources) - len(filtered_sources)
            logger.info(
                "Pre-filtered %d/%d low-relevance sources for claim %s",
                dropped, len(sources), claim.claim_id,
            )

        # If no sources pass the filter, use the best available (top 2) but mark as weak
        if not filtered_sources and sources:
            filtered_sources = sorted(sources, key=lambda s: s.relevance_score, reverse=True)[:2]
            logger.info(
                "No high-relevance sources for claim %s — using top %d (best score=%.2f)",
                claim.claim_id, len(filtered_sources), filtered_sources[0].relevance_score,
            )

        # Build research results text — use filtered sources (up to 5)
        source_texts = []
        for i, s in enumerate(filtered_sources[:5]):
            source_texts.append(
                f"Source {i+1} [{s.source_type}, relevance={s.relevance_score:.2f}]: {s.source_title}\n"
                f"  URL: {s.source_url}\n"
                f"  Author: {s.author or 'unknown'}\n"
                f"  Published: {s.published_date or 'unknown'}\n"
                f"  Content: {s.snippet[:500]}"
            )

        # Get a relevant style example for this claim type
        style_example = _get_style_example(claim.claim_type)

        # Build dynamic style section from profile
        style_section = _build_style_section(getattr(self, '_style_profile', None))

        # Build the system prompt with dynamic style
        system_prompt = SYSTEM_PROMPT.format(style_section=style_section)

        # Build style summary for user prompt
        sp = getattr(self, '_style_profile', None) or {}
        style_summary = (
            f"- Grade: {sp.get('grade_level', 'college_senior')}\n"
            f"- Tone: {sp.get('tone', 'formal_academic')}\n"
            f"- Voice: {'Active preferred' if sp.get('passive_voice_usage', 'moderate') == 'rare' else 'Mixed active/passive'}, "
            f"~{sp.get('avg_sentence_length', 25)} words/sentence\n"
            f"- Terminology: {sp.get('terminology_level', 'technical')}\n"
            f"- Numbers: Always specific — exact dates, real figures"
        )

        # Build source quality note — warns GPT when sources are weak
        if not source_texts:
            source_quality_note = (
                "WARNING: No research sources are available for this claim. "
                "Return {\"proposals\": []} unless you can make a correction based purely "
                "on the fact that this claim is from a ~{age}-year-old document and is "
                "clearly outdated on its face (e.g., a prediction whose timeframe has passed)."
            ).format(age=getattr(self, '_document_age', 'unknown') or 'unknown')
        elif all(s.relevance_score < 0.40 for s in filtered_sources):
            source_quality_note = (
                "NOTE: The available sources have LOW relevance to this specific claim. "
                "Only use facts that DIRECTLY relate to the claim topic. If none do, "
                "return {\"proposals\": []} rather than forcing an irrelevant update. "
                "Set confidence to \"low\" if you proceed."
            )
        else:
            source_quality_note = ""

        # Build user prompt from template
        user_prompt = USER_PROMPT_TEMPLATE.format(
            today_date=datetime.now().strftime("%Y-%m-%d"),
            old_content=claim.text,
            context=context,
            focus_areas=claim.focus_area or "general",
            research_results="\n\n".join(source_texts) if source_texts else "(No sources available)",
            source_quality_note=source_quality_note,
            style_summary=style_summary,
            style_example=style_example,
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

                # ── Post-processing: fix forbidden ending patterns ────
                new_content = _fix_forbidden_endings(p["new_content"])
                if new_content != p["new_content"]:
                    logger.info(
                        "Fixed forbidden ending pattern in proposal for claim %s",
                        claim.claim_id,
                    )

                # ── Post-processing: clean mid-text editorial language ────
                cleaned = _clean_midtext_editorial(new_content)
                if cleaned != new_content:
                    logger.info(
                        "Cleaned mid-text editorial language in proposal for claim %s",
                        claim.claim_id,
                    )
                    new_content = cleaned

                # ── Post-processing: fix truncated start (lowercase first char) ──
                new_content = _fix_truncated_start(new_content)

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
                    "reference_update": ChangeType.REFERENCE_UPDATE,
                    "methodology_update": ChangeType.METHODOLOGY_UPDATE,
                    "landscape_update": ChangeType.LANDSCAPE_UPDATE,
                    "prediction_update": ChangeType.PREDICTION_UPDATE,
                }.get(change_type_str, ChangeType.TECH_UPDATE)

                # Parse core_claim_status
                core_status = p.get("core_claim_status", "").lower()
                if core_status not in ("false", "outdated", "incomplete", "still_true"):
                    core_status = None

                # Downgrade confidence if sources are weak
                if all(s.relevance_score < 0.40 for s in filtered_sources) and confidence != ConfidenceLevel.LOW:
                    logger.info(
                        "Downgrading confidence for claim %s: all sources below 0.40 relevance",
                        claim.claim_id,
                    )
                    confidence = ConfidenceLevel.LOW

                results.append(ChangeProposal(
                    change_id=f"change_{uuid.uuid4().hex[:12]}",
                    document_id=document_id,
                    claim_id=claim.claim_id,
                    old_content=p["old_content"],
                    new_content=new_content,
                    change_type=change_type,
                    confidence=confidence,
                    core_claim_status=core_status,
                    sources=filtered_sources,  # Only include relevant sources
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
            # Build dynamic system prompt with style profile
            style_section = _build_style_section(getattr(self, '_style_profile', None))
            system_prompt = SYSTEM_PROMPT.format(style_section=style_section)

            response = await self._call_with_retry(system_prompt, combined_prompt)
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
            new_content = _clean_midtext_editorial(new_content)

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
        """Call OpenAI with TPM-aware throttling and retry on transient errors."""
        import re
        from app.agents.ingestion_agent import _throttle_for_tpm
        estimated_tokens = (len(system) + len(user_content)) // 4
        await _throttle_for_tpm(estimated_tokens)

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
            except RateLimitError as e:
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
