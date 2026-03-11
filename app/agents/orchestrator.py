"""
LangGraph Orchestrator — coordinates the full AI analysis pipeline:
  age_estimate → style_analyze → analyze → research → propose → validate → complete

Enhanced with:
- Document age estimation (determines publication era for age-aware detection)
- Focus area filtering (only analyze/research selected categories)
- Writing style analysis (run once, pass to proposal generation)
- Style-matched update generation
- Source-claim relevance validation
- Robust claims storage with diagnostic logging
"""
import uuid
import asyncio
import sys
from typing import TypedDict, List, Dict, Optional
from datetime import datetime
from langgraph.graph import StateGraph, END
from app.core.logger import get_logger
from app.models.change import (
    FactualClaim,
    ResearchResult,
    ChangeProposal,
    ChangeLog,
    ConfidenceLevel,
    StyleProfile,
)
from app.agents.ingestion_agent import ContentAnalysisAgent
from app.agents.research_agent import ResearchAgent
from app.agents.update_agent import UpdateAgent
from app.database.repositories.change_repo import ChangeRepository
from app.database.repositories.document_repo import DocumentRepository
from app.models.document import DocumentStatus
from app.core.config import settings

logger = get_logger(__name__)


# ------------------------------------------------------------------ #
# LangGraph State                                                      #
# ------------------------------------------------------------------ #

class AnalysisState(TypedDict):
    document_id: str
    text_content: str
    paragraphs: List[str]
    focus_areas: List[str]
    estimated_pub_year: Optional[int]  # Estimated publication year from age detection
    document_age: Optional[int]  # Estimated document age in years
    style_profile: Optional[dict]  # StyleProfile as dict for serialization
    claims: List[FactualClaim]
    research: Dict[str, List[ResearchResult]]
    proposals: List[ChangeProposal]
    validated_proposals: List[ChangeProposal]
    error: Optional[str]
    stage: str
    progress: int
    db: Optional[object]  # Database reference for progress updates


# ------------------------------------------------------------------ #
# Node functions                                                       #
# ------------------------------------------------------------------ #

content_agent = ContentAnalysisAgent()
research_agent = ResearchAgent()
update_agent = UpdateAgent()


async def _update_analysis_progress(state: AnalysisState, stage: str, progress: int, message: str = ""):
    """Write analysis progress to the document record in DB."""
    db = state.get("db")
    if db is None:
        return
    try:
        doc_repo = DocumentRepository(db)
        entry = {
            "stage": stage,
            "progress": progress,
            "timestamp": datetime.utcnow(),
            "message": message,
        }
        await doc_repo.update_fields(state["document_id"], {
            "current_stage": stage,
            "progress": progress,
        })
        await doc_repo.push_history_entry(state["document_id"], entry)
        # Yield to event loop so other HTTP requests (list docs, status polls) aren't starved
        await asyncio.sleep(0)
    except Exception as e:
        logger.warning("Failed to update analysis progress: %s", e)


async def age_estimate_node(state: AnalysisState) -> dict:
    """Stage 0: Estimate document publication year for age-aware analysis."""
    logger.info("Orchestrator [age_estimate]: estimating document age for %s", state["document_id"])
    await _update_analysis_progress(
        state, "Estimating document age", 2,
        "Analyzing references and language to determine publication era",
    )
    try:
        age_data = await content_agent.estimate_document_age(state["text_content"])
        est_year = age_data.get("estimated_publication_year", datetime.utcnow().year - 10)
        doc_age = age_data.get("document_age_years", 10)

        logger.info(
            "Document age estimation complete: published ~%d (%d years old, confidence=%s)",
            est_year, doc_age, age_data.get("confidence", "unknown"),
        )

        # Store in document metadata
        db = state.get("db")
        if db is not None:
            doc_repo = DocumentRepository(db)
            await doc_repo.update_fields(state["document_id"], {
                "estimated_pub_year": est_year,
                "document_age": doc_age,
            })

        return {
            "estimated_pub_year": est_year,
            "document_age": doc_age,
            "stage": "Document age estimated",
            "progress": 4,
        }
    except Exception as e:
        logger.error("Age estimation node failed: %s", e, exc_info=True)
        current_year = datetime.utcnow().year
        return {
            "estimated_pub_year": current_year - 10,
            "document_age": 10,
            "stage": "Age estimation failed (using 10-year default)",
            "progress": 3,
        }


async def style_analyze_node(state: AnalysisState) -> dict:
    """Stage 1: Analyze writing style of the document (run once)."""
    logger.info("Orchestrator [style]: analyzing writing style for document %s", state["document_id"])
    await _update_analysis_progress(
        state, "Analyzing writing style", 5,
        "Determining document grade level, tone, and complexity",
    )
    try:
        profile = await content_agent.analyze_style(state["text_content"])
        profile_dict = profile.model_dump()

        # Verify we got a real profile (not all defaults)
        if profile.grade_level == "college_senior" and profile.tone == "formal_academic":
            logger.warning(
                "Style profile returned all defaults — GPT may have failed silently. "
                "Profile: %s", profile_dict,
            )

        # Store style profile in document metadata
        db = state.get("db")
        if db is not None:
            doc_repo = DocumentRepository(db)
            await doc_repo.update_fields(state["document_id"], {
                "style_profile": profile_dict,
            })

        logger.info(
            "Style analysis complete: grade=%s, depth=%s, tone=%s",
            profile.grade_level, profile.technical_depth, profile.tone,
        )
        return {
            "style_profile": profile_dict,
            "stage": "Style analysis complete",
            "progress": 8,
        }
    except Exception as e:
        logger.error("Style analysis node failed: %s", e, exc_info=True)
        # Return a default profile instead of None so downstream nodes can still use it
        default_profile = StyleProfile().model_dump()
        return {
            "style_profile": default_profile,
            "stage": "Style analysis failed (using defaults)",
            "progress": 5,
        }


async def analyze_node(state: AnalysisState) -> dict:
    """Stage 1: Analyze document text for factual claims (with focus area filtering)."""
    focus_areas = state.get("focus_areas", ["all"])
    paragraphs = state["paragraphs"]
    total_paras = len(paragraphs)
    logger.info(
        "Orchestrator [analyze]: starting for document %s (%d paragraphs, focus: %s)",
        state["document_id"], total_paras, focus_areas,
    )
    await _update_analysis_progress(
        state, "Scanning for factual claims", 10,
        f"Analyzing {total_paras} paragraphs for factual claims (focus: {', '.join(focus_areas)})",
    )
    try:
        claims = await content_agent.analyze_document(
            document_id=state["document_id"],
            text_content=state["text_content"],
            paragraphs=paragraphs,
            focus_areas=focus_areas,
        )
        outdated_count = sum(1 for c in claims if c.is_outdated)
        await _update_analysis_progress(
            state, "Verifying claims", 28,
            f"Found {len(claims)} claims, {outdated_count} flagged outdated — verifying...",
        )
        await _update_analysis_progress(
            state, "Content analysis complete", 30,
            f"Found {len(claims)} claims, {outdated_count} confirmed outdated",
        )
        return {
            "claims": claims,
            "stage": "Content analysis complete",
            "progress": 30,
        }
    except Exception as e:
        logger.error("Analyze node failed: %s", e)
        return {"claims": [], "error": str(e), "stage": "Error", "progress": 0}


async def research_node(state: AnalysisState) -> dict:
    """Stage 2: Research outdated claims via Tavily."""
    outdated_count = sum(1 for c in state["claims"] if c.is_outdated)
    logger.info("Orchestrator [research]: researching %d claims", outdated_count)
    await _update_analysis_progress(
        state, "Researching outdated claims", 35,
        f"Web searching for updated information on {outdated_count} claims",
    )
    try:
        research = await research_agent.research_claims(state["claims"])
        await _update_analysis_progress(
            state, "Research complete", 50,
            f"Found authoritative sources for {len(research)} claims",
        )
        return {
            "research": research,
            "stage": "Research complete",
            "progress": 50,
        }
    except Exception as e:
        logger.error("Research node failed: %s", e)
        return {"research": {}, "error": str(e), "stage": "Error", "progress": 0}


async def propose_node(state: AnalysisState) -> dict:
    """Stage 3: Generate style-matched change proposals from research."""
    num_researched = len(state['research'])
    logger.info("Orchestrator [propose]: generating proposals for %d claims", num_researched)
    await _update_analysis_progress(
        state, "Generating update proposals", 55,
        f"Writing style-matched replacement text for {num_researched} outdated claims",
    )
    try:
        proposals = await update_agent.generate_proposals(
            claims=state["claims"],
            research=state["research"],
            document_id=state["document_id"],
            paragraphs=state.get("paragraphs", []),
            style_profile=state.get("style_profile"),
            document_age=state.get("document_age"),
        )
        await _update_analysis_progress(
            state, "Proposals generated", 80,
            f"Generated {len(proposals)} change proposals",
        )
        return {
            "proposals": proposals,
            "stage": "Proposals generated",
            "progress": 80,
        }
    except Exception as e:
        logger.error("Propose node failed: %s", e)
        return {"proposals": [], "error": str(e), "stage": "Error", "progress": 0}


def _check_source_relevance(proposal: ChangeProposal) -> float:
    """
    Check how many sources actually mention the key entities from the claim.
    Returns the fraction of sources that are content-relevant (0.0 to 1.0).
    """
    old_lower = proposal.old_content.lower()

    # Extract key terms from old_content (words 4+ chars, excluding common words)
    stop_words = {
        "that", "this", "with", "from", "will", "have", "been", "were", "which",
        "their", "they", "than", "also", "more", "into", "such", "about", "other",
        "first", "would", "could", "should", "most", "some", "when", "what", "over",
        "being", "made", "after", "before", "these", "those", "under", "while",
        "including", "satellites", "satellite", "mission", "missions", "space",
        "launch", "launched", "company", "plan", "plans", "planned",
    }

    words = set()
    for w in old_lower.split():
        cleaned = w.strip(".,;:()\"'")
        if len(cleaned) >= 4 and cleaned not in stop_words:
            words.add(cleaned)

    # Find proper nouns / key entities (capitalized words from original)
    key_entities = set()
    for w in proposal.old_content.split():
        cleaned = w.strip(".,;:()\"'")
        if cleaned and cleaned[0].isupper() and len(cleaned) >= 3:
            key_entities.add(cleaned.lower())

    # Prioritize entity matching — if entities exist, check those
    search_terms = key_entities if key_entities else words

    if not search_terms:
        return 1.0  # Can't check, assume relevant

    relevant_count = 0
    for source in proposal.sources:
        source_text = f"{source.source_title} {source.snippet}".lower()
        # A source is relevant if it mentions at least one key entity
        if any(term in source_text for term in search_terms):
            relevant_count += 1

    return relevant_count / len(proposal.sources) if proposal.sources else 0.0


def _detect_fabrication_signals(proposal: ChangeProposal) -> list[str]:
    """
    Detect signs that GPT may have fabricated content not supported by sources.
    Returns a list of warning reasons (empty = no issues detected).
    """
    import re
    warnings = []
    new_lower = proposal.new_content.lower()
    source_text = " ".join(
        f"{s.source_title} {s.snippet}".lower() for s in proposal.sources
    ) if proposal.sources else ""

    # Check for specific dollar amounts / thresholds in new_content
    # that don't appear in any source
    dollar_amounts = re.findall(r'\$[\d,.]+\s*(?:billion|million|trillion|B|M|T)?', proposal.new_content, re.IGNORECASE)
    for amount in dollar_amounts:
        # Normalize for matching: strip $ and spaces
        amount_core = amount.replace("$", "").replace(",", "").strip().split()[0]
        if amount_core not in source_text and amount_core.replace(".", "") not in source_text:
            warnings.append(f"Dollar amount '{amount}' not found in any source")

    # Check for specific percentages not in sources
    percentages = re.findall(r'\d+(?:\.\d+)?%', proposal.new_content)
    for pct in percentages:
        pct_num = pct.replace("%", "")
        if pct_num not in source_text:
            warnings.append(f"Percentage '{pct}' not found in any source")

    # Check for invented successor names (e.g., "FireSat II", "XYZ-2")
    # If new_content introduces a proper noun not in old_content OR sources
    old_lower = proposal.old_content.lower()
    new_proper_nouns = set(
        w.strip(".,;:()\"'") for w in proposal.new_content.split()
        if w and w[0].isupper() and len(w) >= 3
    )
    old_proper_nouns = set(
        w.strip(".,;:()\"'") for w in proposal.old_content.split()
        if w and w[0].isupper() and len(w) >= 3
    )
    novel_nouns = new_proper_nouns - old_proper_nouns
    for noun in novel_nouns:
        noun_lower = noun.lower()
        if noun_lower not in source_text and len(noun) >= 4:
            # Skip common words that happen to be capitalized (sentence starters)
            common_starts = {
                "the", "this", "these", "that", "such", "for", "more", "since",
                "recent", "furthermore", "consequently", "however", "contrary",
                "nasa", "esa",  # well-known abbreviations
            }
            if noun_lower not in common_starts:
                warnings.append(f"Novel proper noun '{noun}' not found in sources")

    return warnings


def _detect_tense_fabrication(proposal: ChangeProposal) -> list[str]:
    """
    Detect cases where GPT presents something as completed/achieved when sources
    only describe it as planned/proposed/tested. This catches subtle fabrications
    like "oxygen was produced on the Moon in 2025" when sources only say "planned."
    """
    import re
    warnings = []
    if not proposal.sources:
        return warnings

    new_text = proposal.new_content.lower()
    source_text = " ".join(
        f"{s.source_title} {s.snippet}".lower() for s in proposal.sources
    )

    # Words that indicate something HAPPENED vs was PLANNED
    completed_indicators = [
        "enabled", "achieved", "produced", "demonstrated", "completed",
        "successfully", "has been deployed", "was deployed", "landed",
        "constructed", "built", "established", "operational since",
    ]
    planned_indicators = [
        "planned", "proposed", "aims to", "expected to", "scheduled",
        "will be", "to be launched", "under development", "in development",
        "preparing for", "demonstration", "demo", "test", "prototype",
        "concept", "feasibility",
    ]

    # For each "completed" claim in new_content, check if sources only mention it as "planned"
    for indicator in completed_indicators:
        if indicator in new_text:
            # Find the sentence containing this indicator
            sentences = re.split(r'[.!?]\s+', proposal.new_content)
            for sent in sentences:
                sent_lower = sent.lower()
                if indicator in sent_lower:
                    # Extract key nouns from this sentence to search in sources
                    key_words = [w for w in sent_lower.split() if len(w) >= 5
                                 and w not in {"which", "their", "these", "those", "about",
                                               "being", "while", "after", "since", "through"}]
                    # Check if sources only describe this as planned/proposed
                    for kw in key_words[:5]:
                        if kw in source_text:
                            # Found the topic in sources — check tense context
                            # Get surrounding context in source
                            idx = source_text.index(kw)
                            context = source_text[max(0, idx-100):idx+100]
                            source_is_planned = any(p in context for p in planned_indicators)
                            source_is_completed = any(c in context for c in completed_indicators)
                            if source_is_planned and not source_is_completed:
                                warnings.append(
                                    f"Claim uses '{indicator}' but source context for '{kw}' "
                                    f"only mentions planned/proposed status"
                                )
                            break  # Only check first matching keyword per sentence

    return warnings


def _compute_content_sourcing_score(proposal: ChangeProposal) -> float:
    """
    Measure what fraction of key facts in new_content are actually grounded
    in source snippets. Returns 0.0 (nothing sourced) to 1.0 (fully sourced).
    """
    import re
    if not proposal.sources:
        return 0.0

    source_text = " ".join(
        f"{s.source_title} {s.snippet}".lower() for s in proposal.sources
    )

    # Extract verifiable facts from new_content:
    # 1. Years (4-digit numbers that look like years)
    # 2. Specific numbers (with units or context)
    # 3. Proper nouns not in original
    new_lower = proposal.new_content.lower()
    old_lower = proposal.old_content.lower()

    # Extract years from new_content that aren't in old_content
    new_years = set(re.findall(r'\b(19\d{2}|20\d{2})\b', proposal.new_content))
    old_years = set(re.findall(r'\b(19\d{2}|20\d{2})\b', proposal.old_content))
    novel_years = new_years - old_years

    # Extract numbers with context (e.g., "400 small satellites", "$365 million")
    new_numbers = set(re.findall(r'\b(\d{2,})\b', proposal.new_content))
    old_numbers = set(re.findall(r'\b(\d{2,})\b', proposal.old_content))
    novel_numbers = new_numbers - old_numbers - new_years

    # Combine all verifiable facts
    all_facts = list(novel_years) + list(novel_numbers)
    if not all_facts:
        return 0.8  # No novel facts to verify — likely a qualitative rewrite, OK

    sourced_count = sum(1 for fact in all_facts if fact in source_text)
    return sourced_count / len(all_facts)


async def validate_node(state: AnalysisState) -> dict:
    """Stage 4: Quality validation — cross-reference sources, check confidence,
    reject fabricated content, and verify source-claim relevance."""
    logger.info("Orchestrator [validate]: validating %d proposals", len(state["proposals"]))
    await _update_analysis_progress(
        state, "Validating proposals", 85,
        f"Quality-checking {len(state['proposals'])} proposals against sources",
    )
    validated = []

    for proposal in state["proposals"]:
        # Skip proposals with no actual change
        if proposal.old_content.strip() == proposal.new_content.strip():
            logger.info("Skipping no-change proposal: %s", proposal.change_id)
            continue

        # ── Fabrication detection ─────────────────────────────────────
        fab_warnings = _detect_fabrication_signals(proposal)

        # ── Tense fabrication: "completed" claims sourced only as "planned" ──
        tense_warnings = _detect_tense_fabrication(proposal)
        if tense_warnings:
            logger.warning(
                "Tense fabrication in proposal %s: %s",
                proposal.change_id, "; ".join(tense_warnings[:2]),
            )
            fab_warnings.extend(tense_warnings)

        if len(fab_warnings) >= 3:
            logger.warning(
                "REJECTING proposal %s: %d fabrication signals detected: %s",
                proposal.change_id, len(fab_warnings), "; ".join(fab_warnings[:3]),
            )
            continue
        elif fab_warnings:
            logger.warning(
                "Fabrication signals in proposal %s (%d): %s — downgrading confidence",
                proposal.change_id, len(fab_warnings), "; ".join(fab_warnings),
            )
            proposal.confidence = ConfidenceLevel.LOW

        # ── Content-sourcing score: are facts in new_content backed by sources? ──
        sourcing_score = _compute_content_sourcing_score(proposal)
        if sourcing_score < 0.25:
            # Less than 25% of novel facts traceable to sources → reject
            logger.warning(
                "REJECTING proposal %s: sourcing score %.0f%% — most facts unsupported by sources",
                proposal.change_id, sourcing_score * 100,
            )
            continue
        elif sourcing_score < 0.50:
            # Less than 50% sourced → force LOW confidence
            logger.info(
                "Proposal %s: sourcing score %.0f%% — downgrading to LOW",
                proposal.change_id, sourcing_score * 100,
            )
            proposal.confidence = ConfidenceLevel.LOW

        # ── Source-claim relevance check ──────────────────────────────
        relevance_ratio = _check_source_relevance(proposal)
        if relevance_ratio < 0.2:
            logger.warning(
                "Proposal %s: only %.0f%% sources are content-relevant — downgrading to LOW",
                proposal.change_id, relevance_ratio * 100,
            )
            proposal.confidence = ConfidenceLevel.LOW
        elif relevance_ratio < 0.5:
            logger.info(
                "Proposal %s: %.0f%% sources content-relevant — capping at MEDIUM",
                proposal.change_id, relevance_ratio * 100,
            )
            if proposal.confidence == ConfidenceLevel.HIGH:
                proposal.confidence = ConfidenceLevel.MEDIUM

        # ── Cross-reference: upgrade confidence if authoritative sources agree ──
        gov_sources = sum(1 for s in proposal.sources if s.source_type == "government")
        academic_sources = sum(1 for s in proposal.sources if s.source_type == "academic")
        industry_sources = sum(1 for s in proposal.sources if s.source_type == "industry")
        technical_sources = sum(1 for s in proposal.sources if s.source_type == "technical")

        authoritative_count = gov_sources + academic_sources + industry_sources

        # Only upgrade if sources relevant, no fabrication, AND good sourcing score
        if relevance_ratio >= 0.5 and not fab_warnings and sourcing_score >= 0.50:
            if gov_sources >= 1 and (academic_sources >= 1 or industry_sources >= 1):
                proposal.confidence = ConfidenceLevel.HIGH
            elif authoritative_count >= 2:
                if proposal.confidence != ConfidenceLevel.HIGH:
                    proposal.confidence = ConfidenceLevel.MEDIUM  # Cap at medium, not auto-high
            elif authoritative_count >= 1 or technical_sources >= 2:
                if proposal.confidence == ConfidenceLevel.LOW:
                    proposal.confidence = ConfidenceLevel.MEDIUM

        # Verify sources exist (at least one must have a URL)
        valid_sources = [s for s in proposal.sources if s.source_url]
        if not valid_sources:
            logger.info("Skipping proposal with no valid source URLs: %s", proposal.change_id)
            continue

        validated.append(proposal)

    logger.info("Validation complete: %d/%d proposals passed", len(validated), len(state["proposals"]))
    await _update_analysis_progress(
        state, "Validation complete", 90,
        f"{len(validated)} proposals passed quality checks",
    )
    return {
        "validated_proposals": validated,
        "stage": "Validation complete",
        "progress": 90,
    }


# ------------------------------------------------------------------ #
# Routing                                                              #
# ------------------------------------------------------------------ #

def should_research(state: AnalysisState) -> str:
    """Route after analysis: research only if outdated claims exist."""
    outdated = [c for c in state.get("claims", []) if c.is_outdated]
    if outdated:
        return "research"
    return "complete"


def should_propose(state: AnalysisState) -> str:
    """Route after research: propose only if results found."""
    if state.get("research"):
        return "propose"
    return "complete"


# ------------------------------------------------------------------ #
# Graph                                                                #
# ------------------------------------------------------------------ #

def build_graph() -> StateGraph:
    graph = StateGraph(AnalysisState)

    graph.add_node("age_estimate", age_estimate_node)
    graph.add_node("style_analyze", style_analyze_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("research", research_node)
    graph.add_node("propose", propose_node)
    graph.add_node("validate", validate_node)

    graph.set_entry_point("age_estimate")
    graph.add_edge("age_estimate", "style_analyze")
    graph.add_edge("style_analyze", "analyze")

    graph.add_conditional_edges("analyze", should_research, {
        "research": "research",
        "complete": END,
    })
    graph.add_conditional_edges("research", should_propose, {
        "propose": "propose",
        "complete": END,
    })
    graph.add_edge("propose", "validate")
    graph.add_edge("validate", END)

    return graph.compile()


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

async def run_analysis(
    document_id: str,
    db,
    focus_areas: Optional[List[str]] = None,
) -> ChangeLog:
    """
    Run the full analysis pipeline for a document.
    Saves results to the changes and changelogs collections.

    Args:
        document_id: ID of the document to analyze
        db: Database connection
        focus_areas: Optional list of focus areas to filter claims.
                     Options: missions, constellations, technology, statistics,
                     companies, business_philosophy, historical_facts, all
                     Default: ["all"]
    """
    doc_repo = DocumentRepository(db)
    change_repo = ChangeRepository(db)

    # Load only the fields the pipeline needs (skip heavy figures/equations/tables)
    document = await doc_repo.find_by_id(document_id, analysis_mode=True)
    if not document:
        raise ValueError(f"Document {document_id} not found")

    text_content = document.get("text_content", "")
    if not text_content:
        raise ValueError(f"Document {document_id} has no text content — process it first")

    paragraphs = [p for p in text_content.split("\n") if p.strip()]

    # Load paragraph→page map built during document processing
    para_to_page = document.get("para_to_page", {})

    # Default focus areas
    if not focus_areas:
        focus_areas = ["all"]

    # Update document status and set initial progress immediately
    await doc_repo.update_fields(document_id, {
        "status": DocumentStatus.ANALYZING,
        "current_stage": "Initializing analysis pipeline",
        "progress": 2,
    })

    # Snapshot token counts before pipeline (agents are singletons, counters accumulate)
    _pre_analysis_prompt = content_agent.total_prompt_tokens
    _pre_analysis_completion = content_agent.total_completion_tokens
    _pre_update_prompt = update_agent.total_prompt_tokens
    _pre_update_completion = update_agent.total_completion_tokens

    # Build and run the LangGraph pipeline
    workflow = build_graph()

    initial_state: AnalysisState = {
        "document_id": document_id,
        "text_content": text_content,
        "paragraphs": paragraphs,
        "focus_areas": focus_areas,
        "estimated_pub_year": None,
        "document_age": None,
        "style_profile": None,
        "claims": [],
        "research": {},
        "proposals": [],
        "validated_proposals": [],
        "error": None,
        "stage": "Starting",
        "progress": 0,
        "db": db,
    }

    try:
        result = await workflow.ainvoke(initial_state)

        claims = result.get("claims", [])
        proposals = result.get("validated_proposals", []) or result.get("proposals", [])
        style_profile = result.get("style_profile")

        # ── Deduplicate proposals: keep only one per claim (highest confidence) ──
        # This ensures total_changes <= total_outdated
        if proposals:
            best_per_claim: dict[str, ChangeProposal] = {}
            confidence_rank = {ConfidenceLevel.HIGH: 3, ConfidenceLevel.MEDIUM: 2, ConfidenceLevel.LOW: 1}
            for p in proposals:
                existing = best_per_claim.get(p.claim_id)
                if existing is None or confidence_rank.get(p.confidence, 0) > confidence_rank.get(existing.confidence, 0):
                    best_per_claim[p.claim_id] = p
            if len(best_per_claim) < len(proposals):
                logger.info(
                    "Deduplicated proposals: %d → %d (one per claim)",
                    len(proposals), len(best_per_claim),
                )
            proposals = list(best_per_claim.values())

        # ── Diagnostic logging ────────────────────────────────────────
        outdated_count = sum(1 for c in claims if c.is_outdated)
        claims_size_bytes = sys.getsizeof(str([c.model_dump() for c in claims]))
        logger.info(
            "Pipeline results for %s: %d claims (%d outdated), %d proposals, "
            "style_profile=%s, claims_serialized_size=%d bytes",
            document_id, len(claims), outdated_count, len(proposals),
            "present" if style_profile else "NULL", claims_size_bytes,
        )

        if style_profile is None:
            logger.error(
                "style_profile is NULL after pipeline — style_analyze_node may have failed. "
                "Check OpenAI API key and response parsing."
            )

        # Resolve page numbers from the paragraph→page map
        if para_to_page:
            for claim in claims:
                if claim.page is None and claim.paragraph_idx is not None:
                    claim.page = para_to_page.get(str(claim.paragraph_idx))
            for proposal in proposals:
                if proposal.page is None and proposal.paragraph_idx is not None:
                    proposal.page = para_to_page.get(str(proposal.paragraph_idx))

        # Delete previous analysis results for this document
        await change_repo.delete_by_document(document_id)

        # Delete previous changelogs for this document so we don't return stale data
        await change_repo.delete_changelogs_by_document(document_id)

        # Save proposals to DB
        if proposals:
            await change_repo.create_many([p.model_dump() for p in proposals])

        # Build changelog — store claims as dicts for serialization
        claims_dicts = [c.model_dump() for c in claims]
        proposals_dicts = [p.model_dump() for p in proposals]

        # ── Capture token usage ─────────────────────────────────────────
        analysis_prompt = content_agent.total_prompt_tokens - _pre_analysis_prompt
        analysis_completion = content_agent.total_completion_tokens - _pre_analysis_completion
        update_prompt = update_agent.total_prompt_tokens - _pre_update_prompt
        update_completion = update_agent.total_completion_tokens - _pre_update_completion

        token_usage = {
            "analysis_prompt_tokens": analysis_prompt,
            "analysis_completion_tokens": analysis_completion,
            "update_prompt_tokens": update_prompt,
            "update_completion_tokens": update_completion,
            "total_prompt_tokens": analysis_prompt + update_prompt,
            "total_completion_tokens": analysis_completion + update_completion,
            "model": settings.GPT_MODEL,
        }
        logger.info(
            "Token usage for %s: prompt=%d, completion=%d",
            document_id,
            token_usage["total_prompt_tokens"],
            token_usage["total_completion_tokens"],
        )

        changelog_data = {
            "log_id": f"log_{uuid.uuid4().hex[:12]}",
            "document_id": document_id,
            "total_claims": len(claims),
            # Show outdated count = proposals count so the numbers are consistent
            # (outdated claims without research/proposals aren't actionable)
            "total_outdated": len(proposals),
            "total_changes": len(proposals),
            "claims": claims_dicts,
            "changes": proposals_dicts,
            "focus_areas": focus_areas,
            "style_profile": style_profile,
            "token_usage": token_usage,
            "created_at": datetime.utcnow(),
        }

        # Check serialized size before saving
        import json
        changelog_json = json.dumps(changelog_data, default=str)
        changelog_size_mb = len(changelog_json.encode("utf-8")) / (1024 * 1024)
        logger.info(
            "Changelog size for %s: %.2f MB (%d claims, %d changes)",
            document_id, changelog_size_mb, len(claims), len(proposals),
        )

        if changelog_size_mb > 15:
            # MongoDB 16MB limit — store claims separately if too large
            logger.warning(
                "Changelog too large (%.2f MB) — storing claims in separate collection",
                changelog_size_mb,
            )
            await change_repo.save_claims_batch(document_id, claims_dicts)
            changelog_data["claims"] = []  # Don't embed in changelog
            changelog_data["claims_stored_separately"] = True

        await change_repo.save_changelog(changelog_data)

        # Build the ChangeLog model for return value
        changelog = ChangeLog(
            log_id=changelog_data["log_id"],
            document_id=document_id,
            total_claims=len(claims),
            total_outdated=len(proposals),
            total_changes=len(proposals),
            claims=claims,
            changes=proposals,
            focus_areas=focus_areas,
            style_profile=style_profile,
            created_at=changelog_data["created_at"],
        )

        # Restore document status and final progress
        await doc_repo.update_fields(document_id, {
            "status": DocumentStatus.COMPLETED,
            "current_stage": "Analysis complete",
            "progress": 100,
        })
        entry = {
            "stage": "Analysis complete",
            "progress": 100,
            "timestamp": datetime.utcnow(),
            "message": f"Analysis complete: {len(claims)} claims, {len(proposals)} proposals (focus: {', '.join(focus_areas)})",
        }
        await doc_repo.push_history_entry(document_id, entry)

        logger.info(
            "Analysis pipeline complete for %s: %d claims, %d outdated, %d proposals (focus: %s)",
            document_id, len(claims), outdated_count, len(proposals), focus_areas,
        )
        return changelog

    except Exception as e:
        logger.error("Analysis pipeline failed for %s: %s", document_id, e, exc_info=True)
        await doc_repo.update_fields(document_id, {
            "status": DocumentStatus.ERROR,
            "error_message": f"Analysis failed: {str(e)}",
        })
        raise
