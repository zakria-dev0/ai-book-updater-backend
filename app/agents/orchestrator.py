"""
LangGraph Orchestrator — coordinates the full AI analysis pipeline:
  style_analyze → analyze → research → propose → validate → complete

Enhanced with:
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

logger = get_logger(__name__)


# ------------------------------------------------------------------ #
# LangGraph State                                                      #
# ------------------------------------------------------------------ #

class AnalysisState(TypedDict):
    document_id: str
    text_content: str
    paragraphs: List[str]
    focus_areas: List[str]
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
    except Exception as e:
        logger.warning("Failed to update analysis progress: %s", e)


async def style_analyze_node(state: AnalysisState) -> dict:
    """Stage 0: Analyze writing style of the document (run once)."""
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
    logger.info(
        "Orchestrator [analyze]: starting for document %s (focus: %s)",
        state["document_id"], focus_areas,
    )
    await _update_analysis_progress(
        state, "Analyzing content", 10,
        f"Scanning text for factual claims via GPT-4o (focus: {', '.join(focus_areas)})",
    )
    try:
        claims = await content_agent.analyze_document(
            document_id=state["document_id"],
            text_content=state["text_content"],
            paragraphs=state["paragraphs"],
            focus_areas=focus_areas,
        )
        outdated_count = sum(1 for c in claims if c.is_outdated)
        await _update_analysis_progress(
            state, "Analysis complete", 30,
            f"Found {len(claims)} claims, {outdated_count} flagged outdated",
        )
        return {
            "claims": claims,
            "stage": "Analysis complete",
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
        state, "Researching claims", 35,
        f"Searching for updated information on {outdated_count} outdated claims",
    )
    try:
        research = await research_agent.research_claims(state["claims"])
        await _update_analysis_progress(
            state, "Research complete", 55,
            f"Found research results for {len(research)} claims",
        )
        return {
            "research": research,
            "stage": "Research complete",
            "progress": 55,
        }
    except Exception as e:
        logger.error("Research node failed: %s", e)
        return {"research": {}, "error": str(e), "stage": "Error", "progress": 0}


async def propose_node(state: AnalysisState) -> dict:
    """Stage 3: Generate style-matched change proposals from research."""
    logger.info("Orchestrator [propose]: generating proposals")
    await _update_analysis_progress(
        state, "Generating proposals", 60,
        f"Generating style-matched change proposals for {len(state['research'])} researched claims",
    )
    try:
        proposals = await update_agent.generate_proposals(
            claims=state["claims"],
            research=state["research"],
            document_id=state["document_id"],
            paragraphs=state.get("paragraphs", []),
        )
        await _update_analysis_progress(
            state, "Proposals generated", 75,
            f"Generated {len(proposals)} change proposals",
        )
        return {
            "proposals": proposals,
            "stage": "Proposals generated",
            "progress": 75,
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


async def validate_node(state: AnalysisState) -> dict:
    """Stage 4: Quality validation — cross-reference sources, check confidence,
    and verify source-claim relevance."""
    logger.info("Orchestrator [validate]: validating %d proposals", len(state["proposals"]))
    await _update_analysis_progress(
        state, "Validating proposals", 80,
        f"Quality-checking {len(state['proposals'])} proposals",
    )
    validated = []

    for proposal in state["proposals"]:
        # Skip proposals with no actual change
        if proposal.old_content.strip() == proposal.new_content.strip():
            logger.info("Skipping no-change proposal: %s", proposal.change_id)
            continue

        # ── Source-claim relevance check ──────────────────────────────
        relevance_ratio = _check_source_relevance(proposal)
        if relevance_ratio < 0.2:
            # Less than 20% of sources mention claim entities → downgrade to LOW
            logger.warning(
                "Proposal %s: only %.0f%% sources are content-relevant — downgrading to LOW",
                proposal.change_id, relevance_ratio * 100,
            )
            proposal.confidence = ConfidenceLevel.LOW
        elif relevance_ratio < 0.5:
            # Less than 50% relevant → cap at MEDIUM
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

        # Only upgrade confidence if sources are also content-relevant
        if relevance_ratio >= 0.5:
            if gov_sources >= 1 and (academic_sources >= 1 or industry_sources >= 1):
                proposal.confidence = ConfidenceLevel.HIGH
            elif authoritative_count >= 2:
                proposal.confidence = ConfidenceLevel.HIGH
            elif authoritative_count >= 1 or technical_sources >= 2:
                if proposal.confidence == ConfidenceLevel.LOW:
                    proposal.confidence = ConfidenceLevel.MEDIUM
            elif len(proposal.sources) >= 3:
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

    graph.add_node("style_analyze", style_analyze_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("research", research_node)
    graph.add_node("propose", propose_node)
    graph.add_node("validate", validate_node)

    graph.set_entry_point("style_analyze")
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

    # Load document text
    document = await doc_repo.find_by_id(document_id)
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

    # Update document status
    await doc_repo.update_fields(document_id, {"status": DocumentStatus.ANALYZING})

    # Build and run the LangGraph pipeline
    workflow = build_graph()

    initial_state: AnalysisState = {
        "document_id": document_id,
        "text_content": text_content,
        "paragraphs": paragraphs,
        "focus_areas": focus_areas,
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

        changelog_data = {
            "log_id": f"log_{uuid.uuid4().hex[:12]}",
            "document_id": document_id,
            "total_claims": len(claims),
            "total_outdated": outdated_count,
            "total_changes": len(proposals),
            "claims": claims_dicts,
            "changes": proposals_dicts,
            "focus_areas": focus_areas,
            "style_profile": style_profile,
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
            total_outdated=outdated_count,
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
