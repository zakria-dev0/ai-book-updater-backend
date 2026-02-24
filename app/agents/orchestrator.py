"""
LangGraph Orchestrator — coordinates the full AI analysis pipeline:
  analyze → research → propose → validate → complete
"""
import uuid
import asyncio
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


async def analyze_node(state: AnalysisState) -> dict:
    """Stage 1: Analyze document text for factual claims."""
    logger.info("Orchestrator [analyze]: starting for document %s", state["document_id"])
    await _update_analysis_progress(state, "Analyzing content", 10, "Scanning text for factual claims via GPT-4o")
    try:
        claims = await content_agent.analyze_document(
            document_id=state["document_id"],
            text_content=state["text_content"],
            paragraphs=state["paragraphs"],
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
    """Stage 3: Generate change proposals from research."""
    logger.info("Orchestrator [propose]: generating proposals")
    await _update_analysis_progress(
        state, "Generating proposals", 60,
        f"Generating change proposals for {len(state['research'])} researched claims",
    )
    try:
        proposals = await update_agent.generate_proposals(
            claims=state["claims"],
            research=state["research"],
            document_id=state["document_id"],
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


async def validate_node(state: AnalysisState) -> dict:
    """Stage 4: Quality validation — cross-reference sources, check confidence."""
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

        # Cross-reference: upgrade confidence if multiple authoritative sources agree
        gov_sources = sum(1 for s in proposal.sources if s.source_type == "government")
        academic_sources = sum(1 for s in proposal.sources if s.source_type == "academic")

        if gov_sources >= 1 and academic_sources >= 1:
            proposal.confidence = ConfidenceLevel.HIGH
        elif gov_sources >= 1 or academic_sources >= 2:
            proposal.confidence = ConfidenceLevel.HIGH
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

    graph.add_node("analyze", analyze_node)
    graph.add_node("research", research_node)
    graph.add_node("propose", propose_node)
    graph.add_node("validate", validate_node)

    graph.set_entry_point("analyze")

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

async def run_analysis(document_id: str, db) -> ChangeLog:
    """
    Run the full analysis pipeline for a document.
    Saves results to the changes and changelogs collections.
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

    # Update document status
    await doc_repo.update_fields(document_id, {"status": DocumentStatus.ANALYZING})

    # Build and run the LangGraph pipeline
    workflow = build_graph()

    initial_state: AnalysisState = {
        "document_id": document_id,
        "text_content": text_content,
        "paragraphs": paragraphs,
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

        # Delete previous analysis results for this document
        await change_repo.delete_by_document(document_id)

        # Save proposals to DB
        if proposals:
            await change_repo.create_many([p.model_dump() for p in proposals])

        # Build and save changelog
        changelog = ChangeLog(
            log_id=f"log_{uuid.uuid4().hex[:12]}",
            document_id=document_id,
            total_claims=len(claims),
            total_outdated=sum(1 for c in claims if c.is_outdated),
            total_changes=len(proposals),
            claims=claims,
            changes=proposals,
            created_at=datetime.utcnow(),
        )
        await change_repo.save_changelog(changelog.model_dump())

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
            "message": f"Analysis complete: {len(claims)} claims, {len(proposals)} proposals",
        }
        await doc_repo.push_history_entry(document_id, entry)

        logger.info(
            "Analysis pipeline complete for %s: %d claims, %d outdated, %d proposals",
            document_id, len(claims),
            sum(1 for c in claims if c.is_outdated),
            len(proposals),
        )
        return changelog

    except Exception as e:
        logger.error("Analysis pipeline failed for %s: %s", document_id, e)
        await doc_repo.update_fields(document_id, {
            "status": DocumentStatus.ERROR,
            "error_message": f"Analysis failed: {str(e)}",
        })
        raise
