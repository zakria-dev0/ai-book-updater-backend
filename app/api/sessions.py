from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import uuid
import copy
import shutil
import os
import subprocess

from app.core.security import get_current_user_dep, verify_download_token, decode_token
from app.database.connection import get_database
from app.database.repositories.session_repo import SessionRepository
from app.database.repositories.document_repo import DocumentRepository
from app.models.session import (
    SessionStatus, EditorialRules, OutlineItem, IssueType, Severity,
    DiagnosticSummary, TriggerType, PatchStatus,
)
from app.core.logger import get_logger
from app.core.rate_limit import limiter
from app.core.config import settings

logger = get_logger(__name__)

router = APIRouter(prefix="/sessions", tags=["Editorial Pipeline"])


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_repos(db):
    return SessionRepository(db), DocumentRepository(db)


def _resolve_file_path(stored_path: str) -> str:
    """Resolve a file path from DB, falling back to local UPLOAD_DIR if the stored
    (production) path doesn't exist. This allows the same MongoDB to work on both
    the production server and a local dev machine."""
    if stored_path and os.path.exists(stored_path):
        return stored_path
    if stored_path:
        filename = os.path.basename(stored_path)
        local_path = os.path.join(settings.UPLOAD_DIR, filename)
        if os.path.exists(local_path):
            logger.info("Resolved production path to local: %s -> %s", stored_path, local_path)
            return local_path
        # Also try absolute path from UPLOAD_DIR
        abs_local = os.path.abspath(local_path)
        if os.path.exists(abs_local):
            logger.info("Resolved production path to local: %s -> %s", stored_path, abs_local)
            return abs_local
        logger.warning("File not found locally either: %s (tried %s)", stored_path, abs_local)
    return stored_path  # return original (will fail downstream with a clear error)


# ── Request / Response Models ────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    document_id: str


class CreateSessionResponse(BaseModel):
    session_id: str
    document_id: str
    status: str
    message: str


class RulesRequest(BaseModel):
    date_cutoff: Optional[str] = None
    preserve_sections: List[str] = []
    voice_preservation: bool = True
    citation_style: str = "inline"
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    allowed_source_types: List[str] = ["government", "academic", "news", "technical", "commercial"]
    excluded_topics: List[str] = []
    max_sentence_change_pct: float = Field(default=80.0, ge=0.0, le=100.0)


class OutlineConfirmRequest(BaseModel):
    selections: Dict[str, bool]  # {outline_item_id: in_scope}


class OpportunitySelectRequest(BaseModel):
    selections: Dict[str, bool]  # {opportunity_id: selected}


class PlanApproveRequest(BaseModel):
    plan_id: str
    approved: bool = True


class EvidenceDecisionRequest(BaseModel):
    evidence_id: str
    accepted: bool


class PatchReviewRequest(BaseModel):
    action: str  # "approve", "reject", "edit"
    editor_revision: Optional[str] = None


class DatedStatementResolveRequest(BaseModel):
    statement_id: str
    resolution: str  # "still_current", "flag_for_patch", "acceptable"


class ResetToStepRequest(BaseModel):
    target_status: str  # e.g. "created", "rules_confirmed", "outline_extracted", etc.


# ══════════════════════════════════════════════════════════════════════════════
# SESSION CRUD
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/", response_model=CreateSessionResponse)
async def create_session(
    req: CreateSessionRequest,
    user=Depends(get_current_user_dep),
):
    """Find existing active session or create a new one for a document."""
    db = get_database()
    session_repo, doc_repo = _get_repos(db)

    doc = await doc_repo.find_by_id(req.document_id, lightweight=True)
    if not doc:
        raise HTTPException(404, "Document not found")
    if doc.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")
    if doc.get("status") not in ("completed", "export_ready"):
        raise HTTPException(400, "Document must be processed before creating a session")

    # Req 5: Find existing session (not error) before creating new
    # Resume even "exported" sessions so user can navigate all completed stages
    existing_sessions = await session_repo.find_sessions_by_document(req.document_id)
    for s in existing_sessions:
        if s.get("status") != "error":
            logger.info("Resuming existing session %s for document %s (status=%s)", s["id"], req.document_id, s.get("status"))
            return CreateSessionResponse(
                session_id=s["id"],
                document_id=req.document_id,
                status=s["status"],
                message="Resuming existing session.",
            )

    session_data = {
        "document_id": req.document_id,
        "user_id": user["email"],
        "status": SessionStatus.CREATED.value,
        "rules": None,
        "outline": [],
        "diagnostic": None,
        "working_doc_path": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    session_id = await session_repo.create_session(session_data)

    return CreateSessionResponse(
        session_id=session_id,
        document_id=req.document_id,
        status=SessionStatus.CREATED.value,
        message="Editorial session created. Proceed to define rules.",
    )


@router.get("/{session_id}")
async def get_session(session_id: str, user=Depends(get_current_user_dep)):
    """Get session details including current status."""
    db = get_database()
    session_repo = SessionRepository(db)
    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")
    return session


@router.get("/document/{document_id}")
async def get_sessions_for_document(document_id: str, user=Depends(get_current_user_dep)):
    """List all sessions for a document."""
    db = get_database()
    session_repo = SessionRepository(db)
    sessions = await session_repo.find_sessions_by_document(document_id)
    # Filter to user's sessions unless admin
    if user.get("role") != "admin":
        sessions = [s for s in sessions if s.get("user_id") == user["email"]]
    return {"document_id": document_id, "sessions": sessions}


@router.get("/document/{document_id}/patches")
async def get_document_patches(document_id: str, user=Depends(get_current_user_dep)):
    """Get all patches across all sessions for a document, with session dates."""
    db = get_database()
    session_repo = SessionRepository(db)
    doc_repo = DocumentRepository(db)

    doc = await doc_repo.find_by_id(document_id, lightweight=True)
    if not doc:
        raise HTTPException(404, "Document not found")

    sessions = await session_repo.find_sessions_by_document(document_id)
    if user.get("role") != "admin":
        sessions = [s for s in sessions if s.get("user_id") == user["email"]]

    all_patches = []
    for sess in sessions:
        session_id = sess.get("id", "")
        patches = await session_repo.find_patches(session_id)
        for p in patches:
            p["session_created_at"] = sess.get("created_at")
            p["session_updated_at"] = sess.get("updated_at")
            p["session_status"] = sess.get("status")
        all_patches.extend(patches)

    return {
        "document_id": document_id,
        "document_title": doc.get("metadata", {}).get("title") or doc.get("original_filename", "Untitled"),
        "patches": all_patches,
        "total": len(all_patches),
        "approved": sum(1 for p in all_patches if p.get("status") in ("approved", "edited")),
        "rejected": sum(1 for p in all_patches if p.get("status") == "rejected"),
        "pending": sum(1 for p in all_patches if p.get("status") == "pending"),
    }


@router.delete("/{session_id}")
async def delete_session(session_id: str, user=Depends(get_current_user_dep)):
    """Delete a session and all related data."""
    db = get_database()
    session_repo = SessionRepository(db)
    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")
    await session_repo.delete_session(session_id)
    return {"message": "Session deleted", "session_id": session_id}


# ── Pipeline step order for reset logic ──────────────────────────────────────
_STATUS_ORDER = [
    SessionStatus.CREATED.value,
    SessionStatus.RULES_CONFIRMED.value,
    SessionStatus.OUTLINE_EXTRACTED.value,
    SessionStatus.DIAGNOSTIC_COMPLETE.value,
    SessionStatus.OPPORTUNITIES_SELECTED.value,
    SessionStatus.RESEARCH_PLANNED.value,
    SessionStatus.RESEARCHING.value,
    SessionStatus.RESEARCH_DONE.value,
    SessionStatus.EVIDENCE_REVIEWED.value,
    SessionStatus.PATCHES_GENERATED.value,
    SessionStatus.EDITS_APPLIED.value,
    SessionStatus.EXPORTED.value,
]


@router.post("/{session_id}/reset-to-step")
async def reset_to_step(
    session_id: str,
    req: ResetToStepRequest,
    user=Depends(get_current_user_dep),
):
    """Reset session to a previous step, clearing all downstream data."""
    db = get_database()
    session_repo = SessionRepository(db)
    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")

    target = req.target_status
    if target not in _STATUS_ORDER:
        raise HTTPException(400, f"Invalid target status: {target}")

    target_idx = _STATUS_ORDER.index(target)
    current_idx = _STATUS_ORDER.index(session["status"]) if session["status"] in _STATUS_ORDER else 0

    if target_idx > current_idx:
        raise HTTPException(400, "Cannot reset forward — only backward")

    # Clear downstream data based on target
    update_fields: dict = {"status": target}

    # If resetting to before diagnostic, clear diagnostic + everything downstream
    if target_idx < _STATUS_ORDER.index(SessionStatus.DIAGNOSTIC_COMPLETE.value):
        update_fields["diagnostic"] = None
        await session_repo.delete_opportunities(session_id)

    # If resetting to before research planning, clear plans + downstream
    if target_idx < _STATUS_ORDER.index(SessionStatus.RESEARCH_PLANNED.value):
        await session_repo.delete_research_plans(session_id)

    # If resetting to before evidence, clear evidence + downstream
    if target_idx < _STATUS_ORDER.index(SessionStatus.EVIDENCE_REVIEWED.value):
        await session_repo.delete_evidence_items(session_id)

    # If resetting to before patches, clear patches + downstream
    if target_idx < _STATUS_ORDER.index(SessionStatus.PATCHES_GENERATED.value):
        await session_repo.delete_patches(session_id)

    # If resetting to before apply, clear working doc
    if target_idx < _STATUS_ORDER.index(SessionStatus.EDITS_APPLIED.value):
        update_fields["working_doc_path"] = None

    await session_repo.update_session(session_id, update_fields)
    logger.info("Session %s reset to step: %s", session_id, target)
    return {"message": f"Session reset to {target}", "session_id": session_id, "status": target}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: RULES CONFIRMATION
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/{session_id}/rules")
async def set_rules(
    session_id: str,
    req: RulesRequest,
    user=Depends(get_current_user_dep),
):
    """Define editorial rules for the session."""
    db = get_database()
    session_repo = SessionRepository(db)
    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")

    rules = req.model_dump()
    await session_repo.update_session(session_id, {
        "rules": rules,
        "status": SessionStatus.RULES_CONFIRMED.value,
    })

    return {
        "session_id": session_id,
        "status": SessionStatus.RULES_CONFIRMED.value,
        "rules": rules,
        "message": "Rules confirmed. Proceed to extract outline.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: OUTLINE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/{session_id}/extract-outline")
async def extract_outline(
    session_id: str,
    user=Depends(get_current_user_dep),
):
    """Extract document outline (headings) for scope selection."""
    db = get_database()
    session_repo = SessionRepository(db)
    doc_repo = DocumentRepository(db)

    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")
    # Allow re-running if session is at or past rules_confirmed
    current_status = session.get("status")
    if current_status in _STATUS_ORDER:
        if _STATUS_ORDER.index(current_status) < _STATUS_ORDER.index(SessionStatus.RULES_CONFIRMED.value):
            raise HTTPException(400, "Rules must be confirmed first")
    else:
        raise HTTPException(400, "Rules must be confirmed first")

    doc = await doc_repo.find_by_id(session["document_id"])
    if not doc:
        raise HTTPException(404, "Document not found")

    # Extract headings from DOCX using python-docx
    file_path = _resolve_file_path(doc.get("file_path", ""))
    outline_items = []

    try:
        import re as _re
        from docx import Document as DocxDocument
        docx_doc = DocxDocument(file_path)

        for idx, para in enumerate(docx_doc.paragraphs):
            style_name = para.style.name if para.style else ""
            text = para.text.strip()
            if not text:
                continue

            # Skip headers/footers/captions
            style_lower = style_name.lower()
            if any(skip in style_lower for skip in ("header", "footer", "caption", "toc")):
                continue

            # Detect heading styles: "Heading 1", "Heading #1", "heading1", etc.
            heading_match = _re.match(r"[Hh]eading\s*#?\s*(\d+)", style_name)
            if heading_match:
                level = int(heading_match.group(1))
                outline_items.append({
                    "id": str(uuid.uuid4())[:8],
                    "text": text,
                    "level": level,
                    "in_scope": True,
                    "paragraph_index": idx,
                })
                continue

            # Detect numbered section headings in body text (e.g. "2.1 Early Space Explorers")
            numbered_heading = _re.match(r"^(\d+(?:\.\d+)*)\s+[A-Z]", text)
            if numbered_heading and len(text) < 120:
                depth = numbered_heading.group(1).count(".") + 1
                # Only count as heading if the paragraph is short (not a full body paragraph)
                if len(text) < 80 or all(r.bold for r in para.runs if r.text.strip()):
                    outline_items.append({
                        "id": str(uuid.uuid4())[:8],
                        "text": text,
                        "level": depth,
                        "in_scope": True,
                        "paragraph_index": idx,
                    })

    except Exception as e:
        logger.error("Outline extraction failed: %s", e)
        # Fallback: try to parse from text_content using regex
        import re
        text_content = doc.get("text_content", "")
        lines = text_content.split("\n")
        for idx, line in enumerate(lines):
            line_stripped = line.strip()
            # Detect numbered headings or all-caps lines
            if re.match(r"^\d+(\.\d+)*\s+[A-Z]", line_stripped) or (
                line_stripped.isupper() and 3 < len(line_stripped) < 100
            ):
                outline_items.append({
                    "id": str(uuid.uuid4())[:8],
                    "text": line_stripped,
                    "level": 1 if re.match(r"^\d+\s", line_stripped) else 2,
                    "in_scope": True,
                    "paragraph_index": idx,
                })

    await session_repo.update_session(session_id, {
        "outline": outline_items,
        "status": SessionStatus.OUTLINE_EXTRACTED.value,
    })

    return {
        "session_id": session_id,
        "status": SessionStatus.OUTLINE_EXTRACTED.value,
        "outline": outline_items,
        "total_sections": len(outline_items),
        "message": "Outline extracted. Select sections to analyze.",
    }


@router.put("/{session_id}/outline")
async def confirm_outline(
    session_id: str,
    req: OutlineConfirmRequest,
    user=Depends(get_current_user_dep),
):
    """Confirm which sections are in scope."""
    db = get_database()
    session_repo = SessionRepository(db)
    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")

    outline = session.get("outline", [])
    for item in outline:
        if item["id"] in req.selections:
            item["in_scope"] = req.selections[item["id"]]

    await session_repo.update_session(session_id, {"outline": outline})

    in_scope = sum(1 for i in outline if i["in_scope"])
    return {
        "session_id": session_id,
        "in_scope_count": in_scope,
        "total_sections": len(outline),
        "message": f"{in_scope} sections selected for analysis.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: DIAGNOSTIC REVIEW
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/{session_id}/run-diagnostic")
async def run_diagnostic(
    session_id: str,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user_dep),
):
    """Run AI diagnostic to identify issues in selected sections."""
    db = get_database()
    session_repo = SessionRepository(db)
    doc_repo = DocumentRepository(db)

    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")
    # Allow re-running if session is at or past outline_extracted
    current_status = session.get("status")
    if current_status in _STATUS_ORDER:
        if _STATUS_ORDER.index(current_status) < _STATUS_ORDER.index(SessionStatus.OUTLINE_EXTRACTED.value):
            raise HTTPException(400, "Outline must be extracted first")
    else:
        raise HTTPException(400, "Outline must be extracted first")

    doc = await doc_repo.find_by_id(session["document_id"], analysis_mode=True)
    if not doc:
        raise HTTPException(404, "Document not found")

    background_tasks.add_task(
        _run_diagnostic_task, session_id, session, doc
    )

    return {
        "session_id": session_id,
        "status": "running",
        "message": "Diagnostic analysis started. Poll session status for updates.",
    }


async def _run_diagnostic_task(session_id: str, session: dict, doc: dict):
    """Background task: scan each in-scope section individually for thorough coverage."""
    from openai import AsyncOpenAI
    import json
    import asyncio

    db = get_database()
    session_repo = SessionRepository(db)

    try:
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        rules = session.get("rules", {}) or {}
        outline = session.get("outline", [])
        text_content = doc.get("text_content", "")

        # Get in-scope section texts
        in_scope_headings = [o for o in outline if o.get("in_scope", True)]

        # Build individual section texts from paragraph indices
        paragraphs = text_content.split("\n")
        section_texts = []
        for i, heading in enumerate(in_scope_headings):
            start = heading.get("paragraph_index", 0)
            if i + 1 < len(in_scope_headings):
                end = in_scope_headings[i + 1].get("paragraph_index", len(paragraphs))
            else:
                end = len(paragraphs)
            section_text = "\n".join(paragraphs[start:end]).strip()
            if len(section_text) > 50:  # skip tiny/empty sections
                section_texts.append({
                    "section": heading.get("text", ""),
                    "text": section_text[:8000],
                })

        # Build rule instructions
        confidence_threshold = rules.get("confidence_threshold", 0.5)

        rule_lines = []
        if rules.get("date_cutoff"):
            rule_lines.append(f"- Flag facts that may be outdated relative to {rules['date_cutoff']}.")
        else:
            rule_lines.append("- Flag any facts, statistics, figures, or claims that may be outdated or no longer accurate.")
        if rules.get("excluded_topics"):
            rule_lines.append(f"- Do NOT flag issues related to these topics: {', '.join(rules['excluded_topics'])}.")
        if rules.get("preserve_sections"):
            rule_lines.append(f"- Do NOT flag issues in these sections: {', '.join(rules['preserve_sections'])}.")
        rules_block = "\n".join(rule_lines)

        system_msg = "You are a meticulous document auditor. Your job is to find EVERY sentence that contains potentially outdated information, stale references, temporal language, or changed real-world status. Be extremely thorough — missing an issue is worse than flagging one that turns out to be fine. Return only valid JSON."

        # Delete old opportunities for re-run
        await session_repo.delete_opportunities(session_id)

        # ── Scan each section with its own AI call ──
        async def scan_section(section_data: dict) -> list:
            section_name = section_data["section"]
            section_text = section_data["text"]

            prompt = f"""Audit this section of a document for content that may need updating.

Section: "{section_name}"

Rules:
{rules_block}

Read every sentence below carefully and flag ALL of the following:
1. **Outdated facts** — statistics, numbers, figures, counts, mission outcomes, technology specs, organizational names, or claims that may have changed since the text was written.
2. **Broken or stale citations** — references to URLs, reports, or sources that may no longer exist or have been superseded.
3. **Date references** — explicit dates ("as of 2018", "in 2015", "planned for 2020") and temporal language ("currently", "recently", "upcoming", "today", "now") that may now be inaccurate.
4. **Changed status** — anything described as "planned", "proposed", "under development", "upcoming", "new", "latest" that may have since launched, been completed, failed, been cancelled, merged, shut down, or otherwise changed.

For EACH issue found, return a JSON object:
- "sentence": the EXACT sentence from the text (copy it word-for-word, do not paraphrase)
- "section_ref": "{section_name}"
- "issue_type": one of "outdated_fact", "broken_citation", "date_reference", "changed_status"
- "severity": "high" (very likely wrong/outdated), "medium" (probably needs checking), or "low" (minor/cosmetic)
- "brief_reason": one-line explanation of why this may need updating
- "confidence": float 0.0-1.0

Only include issues with confidence >= {confidence_threshold}.
Be thorough. Check EVERY sentence for dates, numbers, statistics, status words, and temporal language.
Return ONLY a valid JSON array. No markdown, no explanation. If no issues found, return [].

Text:
{section_text}"""

            try:
                response = await client.chat.completions.create(
                    model=settings.GPT_MODEL,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=4000,
                )
                raw = response.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
                return json.loads(raw)
            except Exception as e:
                logger.warning("Diagnostic failed for section '%s': %s", section_name, e)
                return []

        # Run sections in parallel batches of 3 to avoid rate limits
        all_issues = []
        batch_size = 3
        for batch_start in range(0, len(section_texts), batch_size):
            batch = section_texts[batch_start:batch_start + batch_size]
            results = await asyncio.gather(*[scan_section(s) for s in batch])
            for issues_list in results:
                if isinstance(issues_list, list):
                    all_issues.extend(issues_list)

        # Deduplicate by sentence text
        seen_sentences = set()
        opportunities = []
        for issue in all_issues:
            sentence = issue.get("sentence", "").strip()
            if not sentence or sentence in seen_sentences:
                continue
            conf = issue.get("confidence", 0.5)
            if conf < confidence_threshold:
                continue
            seen_sentences.add(sentence)
            opportunities.append({
                "opportunity_id": str(uuid.uuid4())[:8],
                "session_id": session_id,
                "section_ref": issue.get("section_ref", ""),
                "original_sentence": sentence,
                "issue_type": issue.get("issue_type", "outdated_fact"),
                "severity": issue.get("severity", "medium"),
                "confidence": conf,
                "brief_reason": issue.get("brief_reason", ""),
                "selected": False,
            })

        await session_repo.create_opportunities(opportunities)

        # Build summary
        high = sum(1 for o in opportunities if o["severity"] == "high")
        medium = sum(1 for o in opportunities if o["severity"] == "medium")
        low = sum(1 for o in opportunities if o["severity"] == "low")
        by_type = {}
        for o in opportunities:
            t = o["issue_type"]
            by_type[t] = by_type.get(t, 0) + 1

        diagnostic = {
            "total_issues": len(opportunities),
            "high_count": high,
            "medium_count": medium,
            "low_count": low,
            "by_type": by_type,
        }

        await session_repo.update_session(session_id, {
            "diagnostic": diagnostic,
            "status": SessionStatus.DIAGNOSTIC_COMPLETE.value,
        })

        logger.info("Diagnostic complete for session %s: %d issues found across %d sections",
                     session_id, len(opportunities), len(section_texts))

    except Exception as e:
        logger.error("Diagnostic failed for session %s: %s", session_id, e)
        await session_repo.update_session(session_id, {
            "status": SessionStatus.ERROR.value,
            "error_message": str(e),
        })


@router.get("/{session_id}/diagnostic")
async def get_diagnostic(session_id: str, user=Depends(get_current_user_dep)):
    """Get diagnostic summary and opportunities."""
    db = get_database()
    session_repo = SessionRepository(db)
    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")

    opportunities = await session_repo.find_opportunities(session_id)
    return {
        "session_id": session_id,
        "status": session.get("status"),
        "diagnostic": session.get("diagnostic"),
        "opportunities": opportunities,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4: OPPORTUNITY SELECTION
# ══════════════════════════════════════════════════════════════════════════════

@router.put("/{session_id}/opportunities")
async def select_opportunities(
    session_id: str,
    req: OpportunitySelectRequest,
    user=Depends(get_current_user_dep),
):
    """Select which opportunities to pursue."""
    db = get_database()
    session_repo = SessionRepository(db)
    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")

    total = await session_repo.update_opportunity_selections(session_id, req.selections)

    selected_count = sum(1 for v in req.selections.values() if v)
    if selected_count == 0:
        raise HTTPException(400, "At least one opportunity must be selected")

    await session_repo.update_session(session_id, {
        "status": SessionStatus.OPPORTUNITIES_SELECTED.value,
    })

    return {
        "session_id": session_id,
        "status": SessionStatus.OPPORTUNITIES_SELECTED.value,
        "updated": total,
        "selected_count": selected_count,
        "message": f"{selected_count} opportunities selected. Proceed to research planning.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5: RESEARCH PLANNING
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/{session_id}/plan-research")
async def plan_research(
    session_id: str,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user_dep),
):
    """AI generates research plans for each selected opportunity."""
    db = get_database()
    session_repo = SessionRepository(db)
    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")
    # Allow re-running if session is at or past opportunities_selected
    current_status = session.get("status")
    if current_status in _STATUS_ORDER:
        if _STATUS_ORDER.index(current_status) < _STATUS_ORDER.index(SessionStatus.OPPORTUNITIES_SELECTED.value):
            raise HTTPException(400, "Opportunities must be selected first")
    else:
        raise HTTPException(400, "Opportunities must be selected first")

    background_tasks.add_task(_plan_research_task, session_id, session)

    return {
        "session_id": session_id,
        "status": "planning",
        "message": "Research planning started. Poll session for updates.",
    }


async def _plan_research_task(session_id: str, session: dict):
    """Background task: generate research plans for selected opportunities."""
    from openai import AsyncOpenAI
    import json

    db = get_database()
    session_repo = SessionRepository(db)

    try:
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        rules = session.get("rules", {}) or {}
        allowed_sources = rules.get("allowed_source_types", ["government", "academic", "news", "technical", "commercial"])

        selected = await session_repo.find_selected_opportunities(session_id)
        if not selected:
            await session_repo.update_session(session_id, {
                "status": SessionStatus.ERROR.value,
                "error_message": "No opportunities selected",
            })
            return

        # Delete old plans for re-run
        await session_repo.delete_research_plans(session_id)

        plans = []
        for opp in selected:
            opp_id = opp.get("opportunity_id", opp.get("id", ""))
            prompt = f"""Given this outdated claim from a technical document:
Claim: "{opp.get('original_sentence', '')}"
Issue: {opp.get('brief_reason', '')}
Allowed source types: {', '.join(allowed_sources)}

Generate a research plan to verify/update this claim. Return JSON with:
- "search_queries": list of 2-4 specific search queries
- "target_sources": list of specific domain names or organizations to search
- "facts_to_verify": list of specific factual claims to check

Return ONLY valid JSON. No other text."""

            response = await client.chat.completions.create(
                model=settings.GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a research planner for technical document updates. Generate focused, efficient research plans."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            plan_data = json.loads(raw)

            plans.append({
                "plan_id": str(uuid.uuid4())[:8],
                "opportunity_id": opp_id,
                "session_id": session_id,
                "search_queries": plan_data.get("search_queries", []),
                "target_sources": plan_data.get("target_sources", []),
                "facts_to_verify": plan_data.get("facts_to_verify", []),
                "approved": False,
            })

        await session_repo.create_research_plans(plans)
        await session_repo.update_session(session_id, {
            "status": SessionStatus.RESEARCH_PLANNED.value,
        })

        logger.info("Research planning complete for session %s: %d plans", session_id, len(plans))

    except Exception as e:
        logger.error("Research planning failed for session %s: %s", session_id, e)
        await session_repo.update_session(session_id, {
            "status": SessionStatus.ERROR.value,
            "error_message": str(e),
        })


@router.get("/{session_id}/research-plans")
async def get_research_plans(session_id: str, user=Depends(get_current_user_dep)):
    """Get all research plans for a session."""
    db = get_database()
    session_repo = SessionRepository(db)
    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")

    plans = await session_repo.find_research_plans(session_id)
    opportunities = await session_repo.find_selected_opportunities(session_id)

    # Build lookup for opportunity info
    opp_map = {}
    for o in opportunities:
        oid = o.get("opportunity_id", o.get("id", ""))
        opp_map[oid] = o

    # Enrich plans with opportunity data
    enriched = []
    for p in plans:
        p["opportunity"] = opp_map.get(p.get("opportunity_id", ""), {})
        enriched.append(p)

    return {
        "session_id": session_id,
        "plans": enriched,
        "total": len(enriched),
    }


@router.put("/{session_id}/research-plans/{plan_id}/approve")
async def approve_plan(
    session_id: str,
    plan_id: str,
    req: PlanApproveRequest,
    user=Depends(get_current_user_dep),
):
    """Approve or reject a research plan."""
    db = get_database()
    session_repo = SessionRepository(db)
    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    updated = await session_repo.approve_research_plan(plan_id, req.approved)
    return {"plan_id": plan_id, "approved": req.approved, "updated": updated}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6: EVIDENCE REVIEW (run research + review)
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/{session_id}/run-research")
async def run_research(
    session_id: str,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user_dep),
):
    """Run research for all approved plans using Tavily API."""
    db = get_database()
    session_repo = SessionRepository(db)
    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")

    approved_plans = await session_repo.find_approved_plans(session_id)
    if not approved_plans:
        raise HTTPException(400, "No approved research plans. Approve at least one plan first.")
    logger.info("Starting research for session %s: %d approved plans", session_id, len(approved_plans))
    for p in approved_plans[:3]:
        logger.info("  Plan %s queries: %s", p.get("plan_id", "?"), p.get("search_queries", []))

    # Set status to RESEARCHING immediately
    await session_repo.update_session(session_id, {"status": SessionStatus.RESEARCHING.value})

    background_tasks.add_task(_run_research_task, session_id, session, approved_plans)

    return {
        "session_id": session_id,
        "status": "researching",
        "plans_to_run": len(approved_plans),
        "message": "Research started. Poll session for updates.",
    }


async def _run_research_task(session_id: str, session: dict, approved_plans: list):
    """Background task: fast, focused research — capped plans, high parallelism, incremental saves."""
    import asyncio
    import httpx

    db = get_database()
    session_repo = SessionRepository(db)

    try:
        # Delete old evidence for re-run
        await session_repo.delete_evidence_items(session_id)

        # Cap at 20 plans max to keep research fast
        plans_to_run = approved_plans[:20]
        if len(approved_plans) > 20:
            logger.info("Capping research from %d to 20 plans for session %s",
                        len(approved_plans), session_id)

        # Pre-fetch all opportunities in one go to avoid N+1 DB queries
        opp_ids = list(set(p.get("opportunity_id", "") for p in plans_to_run if p.get("opportunity_id")))
        all_opps = {}
        for opp_id in opp_ids:
            opp = await session_repo.find_opportunity(opp_id)
            if opp:
                all_opps[opp_id] = opp

        async def research_plan(plan: dict, http_client: httpx.AsyncClient) -> list:
            """Research a single plan: 1 focused Tavily search, keep top 2 results."""
            plan_id = plan.get("plan_id", plan.get("id", ""))
            queries = plan.get("search_queries", [])
            target_sources = plan.get("target_sources", [])

            opp = all_opps.get(plan.get("opportunity_id", ""))
            original_sentence = opp.get("original_sentence", "") if opp else ""

            evidence = []
            seen_urls = set()

            # Use only the FIRST query (most targeted)
            query = queries[0] if queries else ""
            if not query:
                logger.warning("Plan %s has no search queries, skipping", plan_id)
                return evidence

            try:
                logger.info("Tavily search for plan %s: %s", plan_id, query[:80])
                payload = {
                    "api_key": settings.TAVILY_API_KEY,
                    "query": query,
                    "max_results": 2,
                    "search_depth": "basic",
                }
                # Only add include_domains if we have actual domains
                if target_sources:
                    payload["include_domains"] = target_sources[:3]

                tavily_resp = await http_client.post(
                    "https://api.tavily.com/search",
                    json=payload,
                    timeout=15,
                )
                if tavily_resp.status_code != 200:
                    err_body = tavily_resp.text[:300]
                    logger.warning("Tavily returned %d for plan %s: %s", tavily_resp.status_code, plan_id, err_body)
                    return evidence
                results = tavily_resp.json().get("results", [])
                logger.info("Tavily returned %d results for plan %s", len(results), plan_id)

                for r in results[:2]:
                    url = r.get("url", "")
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    tavily_score = r.get("score", 0.5)
                    excerpt = r.get("content", "")[:800]
                    claim_words = set(original_sentence.lower().split())
                    excerpt_words = set(excerpt.lower().split())
                    overlap = len(claim_words & excerpt_words)
                    keyword_boost = min(overlap / max(len(claim_words), 1), 0.3)
                    relevance = round(min(tavily_score + keyword_boost, 1.0), 2)

                    evidence.append({
                        "evidence_id": str(uuid.uuid4())[:8],
                        "research_plan_id": plan_id,
                        "session_id": session_id,
                        "source_url": url,
                        "source_title": r.get("title", ""),
                        "excerpt": excerpt,
                        "relevance_score": relevance,
                        "accepted": None,
                    })

            except Exception as e:
                logger.warning("Research query failed for plan %s: %s", plan_id, e)

            return evidence

        # Run in parallel batches of 10 — save after each batch for incremental progress
        total_evidence = 0
        async with httpx.AsyncClient() as http_client:
            batch_size = 10
            for batch_start in range(0, len(plans_to_run), batch_size):
                batch = plans_to_run[batch_start:batch_start + batch_size]
                results = await asyncio.gather(
                    *[research_plan(p, http_client) for p in batch]
                )
                batch_evidence = []
                for evidence_list in results:
                    batch_evidence.extend(evidence_list)

                # Save this batch immediately so the frontend sees progress
                if batch_evidence:
                    await session_repo.create_evidence_items(batch_evidence)
                    total_evidence += len(batch_evidence)
                    logger.info("Research batch saved: %d items (total %d) for session %s",
                                len(batch_evidence), total_evidence, session_id)

        # Mark research as done so frontend stops polling
        await session_repo.update_session(session_id, {"status": SessionStatus.RESEARCH_DONE.value})
        logger.info("Research complete for session %s: %d evidence items", session_id, total_evidence)

    except Exception as e:
        logger.error("Research failed for session %s: %s", session_id, e)
        await session_repo.update_session(session_id, {
            "status": SessionStatus.ERROR.value,
            "error_message": str(e),
        })


@router.get("/{session_id}/evidence")
async def get_evidence(session_id: str, user=Depends(get_current_user_dep)):
    """Get all evidence items grouped by research plan."""
    db = get_database()
    session_repo = SessionRepository(db)
    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")

    evidence = await session_repo.find_evidence_items(session_id)
    plans = await session_repo.find_research_plans(session_id)
    opportunities = await session_repo.find_selected_opportunities(session_id)

    # Build lookups
    plan_map = {p.get("plan_id", p.get("id", "")): p for p in plans}
    opp_map = {o.get("opportunity_id", o.get("id", "")): o for o in opportunities}

    # Group evidence by opportunity
    grouped = {}
    for e in evidence:
        plan = plan_map.get(e.get("research_plan_id", ""), {})
        opp_id = plan.get("opportunity_id", "unknown")
        if opp_id not in grouped:
            grouped[opp_id] = {
                "opportunity": opp_map.get(opp_id, {}),
                "evidence": [],
            }
        grouped[opp_id]["evidence"].append(e)

    return {
        "session_id": session_id,
        "evidence_groups": list(grouped.values()),
        "total_evidence": len(evidence),
        "decided": sum(1 for e in evidence if e.get("accepted") is not None),
        "undecided": sum(1 for e in evidence if e.get("accepted") is None),
    }


@router.put("/{session_id}/evidence/{evidence_id}")
async def decide_evidence(
    session_id: str,
    evidence_id: str,
    req: EvidenceDecisionRequest,
    user=Depends(get_current_user_dep),
):
    """Accept or reject an evidence item."""
    db = get_database()
    session_repo = SessionRepository(db)
    updated = await session_repo.decide_evidence(evidence_id, req.accepted)

    # Check if all evidence has been decided
    evidence = await session_repo.find_evidence_items(session_id)
    all_decided = all(e.get("accepted") is not None for e in evidence)

    if all_decided and evidence:
        await session_repo.update_session(session_id, {
            "status": SessionStatus.EVIDENCE_REVIEWED.value,
        })

    return {
        "evidence_id": evidence_id,
        "accepted": req.accepted,
        "updated": updated,
        "all_decided": all_decided,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7: PATCH GENERATION & APPROVAL
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/{session_id}/generate-patches")
async def generate_patches(
    session_id: str,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user_dep),
):
    """Generate replacement patches using accepted evidence."""
    db = get_database()
    session_repo = SessionRepository(db)
    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")

    background_tasks.add_task(_generate_patches_task, session_id, session)

    return {
        "session_id": session_id,
        "status": "generating_patches",
        "message": "Patch generation started. Poll session for updates.",
    }


async def _generate_patches_task(session_id: str, session: dict):
    """Background task: generate sentence-level patches using AI."""
    from openai import AsyncOpenAI
    import json

    db = get_database()
    session_repo = SessionRepository(db)

    try:
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        rules = session.get("rules", {}) or {}
        max_change_pct = rules.get("max_sentence_change_pct", 80.0)
        citation_style = rules.get("citation_style", "inline")
        voice_preservation = rules.get("voice_preservation", True)

        selected = await session_repo.find_selected_opportunities(session_id)

        # Delete old patches for re-run
        await session_repo.delete_patches(session_id)

        logger.info("Generating patches for %d selected opportunities (parallel)", len(selected))

        # Note: when diagnostic re-runs, opportunities get new IDs but research plans
        # still reference old IDs. The fallback pool above handles this gracefully.

        # Pre-fetch ALL accepted evidence for the session as fallback
        # (needed when diagnostic re-run creates new opportunity IDs that don't match plan IDs)
        all_session_evidence = await session_repo.find_evidence_items(session_id)
        all_accepted_evidence = [e for e in all_session_evidence if e.get("accepted") is True]
        logger.info("Session has %d accepted evidence items (fallback pool)", len(all_accepted_evidence))

        async def _generate_one_patch(opp):
            """Generate a single patch for one opportunity — runs in parallel."""
            opp_id = opp.get("opportunity_id", opp.get("id", ""))
            accepted_evidence = await session_repo.find_accepted_evidence_for_opportunity(opp_id)

            if not accepted_evidence and all_accepted_evidence:
                # Fallback: use all accepted session evidence when opp→plan chain is broken
                accepted_evidence = all_accepted_evidence
                logger.info("  Patch gen opp %s: using %d session-level accepted evidence (fallback)", opp_id, len(accepted_evidence))

            if not accepted_evidence:
                logger.warning("  Patch gen opp %s: SKIPPED — no accepted evidence", opp_id)
                return None  # skip opportunities with no accepted evidence

            evidence_text = "\n".join(
                f"- {e.get('source_title', '')}: {e.get('excerpt', '')[:300]}"
                for e in accepted_evidence
            )

            style_instructions = ""
            if voice_preservation:
                style_instructions = "IMPORTANT: Preserve the original author's writing style, tone, and voice. "
            style_instructions += f"Citation style: {citation_style}. "
            style_instructions += f"Maximum change: {max_change_pct}% of the original sentence. "

            prompt = f"""Given this original sentence from a technical document and the research evidence, write a replacement sentence.

Original: "{opp.get('original_sentence', '')}"
Section: {opp.get('section_ref', '')}
Issue: {opp.get('brief_reason', '')}

Evidence:
{evidence_text}

{style_instructions}

Return ONLY valid JSON with:
- "revised_sentence": the replacement text
- "citation": proper citation for the source used
- "rationale": brief explanation of why this change is needed
- "confidence": float 0.0-1.0
- "change_pct": estimated percentage of the sentence that changed

Return ONLY valid JSON. No other text."""

            try:
                response = await client.chat.completions.create(
                    model=settings.GPT_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a precise technical editor. Write concise, accurate replacement sentences that update outdated information while preserving the document's style."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=1000,
                )

                raw = response.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
                patch_data = json.loads(raw)

                # Enforce max change percentage
                change_pct = patch_data.get("change_pct", 50)
                if change_pct > max_change_pct:
                    return None  # skip patches that change too much

                return {
                    "patch_id": str(uuid.uuid4())[:8],
                    "opportunity_id": opp_id,
                    "session_id": session_id,
                    "original_sentence": opp.get("original_sentence", ""),
                    "revised_sentence": patch_data.get("revised_sentence", ""),
                    "citation": patch_data.get("citation", ""),
                    "rationale": patch_data.get("rationale", ""),
                    "confidence": patch_data.get("confidence", 0.5),
                    "change_pct": change_pct,
                    "status": PatchStatus.PENDING.value,
                    "editor_revision": None,
                    "reviewed_at": None,
                    "section_ref": opp.get("section_ref", ""),
                }
            except Exception as e:
                logger.warning("Patch generation failed for opp %s: %s", opp_id, e)
                return None

        # Run all OpenAI calls in parallel
        results = await asyncio.gather(*[_generate_one_patch(opp) for opp in selected])
        patches = [p for p in results if p is not None]

        await session_repo.create_patches(patches)
        await session_repo.update_session(session_id, {
            "status": SessionStatus.PATCHES_GENERATED.value,
        })

        logger.info("Patch generation complete for session %s: %d patches", session_id, len(patches))

    except Exception as e:
        logger.error("Patch generation failed for session %s: %s", session_id, e)
        await session_repo.update_session(session_id, {
            "status": SessionStatus.ERROR.value,
            "error_message": str(e),
        })


@router.get("/{session_id}/patches")
async def get_patches(session_id: str, user=Depends(get_current_user_dep)):
    """Get all patches for review."""
    db = get_database()
    session_repo = SessionRepository(db)
    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")

    patches = await session_repo.find_patches(session_id)
    return {
        "session_id": session_id,
        "patches": patches,
        "total": len(patches),
        "pending": sum(1 for p in patches if p.get("status") == "pending"),
        "approved": sum(1 for p in patches if p.get("status") in ("approved", "edited")),
        "rejected": sum(1 for p in patches if p.get("status") == "rejected"),
    }


@router.put("/{session_id}/patches/{patch_id}")
async def review_patch(
    session_id: str,
    patch_id: str,
    req: PatchReviewRequest,
    user=Depends(get_current_user_dep),
):
    """Approve, reject, or edit a patch."""
    db = get_database()
    session_repo = SessionRepository(db)

    fields = {"reviewed_at": datetime.utcnow()}
    if req.action == "approve":
        fields["status"] = PatchStatus.APPROVED.value
    elif req.action == "reject":
        fields["status"] = PatchStatus.REJECTED.value
    elif req.action == "edit":
        if not req.editor_revision:
            raise HTTPException(400, "editor_revision required for edit action")
        fields["status"] = PatchStatus.EDITED.value
        fields["editor_revision"] = req.editor_revision
    else:
        raise HTTPException(400, "Invalid action. Use 'approve', 'reject', or 'edit'.")

    updated = await session_repo.update_patch(patch_id, fields)
    logger.info("Patch %s reviewed: action=%s, status=%s, updated=%s", patch_id, req.action, fields["status"], updated)
    return {
        "patch_id": patch_id,
        "status": fields["status"],
        "updated": updated,
    }


class AskAIRequest(BaseModel):
    prompt: str
    section_id: str  # outline item id
    section_text: str = ""  # the heading text of the selected section
    position: str = "replace"  # "before", "after", "replace"


@router.post("/{session_id}/ask-ai")
async def ask_ai_patch(
    session_id: str,
    req: AskAIRequest,
    user=Depends(get_current_user_dep),
):
    """Use AI to generate a new patch based on user prompt, section, and position."""
    from openai import AsyncOpenAI
    import json as _json

    if req.position not in ("before", "after", "replace"):
        raise HTTPException(400, "position must be 'before', 'after', or 'replace'")

    db = get_database()
    session_repo = SessionRepository(db)
    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")

    rules = session.get("rules", {}) or {}
    citation_style = rules.get("citation_style", "inline")
    voice_preservation = rules.get("voice_preservation", True)

    style_instructions = ""
    if voice_preservation:
        style_instructions = "Preserve the original author's writing style, tone, and voice. "
    style_instructions += f"Citation style: {citation_style}. "

    position_label = {
        "before": "INSERT NEW CONTENT BEFORE",
        "after": "INSERT NEW CONTENT AFTER",
        "replace": "REPLACE THE CONTENT OF",
    }[req.position]

    prompt_text = f"""You are a technical editor. The user wants to {position_label.lower()} the section "{req.section_text}" in a document.

User instruction: "{req.prompt}"
Target section: "{req.section_text}"
Position: {position_label} this section

{style_instructions}

IMPORTANT: Keep the generated content concise and accurate — maximum 4 to 5 lines. Be direct and to the point. Do not write lengthy paragraphs.

Return ONLY valid JSON with:
- "revised_sentence": the generated content (concise, 4-5 lines max)
- "citation": citation if applicable (empty string if none)
- "rationale": brief explanation of what was generated and why (1 sentence)
- "confidence": float 0.0-1.0

Return ONLY valid JSON. No other text."""

    try:
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model=settings.GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise technical editor. Generate accurate, well-cited content that matches the document's style."},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.3,
            max_tokens=500,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        ai_data = _json.loads(raw)

        position_labels_short = {"before": "Before", "after": "After", "replace": "Replace"}
        section_ref = f"{position_labels_short[req.position]}: {req.section_text}"

        original = f"[{position_labels_short[req.position]} section: {req.section_text}]" if req.position != "replace" else req.section_text

        patch = {
            "patch_id": str(uuid.uuid4())[:8],
            "opportunity_id": "ask-ai",
            "session_id": session_id,
            "original_sentence": original,
            "revised_sentence": ai_data.get("revised_sentence", ""),
            "citation": ai_data.get("citation", ""),
            "rationale": ai_data.get("rationale", ""),
            "confidence": ai_data.get("confidence", 0.5),
            "change_pct": 100.0,
            "status": PatchStatus.PENDING.value,
            "editor_revision": None,
            "reviewed_at": None,
            "section_ref": section_ref,
            "ask_ai_meta": {
                "section_id": req.section_id,
                "section_text": req.section_text,
                "position": req.position,
                "user_prompt": req.prompt,
            },
        }

        await session_repo.create_patches([patch])
        # Remove MongoDB _id if present (not JSON-serializable)
        patch.pop("_id", None)
        logger.info("Ask AI patch created for session %s: %s (section=%s, position=%s)",
                     session_id, patch["patch_id"], req.section_text, req.position)

        return {
            "patch": patch,
            "message": "AI-generated patch created successfully.",
        }
    except Exception as e:
        logger.error("Ask AI failed for session %s: %s", session_id, str(e))
        raise HTTPException(500, f"AI generation failed: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8: APPLY PATCHES
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/{session_id}/apply-patches")
async def apply_patches(
    session_id: str,
    user=Depends(get_current_user_dep),
):
    """Apply approved patches to a copy of the original document."""
    db = get_database()
    session_repo = SessionRepository(db)
    doc_repo = DocumentRepository(db)

    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")

    doc = await doc_repo.find_by_id(session["document_id"])
    if not doc:
        raise HTTPException(404, "Document not found")

    approved_patches = await session_repo.find_approved_patches(session_id)
    ask_ai_count = sum(1 for p in approved_patches if p.get("ask_ai_meta"))
    logger.info("Session %s: %d approved patches (%d regular, %d ask-ai)",
                session_id, len(approved_patches), len(approved_patches) - ask_ai_count, ask_ai_count)

    if not approved_patches:
        # No approved patches — skip application, just advance the session
        await session_repo.update_session(session_id, {
            "status": SessionStatus.EDITS_APPLIED.value,
        })
        return {
            "session_id": session_id,
            "status": SessionStatus.EDITS_APPLIED.value,
            "applied": 0,
            "skipped": 0,
            "message": "No approved patches to apply. Skipped to next stage.",
        }

    original_path = _resolve_file_path(doc.get("file_path", ""))
    applied = 0
    skipped = 0
    working_path = None

    if os.path.exists(original_path):
        # Copy original to working path and apply patches in DOCX
        working_dir = os.path.join(settings.PROCESSING_DIR, session_id)
        os.makedirs(working_dir, exist_ok=True)
        working_path = os.path.join(working_dir, f"working_{doc.get('original_filename', 'document.docx')}")
        shutil.copy2(original_path, working_path)

        try:
            from docx import Document as DocxDocument
            from copy import deepcopy
            docx_doc = DocxDocument(working_path)

            para_count_before = len(docx_doc.paragraphs)
            logger.info("DOCX loaded: %d paragraphs", para_count_before)

            for patch in approved_patches:
                final_text = patch.get("editor_revision") or patch.get("revised_sentence", "")
                ask_ai_meta = patch.get("ask_ai_meta")

                if not final_text:
                    skipped += 1
                    continue

                # --- Ask AI patches: insert/replace relative to a section heading ---
                if ask_ai_meta:
                    section_text = ask_ai_meta.get("section_text", "").strip()
                    position = ask_ai_meta.get("position", "after")

                    if not section_text:
                        skipped += 1
                        continue

                    heading_idx = _find_heading_in_docx(docx_doc, section_text)

                    if heading_idx is None:
                        logger.warning("Ask AI patch: heading '%s' not found in DOCX (patch_id=%s)",
                                       section_text, patch.get("patch_id"))
                        skipped += 1
                        continue

                    logger.info("Ask AI patch: found heading at paragraph %d for '%s' (position=%s)",
                                heading_idx, section_text, position)

                    if position == "before":
                        _insert_text_near_paragraph(docx_doc, heading_idx, final_text, before=True)
                        applied += 1
                    elif position == "after":
                        _insert_text_near_paragraph(docx_doc, heading_idx, final_text, before=False)
                        applied += 1
                    elif position == "replace":
                        _replace_section_content(docx_doc, heading_idx, final_text)
                        applied += 1
                    else:
                        skipped += 1
                    continue

                # --- Regular patches: find-and-replace ---
                original_text = patch.get("original_sentence", "")

                if not original_text:
                    skipped += 1
                    continue

                found = False
                for para in docx_doc.paragraphs:
                    if original_text in para.text:
                        _replace_text_in_paragraph(para, original_text, final_text)
                        found = True
                        applied += 1
                        break

                if not found:
                    skipped += 1

            para_count_after = len(docx_doc.paragraphs)
            logger.info("DOCX after patches: %d paragraphs (was %d, diff=%d)",
                        para_count_after, para_count_before, para_count_after - para_count_before)

            docx_doc.save(working_path)
            logger.info("DOCX saved to %s", working_path)

            # Verify: reload and check paragraph count
            verify_doc = DocxDocument(working_path)
            verify_count = len(verify_doc.paragraphs)
            logger.info("VERIFY: Reloaded saved DOCX has %d paragraphs (expected %d)",
                        verify_count, para_count_after)
            # Log first 5 new paragraphs for debugging
            if verify_count > para_count_before:
                for i in range(min(5, verify_count)):
                    txt = verify_doc.paragraphs[i].text[:80] if verify_doc.paragraphs[i].text else "(empty)"
                    logger.info("  VERIFY para[%d]: %s", i, txt)

        except Exception as e:
            logger.error("Patch application failed for session %s: %s", session_id, e)
            raise HTTPException(500, f"Failed to apply patches: {str(e)}")
    else:
        # File not available locally — record patches as applied without DOCX modification
        logger.warning("Original file not found at %s — marking patches as applied without DOCX edit", original_path)
        applied = len(approved_patches)

    await session_repo.update_session(session_id, {
        "working_doc_path": working_path or "",
        "status": SessionStatus.EDITS_APPLIED.value,
    })

    return {
        "session_id": session_id,
        "status": SessionStatus.EDITS_APPLIED.value,
        "applied": applied,
        "skipped": skipped,
        "message": f"{applied} patches applied, {skipped} skipped.",
    }


def _replace_text_in_paragraph(paragraph, old_text: str, new_text: str):
    """Replace text in a paragraph while preserving run formatting."""
    # Join all runs to get full text
    full_text = "".join(run.text for run in paragraph.runs)
    if old_text not in full_text:
        return

    new_full = full_text.replace(old_text, new_text, 1)

    # Put all text in first run, clear the rest
    if paragraph.runs:
        paragraph.runs[0].text = new_full
        for run in paragraph.runs[1:]:
            run.text = ""


def _normalize_text(text: str) -> str:
    """Normalize whitespace and case for fuzzy heading matching."""
    import re
    return re.sub(r'\s+', ' ', text.strip()).lower()


def _find_heading_in_docx(docx_doc, section_text: str):
    """Find a heading paragraph index using multiple matching strategies."""
    normalized_target = _normalize_text(section_text)

    # Strategy 1: Exact substring match
    for idx, para in enumerate(docx_doc.paragraphs):
        if section_text in para.text:
            return idx

    # Strategy 2: Normalized (whitespace + case-insensitive) match
    for idx, para in enumerate(docx_doc.paragraphs):
        if normalized_target in _normalize_text(para.text):
            return idx

    # Strategy 3: Match by stripping leading numbers (e.g., "1.2 Title" -> "Title")
    import re
    core_text = re.sub(r'^[\d.]+\s*', '', section_text).strip()
    if core_text and len(core_text) > 3:
        core_lower = core_text.lower()
        for idx, para in enumerate(docx_doc.paragraphs):
            para_core = re.sub(r'^[\d.]+\s*', '', para.text).strip().lower()
            if core_lower in para_core or para_core in core_lower:
                return idx

    # Strategy 4: Check if section_text contains most words of any paragraph
    target_words = set(normalized_target.split())
    if len(target_words) >= 3:
        best_idx = None
        best_overlap = 0
        for idx, para in enumerate(docx_doc.paragraphs):
            para_text = para.text.strip()
            if not para_text:
                continue
            para_words = set(_normalize_text(para_text).split())
            overlap = len(target_words & para_words)
            ratio = overlap / max(len(target_words), 1)
            if ratio > 0.7 and overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx
        if best_idx is not None:
            return best_idx

    return None


def _find_reference_body_paragraph(docx_doc, heading_idx: int):
    """Find the nearest body (non-heading) paragraph after the heading to copy formatting from."""
    for idx in range(heading_idx + 1, len(docx_doc.paragraphs)):
        para = docx_doc.paragraphs[idx]
        style_name = para.style.name if para.style else ""
        if not style_name.startswith("Heading") and para.text.strip():
            return para
    for idx in range(heading_idx - 1, -1, -1):
        para = docx_doc.paragraphs[idx]
        style_name = para.style.name if para.style else ""
        if not style_name.startswith("Heading") and para.text.strip():
            return para
    return None


def _build_formatted_paragraph(docx_doc, ref_para, text_content: str):
    """Create a new paragraph that matches the formatting of ref_para using python-docx API."""
    from copy import deepcopy
    from docx.oxml.ns import qn

    # Use add_paragraph with the same style
    style_name = None
    if ref_para and ref_para.style:
        style_name = ref_para.style.name

    new_para = docx_doc.add_paragraph("", style=style_name)

    # Copy paragraph-level XML properties (alignment, spacing, indentation)
    if ref_para:
        source_pPr = ref_para._element.find(qn("w:pPr"))
        if source_pPr is not None:
            existing_pPr = new_para._element.find(qn("w:pPr"))
            if existing_pPr is not None:
                new_para._element.remove(existing_pPr)
            new_para._element.insert(0, deepcopy(source_pPr))

    # Add the text as a run, copying run formatting from reference
    run = new_para.add_run(text_content)

    if ref_para and ref_para.runs:
        src_run = ref_para.runs[0]
        # Copy font properties
        if src_run.font.name:
            run.font.name = src_run.font.name
        if src_run.font.size:
            run.font.size = src_run.font.size
        run.font.bold = src_run.font.bold
        run.font.italic = src_run.font.italic
        if src_run.font.color and src_run.font.color.rgb:
            run.font.color.rgb = src_run.font.color.rgb

        # Also copy raw rPr XML for properties not covered by python-docx API
        source_rPr = src_run._element.find(qn("w:rPr"))
        if source_rPr is not None:
            existing_rPr = run._element.find(qn("w:rPr"))
            if existing_rPr is not None:
                run._element.remove(existing_rPr)
            run._element.insert(0, deepcopy(source_rPr))

    # Copy paragraph alignment
    if ref_para:
        new_para.alignment = ref_para.alignment

    return new_para


def _insert_text_near_paragraph(docx_doc, para_idx: int, text: str, before: bool = False):
    """Insert text content before or after a paragraph, matching document formatting."""
    target_para = docx_doc.paragraphs[para_idx]
    target_element = target_para._element

    ref_para = _find_reference_body_paragraph(docx_doc, para_idx)
    if ref_para:
        logger.info("  Using reference style: '%s', font: %s, size: %s",
                     ref_para.style.name if ref_para.style else "?",
                     ref_para.runs[0].font.name if ref_para.runs else "?",
                     ref_para.runs[0].font.size if ref_para.runs else "?")

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if before:
        for line in reversed(lines):
            new_para = _build_formatted_paragraph(docx_doc, ref_para, line)
            target_element.addprevious(new_para._element)
            logger.info("  Inserted paragraph BEFORE idx %d: '%s...'", para_idx, line[:60])
    else:
        insert_after = target_element
        for line in lines:
            new_para = _build_formatted_paragraph(docx_doc, ref_para, line)
            insert_after.addnext(new_para._element)
            logger.info("  Inserted paragraph AFTER idx %d: '%s...'", para_idx, line[:60])
            insert_after = new_para._element


def _replace_section_content(docx_doc, heading_idx: int, text: str):
    """Replace content between this heading and the next heading with new text."""
    heading_para = docx_doc.paragraphs[heading_idx]

    body_indices = []
    for idx in range(heading_idx + 1, len(docx_doc.paragraphs)):
        para = docx_doc.paragraphs[idx]
        style_name = para.style.name if para.style else ""
        if style_name.startswith("Heading"):
            break
        body_indices.append(idx)

    ref_para = _find_reference_body_paragraph(docx_doc, heading_idx)

    for idx in reversed(body_indices):
        para = docx_doc.paragraphs[idx]
        para._element.getparent().remove(para._element)

    logger.info("  Removed %d body paragraphs for replace", len(body_indices))
    _insert_text_near_paragraph(docx_doc, heading_idx, text, before=False)


# ── Visual Diff / Tracked Changes Helpers ─────────────────────────────────

def _add_highlight_to_rPr(rPr_element, color: str):
    """Add highlight color to a run properties element.
    Colors: yellow, green, red, cyan, magenta, blue, darkYellow, etc.
    """
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    # Remove existing highlight
    existing = rPr_element.find(qn("w:highlight"))
    if existing is not None:
        rPr_element.remove(existing)

    highlight = OxmlElement("w:highlight")
    highlight.set(qn("w:val"), color)
    rPr_element.append(highlight)


def _add_strikethrough_to_rPr(rPr_element):
    """Add strikethrough to a run properties element."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    strike = OxmlElement("w:strike")
    strike.set(qn("w:val"), "true")
    rPr_element.append(strike)


def _add_font_color_to_rPr(rPr_element, hex_color: str):
    """Set font color on a run properties element. hex_color like 'FF0000' for red."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    existing = rPr_element.find(qn("w:color"))
    if existing is not None:
        rPr_element.remove(existing)

    color_elem = OxmlElement("w:color")
    color_elem.set(qn("w:val"), hex_color)
    rPr_element.append(color_elem)


def _build_tracked_changes_docx(original_path: str, approved_patches: list, output_path: str):
    """Build a visual diff DOCX from the original + approved patches.

    Visual format (no Review tab needed — visible immediately):
    - Regular patches: original text with RED highlight + strikethrough,
      new replacement text with GREEN highlight, both inline
    - Ask AI patches (new content): with YELLOW highlight, matching document formatting
    """
    from docx import Document as DocxDocument
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from copy import deepcopy

    docx_doc = DocxDocument(original_path)
    logger.info("TRACKED: building visual diff from %d patches", len(approved_patches))

    regular_applied = 0
    ai_applied = 0

    for patch in approved_patches:
        final_text = patch.get("editor_revision") or patch.get("revised_sentence", "")
        ask_ai_meta = patch.get("ask_ai_meta")

        if not final_text:
            continue

        try:
            if ask_ai_meta:
                section_text = ask_ai_meta.get("section_text", "").strip()
                position = ask_ai_meta.get("position", "after")
                if not section_text:
                    continue

                heading_idx = _find_heading_in_docx(docx_doc, section_text)
                if heading_idx is None:
                    logger.warning("TRACKED: heading not found for '%s'", section_text[:50])
                    continue

                ref_para = _find_reference_body_paragraph(docx_doc, heading_idx)
                target_element = docx_doc.paragraphs[heading_idx]._element

                lines = [line.strip() for line in final_text.split("\n") if line.strip()]

                if position == "before":
                    for line in reversed(lines):
                        new_p = _make_highlighted_paragraph_xml(ref_para, line, "yellow")
                        target_element.addprevious(new_p)
                else:
                    insert_after = target_element
                    for line in lines:
                        new_p = _make_highlighted_paragraph_xml(ref_para, line, "yellow")
                        insert_after.addnext(new_p)
                        insert_after = new_p

                ai_applied += 1
                logger.info("TRACKED: AI patch applied (yellow highlight) near '%s'", section_text[:40])

            else:
                original_text = patch.get("original_sentence", "")
                if not original_text or not final_text:
                    continue

                found = False
                for para in docx_doc.paragraphs:
                    if original_text in para.text:
                        _apply_visual_diff_to_paragraph(para, original_text, final_text)
                        found = True
                        regular_applied += 1
                        logger.info("TRACKED: regular patch applied (red/green) for '%s...'", original_text[:40])
                        break

                if not found:
                    logger.warning("TRACKED: original text not found: '%s...'", original_text[:50])

        except Exception as e:
            logger.error("TRACKED: patch failed (patch_id=%s): %s", patch.get("patch_id"), e)

    docx_doc.save(output_path)
    logger.info("TRACKED: visual diff DOCX saved — %d regular, %d AI patches applied", regular_applied, ai_applied)


def _make_highlighted_paragraph_xml(ref_para, text: str, highlight_color: str):
    """Create a paragraph element with highlighted text, copying formatting from ref_para.
    Uses pure XML — no docx_doc.add_paragraph() to avoid document corruption.
    """
    from copy import deepcopy
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    new_p = OxmlElement("w:p")

    # Copy paragraph properties (alignment, spacing, indentation) from reference
    if ref_para:
        source_pPr = ref_para._element.find(qn("w:pPr"))
        if source_pPr is not None:
            new_p.append(deepcopy(source_pPr))

    # Build run with text
    new_r = OxmlElement("w:r")

    # Build run properties: copy from reference + add highlight
    rPr = OxmlElement("w:rPr")
    if ref_para:
        source_runs = ref_para._element.findall(qn("w:r"))
        if source_runs:
            source_rPr = source_runs[0].find(qn("w:rPr"))
            if source_rPr is not None:
                rPr = deepcopy(source_rPr)

    # Add the highlight color
    _add_highlight_to_rPr(rPr, highlight_color)
    new_r.append(rPr)

    # Add text
    new_t = OxmlElement("w:t")
    new_t.set(qn("xml:space"), "preserve")
    new_t.text = text
    new_r.append(new_t)

    new_p.append(new_r)
    return new_p


def _apply_visual_diff_to_paragraph(para, old_text: str, new_text: str):
    """Apply visual diff: original in red strikethrough+red highlight, new in green highlight.
    All visible directly — no Review tab needed.
    """
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from copy import deepcopy

    full_text = "".join(run.text for run in para.runs)
    if old_text not in full_text:
        return

    start_idx = full_text.index(old_text)
    end_idx = start_idx + len(old_text)

    before_text = full_text[:start_idx]
    after_text = full_text[end_idx:]

    # Get original run formatting
    orig_rPr = None
    if para.runs:
        orig_rPr_elem = para.runs[0]._element.find(qn("w:rPr"))
        if orig_rPr_elem is not None:
            orig_rPr = deepcopy(orig_rPr_elem)

    # Clear all existing runs
    for run in list(para.runs):
        run._element.getparent().remove(run._element)

    def make_run_with_text(text, rPr_source=None):
        r = OxmlElement("w:r")
        if rPr_source:
            r.append(deepcopy(rPr_source))
        else:
            r.append(OxmlElement("w:rPr"))
        t = OxmlElement("w:t")
        t.set(qn("xml:space"), "preserve")
        t.text = text
        r.append(t)
        return r

    # 1. Unchanged text before
    if before_text:
        para._element.append(make_run_with_text(before_text, orig_rPr))

    # 2. Original text — strikethrough only (no red color/highlight)
    old_run = make_run_with_text(old_text, orig_rPr)
    old_rPr = old_run.find(qn("w:rPr"))
    if old_rPr is None:
        old_rPr = OxmlElement("w:rPr")
        old_run.insert(0, old_rPr)
    _add_strikethrough_to_rPr(old_rPr)
    para._element.append(old_run)

    # 3. Space separator between old and new
    sep_run = make_run_with_text(" ", orig_rPr)
    para._element.append(sep_run)

    # 4. New text — green highlight, same font
    new_run = make_run_with_text(new_text, orig_rPr)
    new_rPr = new_run.find(qn("w:rPr"))
    if new_rPr is None:
        new_rPr = OxmlElement("w:rPr")
        new_run.insert(0, new_rPr)
    _add_highlight_to_rPr(new_rPr, "green")
    para._element.append(new_run)

    # 5. Unchanged text after
    if after_text:
        para._element.append(make_run_with_text(after_text, orig_rPr))


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 9: DATED STATEMENT AUDIT
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/{session_id}/run-audit")
async def run_audit(
    session_id: str,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user_dep),
):
    """Run dated-statement audit on in-scope sections."""
    db = get_database()
    session_repo = SessionRepository(db)
    doc_repo = DocumentRepository(db)

    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")

    doc = await doc_repo.find_by_id(session["document_id"], analysis_mode=True)
    if not doc:
        raise HTTPException(404, "Document not found")

    background_tasks.add_task(_run_audit_task, session_id, session, doc)

    return {
        "session_id": session_id,
        "status": "auditing",
        "message": "Dated statement audit started. Poll session for updates.",
    }


async def _run_audit_task(session_id: str, session: dict, doc: dict):
    """Background task: find temporal language in document."""
    from openai import AsyncOpenAI
    import json

    db = get_database()
    session_repo = SessionRepository(db)

    try:
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        outline = session.get("outline", [])
        text_content = doc.get("text_content", "")

        # Get in-scope text
        paragraphs = text_content.split("\n")
        in_scope = [o for o in outline if o.get("in_scope", True)]
        section_texts = []
        for i, heading in enumerate(in_scope):
            start = heading.get("paragraph_index", 0)
            end = in_scope[i + 1].get("paragraph_index", len(paragraphs)) if i + 1 < len(in_scope) else len(paragraphs)
            section_texts.append({
                "section": heading.get("text", ""),
                "text": "\n".join(paragraphs[start:end]),
            })

        combined = "\n\n".join(
            f"[Section: {s['section']}]\n{s['text']}" for s in section_texts
        )

        # Get existing patched sentences to exclude
        patches = await session_repo.find_patches(session_id)
        patched_sentences = set(p.get("original_sentence", "") for p in patches)

        prompt = f"""Find every sentence containing temporal or date-sensitive language in this text.
Look for: explicit dates ("2018", "March 2019"), relative temporal words ("currently", "recently", "now", "today"),
future tense ("will be", "is planned", "upcoming"), and planned status phrases ("as of", "planned for", "expected to").

For each found, return JSON with:
- "sentence": the exact sentence
- "section_ref": which section it's in
- "trigger_word": the specific word/phrase that is temporal
- "trigger_type": one of "explicit_date", "relative_temporal", "future_tense", "planned_status"
- "risk": "high", "medium", or "low"

Return ONLY a JSON array. No other text.

Text:
{combined[:15000]}"""

        response = await client.chat.completions.create(
            model=settings.GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a temporal language auditor. Find all date-sensitive statements that might be outdated. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=4000,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        statements = json.loads(raw)

        # Delete old statements
        await session_repo.delete_dated_statements(session_id)

        dated_items = []
        for s in statements:
            sentence = s.get("sentence", "")
            # Exclude already-patched sentences
            if sentence in patched_sentences:
                continue
            dated_items.append({
                "statement_id": str(uuid.uuid4())[:8],
                "session_id": session_id,
                "sentence": sentence,
                "trigger_word": s.get("trigger_word", ""),
                "trigger_type": s.get("trigger_type", "explicit_date"),
                "section_ref": s.get("section_ref", ""),
                "risk": s.get("risk", "medium"),
                "resolved": False,
                "resolution_note": None,
            })

        await session_repo.create_dated_statements(dated_items)
        # Status stays at EDITS_APPLIED; transitions to AUDIT_COMPLETE
        # only when the editor resolves all dated statements (in resolve_statement endpoint)

        logger.info("Audit complete for session %s: %d dated statements", session_id, len(dated_items))

    except Exception as e:
        logger.error("Audit failed for session %s: %s", session_id, e)
        await session_repo.update_session(session_id, {
            "status": SessionStatus.ERROR.value,
            "error_message": str(e),
        })


@router.get("/{session_id}/dated-statements")
async def get_dated_statements(session_id: str, user=Depends(get_current_user_dep)):
    """Get all dated statements for review."""
    db = get_database()
    session_repo = SessionRepository(db)
    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")

    statements = await session_repo.find_dated_statements(session_id)
    return {
        "session_id": session_id,
        "statements": statements,
        "total": len(statements),
        "resolved": sum(1 for s in statements if s.get("resolved")),
        "unresolved": sum(1 for s in statements if not s.get("resolved")),
    }


@router.put("/{session_id}/dated-statements/{statement_id}")
async def resolve_statement(
    session_id: str,
    statement_id: str,
    req: DatedStatementResolveRequest,
    user=Depends(get_current_user_dep),
):
    """Resolve a dated statement."""
    db = get_database()
    session_repo = SessionRepository(db)

    if req.resolution not in ("still_current", "flag_for_patch", "acceptable"):
        raise HTTPException(400, "Invalid resolution. Use 'still_current', 'flag_for_patch', or 'acceptable'.")

    updated = await session_repo.resolve_dated_statement(statement_id, req.resolution)

    # Check if all resolved
    statements = await session_repo.find_dated_statements(session_id)
    all_resolved = all(s.get("resolved") for s in statements)

    if all_resolved and statements:
        await session_repo.update_session(session_id, {
            "status": SessionStatus.AUDIT_COMPLETE.value,
        })

    return {
        "statement_id": statement_id,
        "resolution": req.resolution,
        "updated": updated,
        "all_resolved": all_resolved,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 10: EXPORT
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/{session_id}/export/tracked-docx")
async def export_tracked_docx(session_id: str, token: Optional[str] = None, request: Request = None):
    """Export tracked-changes DOCX with visual diff (red/green/yellow highlights)."""
    from fastapi.responses import FileResponse

    # Auth: accept either query param token (direct browser download) or Authorization header
    user = None
    if token:
        user = verify_download_token(token)
    else:
        # Try Authorization header
        auth_header = request.headers.get("authorization", "") if request else ""
        if auth_header.startswith("Bearer "):
            jwt_token = auth_header[7:]
            payload = decode_token(jwt_token)
            user = {"email": payload.get("sub"), "role": payload.get("role", "user"), "token": jwt_token}
    if not user:
        raise HTTPException(401, "Authentication required — pass token query param or Authorization header")

    db = get_database()
    session_repo = SessionRepository(db)
    doc_repo = DocumentRepository(db)

    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")

    # Verify session has reached at least edits_applied stage
    allowed_statuses = (
        SessionStatus.EDITS_APPLIED.value,
        SessionStatus.AUDIT_COMPLETE.value,
        SessionStatus.EXPORTED.value,
    )
    if session.get("status") not in allowed_statuses:
        raise HTTPException(400, "Patches must be applied before exporting. Complete stages 1-8 first.")

    doc = await doc_repo.find_by_id(session["document_id"], lightweight=True)
    if not doc:
        raise HTTPException(404, "Document not found")

    original_path = _resolve_file_path(doc.get("file_path", ""))
    working_path = session.get("working_doc_path", "")

    if not working_path or not os.path.exists(working_path):
        raise HTTPException(
            400,
            "Working document not available. The original file may be on a different server. "
            "Use the Changelog export instead, which contains all patch details.",
        )

    output_dir = os.path.join(settings.OUTPUT_DIR, session_id)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"tracked_{doc.get('original_filename', 'document.docx')}")

    logger.info("EXPORT tracked-docx: building tracked changes from patches")

    # Build tracked changes DOCX using approved patches
    approved_patches = await session_repo.find_approved_patches(session_id)
    try:
        _build_tracked_changes_docx(original_path, approved_patches, output_path)
        logger.info("EXPORT: tracked changes DOCX built at %s", output_path)
    except Exception as e:
        logger.error("EXPORT: tracked changes build failed: %s", e)
        # Fallback: copy working doc
        shutil.copy2(working_path, output_path)

    output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    logger.info("EXPORT: serving %s (%d bytes)", output_path, output_size)

    await session_repo.update_session(session_id, {
        "status": SessionStatus.EXPORTED.value,
    })

    return FileResponse(
        output_path,
        filename=f"tracked_{doc.get('original_filename', 'document.docx')}",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@router.get("/{session_id}/export/clean-docx")
async def export_clean_docx(session_id: str, token: Optional[str] = None, request: Request = None):
    """Export clean DOCX with all changes accepted."""
    from fastapi.responses import FileResponse

    # Auth: accept either query param token or Authorization header
    user = None
    if token:
        user = verify_download_token(token)
    else:
        auth_header = request.headers.get("authorization", "") if request else ""
        if auth_header.startswith("Bearer "):
            jwt_token = auth_header[7:]
            payload = decode_token(jwt_token)
            user = {"email": payload.get("sub"), "role": payload.get("role", "user"), "token": jwt_token}
    if not user:
        raise HTTPException(401, "Authentication required")

    db = get_database()
    session_repo = SessionRepository(db)
    doc_repo = DocumentRepository(db)

    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")

    doc = await doc_repo.find_by_id(session["document_id"], lightweight=True)
    working_path = session.get("working_doc_path", "")

    if not working_path or not os.path.exists(working_path):
        raise HTTPException(
            400,
            "Working document not available. The original file may be on a different server. "
            "Use the Changelog export instead, which contains all patch details.",
        )

    logger.info("EXPORT clean-docx: serving working_path=%s (%d bytes)",
                working_path, os.path.getsize(working_path) if os.path.exists(working_path) else 0)

    return FileResponse(
        working_path,
        filename=f"clean_{doc.get('original_filename', 'document.docx')}",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@router.get("/{session_id}/export/changelog")
async def export_changelog(session_id: str, user=Depends(get_current_user_dep)):
    """Export changelog as JSON with all patches, evidence, and decisions."""
    db = get_database()
    session_repo = SessionRepository(db)

    session = await session_repo.find_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("user_id") != user["email"] and user.get("role") != "admin":
        raise HTTPException(403, "Not authorized")

    patches = await session_repo.find_patches(session_id)
    dated_statements = await session_repo.find_dated_statements(session_id)
    opportunities = await session_repo.find_opportunities(session_id)

    return {
        "session_id": session_id,
        "document_id": session.get("document_id"),
        "rules": session.get("rules"),
        "diagnostic": session.get("diagnostic"),
        "total_opportunities": len(opportunities),
        "patches": patches,
        "dated_statements": dated_statements,
        "exported_at": datetime.utcnow().isoformat(),
    }
