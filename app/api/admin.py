import os
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from pydantic import BaseModel, EmailStr, Field
from bson import ObjectId
from app.core.security import get_current_user_dep, get_password_hash
from app.database.connection import get_database
from app.core.logger import get_logger
from app.core.config import settings

logger = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])

ADMIN_ROLES = {"admin", "super_admin"}


@router.get(
    "/stats",
    response_model=dict,
    summary="Get admin dashboard statistics",
    responses={
        200: {"description": "Admin statistics"},
        401: {"description": "Not authenticated"},
        403: {"description": "Not authorized (admin only)"},
    },
)
async def get_admin_stats(
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Get system-wide statistics for the admin dashboard.
    Requires admin role.
    """
    # Check admin role
    user = await db.users.find_one({"email": current_user["email"]})
    if not user or user.get("role", "user") not in ADMIN_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    # Count documents
    total_documents = await db.documents.count_documents({})
    documents_processing = await db.documents.count_documents({"status": "processing"})
    documents_completed = await db.documents.count_documents({
        "status": {"$in": ["completed", "export_ready"]}
    })

    # Count users
    total_users = await db.users.count_documents({})

    # Recent activity: last 20 documents, ordered by most recent
    recent_docs = await db.documents.find(
        {},
        {"user_id": 1, "status": 1, "original_filename": 1, "uploaded_at": 1, "processing_completed_at": 1}
    ).sort("uploaded_at", -1).limit(20).to_list(20)

    recent_activity = []
    for doc in recent_docs:
        action = doc.get("status", "uploaded")
        timestamp = doc.get("processing_completed_at") or doc.get("uploaded_at")
        recent_activity.append({
            "user": doc.get("user_id", "unknown"),
            "action": action,
            "document": doc.get("original_filename", "Unknown"),
            "timestamp": timestamp.isoformat() if timestamp else "",
        })

    return {
        "total_documents": total_documents,
        "documents_processing": documents_processing,
        "documents_completed": documents_completed,
        "total_users": total_users,
        "recent_activity": recent_activity,
    }


@router.get(
    "/token-usage",
    response_model=dict,
    summary="Get aggregated token usage across all analyses",
    responses={
        200: {"description": "Token usage statistics"},
        401: {"description": "Not authenticated"},
        403: {"description": "Not authorized (admin only)"},
    },
)
async def get_token_usage(
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Aggregate token usage from all changelogs.
    Requires admin role.
    """
    user = await db.users.find_one({"email": current_user["email"]})
    if not user or user.get("role", "user") not in ADMIN_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    # Aggregate token usage from changelogs that have token_usage field
    pipeline = [
        {"$match": {"token_usage": {"$exists": True}}},
        {
            "$group": {
                "_id": None,
                "total_prompt_tokens": {"$sum": "$token_usage.total_prompt_tokens"},
                "total_completion_tokens": {"$sum": "$token_usage.total_completion_tokens"},
                "total_analyses": {"$sum": 1},
            }
        },
    ]

    result = await db.changelogs.aggregate(pipeline).to_list(1)
    totals = result[0] if result else {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_analyses": 0,
    }

    # Per-document breakdown (last 20)
    per_doc_cursor = db.changelogs.find(
        {"token_usage": {"$exists": True}},
        {"document_id": 1, "token_usage": 1, "created_at": 1},
    ).sort("created_at", -1).limit(20)
    per_doc_list = await per_doc_cursor.to_list(20)

    by_document = []
    for entry in per_doc_list:
        tu = entry.get("token_usage", {})
        by_document.append({
            "document_id": entry.get("document_id"),
            "prompt_tokens": tu.get("total_prompt_tokens", 0),
            "completion_tokens": tu.get("total_completion_tokens", 0),
            "model": tu.get("model", "unknown"),
            "date": entry.get("created_at").isoformat() if entry.get("created_at") else "",
        })

    return {
        "total_prompt_tokens": totals.get("total_prompt_tokens", 0),
        "total_completion_tokens": totals.get("total_completion_tokens", 0),
        "total_analyses": totals.get("total_analyses", 0),
        "by_document": by_document,
    }


# ------------------------------------------------------------------ #
# Admin helper                                                         #
# ------------------------------------------------------------------ #

async def _require_admin(current_user: dict, db) -> dict:
    """Allow admin or super_admin."""
    user = await db.users.find_one({"email": current_user["email"]})
    if not user or user.get("role", "user") not in ADMIN_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


async def _require_super_admin(current_user: dict, db) -> dict:
    """Allow super_admin only."""
    user = await db.users.find_one({"email": current_user["email"]})
    if not user or user.get("role", "user") != "super_admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super admin access required",
        )
    return user


# ------------------------------------------------------------------ #
# User management                                                      #
# ------------------------------------------------------------------ #

@router.get(
    "/users",
    summary="List all users",
    responses={
        200: {"description": "Paginated user list"},
        403: {"description": "Admin only"},
    },
)
async def list_users(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """List all registered users with role and metadata. Admin only."""
    await _require_admin(current_user, db)

    skip = (page - 1) * page_size
    total = await db.users.count_documents({})
    users_cursor = db.users.find(
        {},
        {"hashed_password": 0},
    ).sort("created_at", -1).skip(skip).limit(page_size)
    users = await users_cursor.to_list(page_size)

    result = []
    for u in users:
        result.append({
            "id": str(u["_id"]),
            "email": u.get("email", ""),
            "role": u.get("role", "user"),
            "is_active": u.get("is_active", True),
            "created_at": u.get("created_at").isoformat() if u.get("created_at") else None,
        })

    return {
        "users": result,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if total else 0,
    }


@router.get(
    "/users/{user_id}/activity",
    summary="Get per-user activity drill-down",
    responses={
        200: {"description": "User activity details"},
        403: {"description": "Admin only"},
        404: {"description": "User not found"},
    },
)
async def get_user_activity(
    user_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Get detailed activity for a specific user: their documents, statuses, and recent actions."""
    await _require_admin(current_user, db)

    try:
        user = await db.users.find_one({"_id": ObjectId(user_id)}, {"hashed_password": 0})
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    user_email = user.get("email", "")

    # Get all documents owned by this user
    docs_cursor = db.documents.find(
        {"user_id": user_email},
        {"original_filename": 1, "status": 1, "uploaded_at": 1, "processing_completed_at": 1},
    ).sort("uploaded_at", -1).limit(50)
    docs = await docs_cursor.to_list(50)

    documents = []
    status_counts: dict = {}
    for doc in docs:
        doc_status = doc.get("status", "uploaded")
        status_counts[doc_status] = status_counts.get(doc_status, 0) + 1
        documents.append({
            "id": str(doc["_id"]),
            "filename": doc.get("original_filename", "Unknown"),
            "status": doc_status,
            "uploaded_at": doc.get("uploaded_at").isoformat() if doc.get("uploaded_at") else None,
            "completed_at": doc.get("processing_completed_at").isoformat() if doc.get("processing_completed_at") else None,
        })

    # Count changelogs (analyses) by this user
    analyses_count = await db.changelogs.count_documents({"user_id": user_email})

    return {
        "user_id": str(user["_id"]),
        "email": user_email,
        "role": user.get("role", "user"),
        "created_at": user.get("created_at").isoformat() if user.get("created_at") else None,
        "total_documents": len(docs),
        "total_analyses": analyses_count,
        "status_breakdown": status_counts,
        "documents": documents,
    }


VALID_ROLES = {"user", "admin", "super_admin"}


class UpdateRoleRequest(BaseModel):
    role: str = Field(..., description="New role: 'user', 'admin', or 'super_admin'")


class CreateUserRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    role: str = Field(default="user", description="Role for the new user")


@router.post(
    "/users",
    summary="Create a new user account (super_admin only)",
    responses={
        201: {"description": "User created"},
        400: {"description": "Email already registered or invalid role"},
        403: {"description": "Super admin only"},
    },
)
async def create_user(
    body: CreateUserRequest,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Create a new user account. Super admin only. Role is limited to 'admin' or 'user'."""
    await _require_super_admin(current_user, db)

    CREATABLE_ROLES = {"user", "admin"}
    if body.role not in CREATABLE_ROLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role must be 'user' or 'admin'",
        )

    existing = await db.users.find_one({"email": body.email})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    new_user = {
        "email": body.email,
        "hashed_password": get_password_hash(body.password),
        "role": body.role,
        "is_active": True,
        "created_at": datetime.utcnow(),
    }
    result = await db.users.insert_one(new_user)

    logger.info("User %s created by super_admin %s", body.email, current_user["email"])
    return {
        "id": str(result.inserted_id),
        "email": body.email,
        "role": body.role,
        "message": "User created successfully",
    }


@router.put(
    "/users/{user_id}/role",
    summary="Update a user's role",
    responses={
        200: {"description": "Role updated"},
        400: {"description": "Invalid role"},
        403: {"description": "Admin only"},
        404: {"description": "User not found"},
    },
)
async def update_user_role(
    user_id: str,
    body: UpdateRoleRequest,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Change a user's role. Admin or super_admin only."""
    caller = await _require_admin(current_user, db)

    if body.role not in VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Role must be one of: {', '.join(sorted(VALID_ROLES))}",
        )

    # Only super_admin can assign the super_admin role
    if body.role == "super_admin" and caller.get("role") != "super_admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only super admins can assign the super_admin role",
        )

    # Prevent self-demotion: admins cannot change their own role
    target_user = await db.users.find_one({"_id": ObjectId(user_id)})
    if target_user and target_user.get("email") == current_user["email"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot change your own role",
        )

    try:
        result = await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"role": body.role}},
        )
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    logger.info("User %s role updated to %s by %s", user_id, body.role, current_user["email"])
    return {"user_id": user_id, "role": body.role, "message": "Role updated"}


@router.patch(
    "/users/{user_id}/status",
    summary="Activate or deactivate a user (super_admin only)",
    responses={
        200: {"description": "Status updated"},
        400: {"description": "Cannot deactivate yourself or another super_admin"},
        403: {"description": "Super admin only"},
        404: {"description": "User not found"},
    },
)
async def update_user_status(
    user_id: str,
    body: dict,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Toggle a user's active status. Super admin only. Cannot deactivate super_admin accounts."""
    await _require_super_admin(current_user, db)

    is_active = body.get("is_active")
    if not isinstance(is_active, bool):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="is_active must be a boolean")

    try:
        target = await db.users.find_one({"_id": ObjectId(user_id)})
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if target.get("email") == current_user["email"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You cannot deactivate your own account")

    if target.get("role") == "super_admin":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot change status of a super_admin account")

    await db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"is_active": is_active}})
    action = "activated" if is_active else "deactivated"
    logger.info("User %s %s by %s", user_id, action, current_user["email"])
    return {"user_id": user_id, "is_active": is_active, "message": f"User {action}"}


@router.delete(
    "/users/{user_id}",
    summary="Delete a user account (super_admin only)",
    responses={
        200: {"description": "User deleted"},
        400: {"description": "Cannot delete yourself or a super_admin"},
        403: {"description": "Super admin only"},
        404: {"description": "User not found"},
    },
)
async def delete_user(
    user_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Permanently delete a user. Super admin only. Cannot delete super_admin accounts."""
    await _require_super_admin(current_user, db)

    try:
        target = await db.users.find_one({"_id": ObjectId(user_id)})
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if target.get("email") == current_user["email"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You cannot delete your own account")

    if target.get("role") == "super_admin":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete a super_admin account")

    await db.users.delete_one({"_id": ObjectId(user_id)})
    logger.info("User %s deleted by %s", user_id, current_user["email"])
    return {"user_id": user_id, "message": "User deleted"}


# ------------------------------------------------------------------ #
# System error logs                                                    #
# ------------------------------------------------------------------ #

@router.get(
    "/logs",
    summary="Get recent system error logs",
    responses={
        200: {"description": "Recent log entries"},
        403: {"description": "Admin only"},
    },
)
async def get_error_logs(
    level: str = Query(default="ERROR", description="Log level filter: ERROR, WARNING, INFO"),
    limit: int = Query(default=50, ge=1, le=200),
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Retrieve recent log entries from the application log file.
    Parses the log file and returns structured entries. Admin only.
    """
    await _require_admin(current_user, db)

    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "app.log")
    entries = []

    if not os.path.exists(log_path):
        return {"entries": [], "total": 0, "log_path": log_path, "message": "Log file not found"}

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        # Parse from the end for the most recent entries
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            # Match level filter
            if level.upper() in line.upper():
                entries.append(line)
            if len(entries) >= limit:
                break
    except Exception as e:
        logger.error("Failed to read log file: %s", e)
        return {"entries": [], "total": 0, "error": str(e)}

    return {
        "entries": entries,
        "total": len(entries),
        "level_filter": level,
    }


# ------------------------------------------------------------------ #
# API usage metrics                                                    #
# ------------------------------------------------------------------ #

@router.get(
    "/api-metrics",
    summary="Get API usage metrics",
    responses={
        200: {"description": "API usage metrics"},
        403: {"description": "Admin only"},
    },
)
async def get_api_metrics(
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Get aggregate API usage metrics: documents processed, analyses run, exports generated.
    Admin only.
    """
    await _require_admin(current_user, db)

    total_documents = await db.documents.count_documents({})
    total_analyses = await db.changelogs.count_documents({})
    total_changes = await db.changes.count_documents({})
    total_approved = await db.changes.count_documents({"status": "approved"})
    total_rejected = await db.changes.count_documents({"status": "rejected"})
    total_pending = await db.changes.count_documents({"status": "pending"})
    total_errors = await db.documents.count_documents({"status": "error"})

    # Documents by status breakdown
    status_pipeline = [
        {"$group": {"_id": "$status", "count": {"$sum": 1}}},
    ]
    status_results = await db.documents.aggregate(status_pipeline).to_list(20)
    documents_by_status = {r["_id"]: r["count"] for r in status_results if r["_id"]}

    return {
        "total_documents": total_documents,
        "total_analyses": total_analyses,
        "total_changes": total_changes,
        "changes_approved": total_approved,
        "changes_rejected": total_rejected,
        "changes_pending": total_pending,
        "total_errors": total_errors,
        "documents_by_status": documents_by_status,
    }


# ------------------------------------------------------------------ #
# Admin: list ALL documents (all users)                               #
# ------------------------------------------------------------------ #

@router.get(
    "/documents",
    summary="List all documents from all users",
    responses={
        200: {"description": "Paginated document list"},
        403: {"description": "Admin only"},
    },
)
async def list_all_documents(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status_filter: Optional[str] = Query(default=None, description="Filter by status"),
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """List all documents across all users. Admin only."""
    await _require_admin(current_user, db)

    query = {}
    if status_filter:
        query["status"] = status_filter

    skip = (page - 1) * page_size
    total = await db.documents.count_documents(query)
    docs_cursor = db.documents.find(
        query,
        {
            "text_content": 0,
            "equations": 0,
            "figures": 0,
            "tables": 0,
            "processing_history": 0,
        },
    ).sort("uploaded_at", -1).skip(skip).limit(page_size)
    docs = await docs_cursor.to_list(page_size)

    documents = []
    for doc in docs:
        documents.append({
            "id": str(doc["_id"]),
            "original_filename": doc.get("original_filename", "Unknown"),
            "file_type": doc.get("file_type", ""),
            "user_id": doc.get("user_id", ""),
            "status": doc.get("status", "uploaded"),
            "uploaded_at": doc["uploaded_at"].isoformat() if hasattr(doc.get("uploaded_at"), "isoformat") else doc.get("uploaded_at"),
            "processing_completed_at": doc["processing_completed_at"].isoformat() if hasattr(doc.get("processing_completed_at"), "isoformat") else doc.get("processing_completed_at"),
            "progress": doc.get("progress", 0),
            "current_stage": doc.get("current_stage", ""),
        })

    return {
        "documents": documents,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if total else 0,
    }


# ------------------------------------------------------------------ #
# Admin: get changes for any document                                  #
# ------------------------------------------------------------------ #

@router.get(
    "/documents/{document_id}/changes",
    summary="List changes for any document (admin)",
    responses={
        200: {"description": "Changes list"},
        403: {"description": "Admin only"},
        404: {"description": "Document not found"},
    },
)
async def admin_list_changes(
    document_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=500),
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """List all change proposals for any document. Admin only."""
    await _require_admin(current_user, db)

    from app.database.repositories.change_repo import ChangeRepository
    change_repo = ChangeRepository(db)
    skip = (page - 1) * page_size
    changes = await change_repo.find_by_document(document_id, skip, page_size)
    total = await change_repo.count_by_document(document_id)

    return {
        "document_id": document_id,
        "changes": changes,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if total else 0,
    }


# ------------------------------------------------------------------ #
# API Keys management                                                  #
# ------------------------------------------------------------------ #

class UpdateApiKeyRequest(BaseModel):
    key: str = Field(..., min_length=1, description="New API key value")


async def get_tavily_key_from_db(db) -> str | None:
    """Read the Tavily API key from the settings collection in MongoDB."""
    doc = await db.settings.find_one({"key": "tavily_api_key"})
    if doc:
        return doc.get("value")
    return None


@router.get(
    "/api-keys",
    summary="Get configured API keys (masked)",
    responses={
        200: {"description": "Masked API keys"},
        403: {"description": "Admin only"},
    },
)
async def get_api_keys(
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Return API keys with values masked (last 4 chars visible). Admin only."""
    await _require_admin(current_user, db)

    # Check DB first, then fall back to .env
    tavily_key = await get_tavily_key_from_db(db) or settings.TAVILY_API_KEY

    def mask(val: str) -> str:
        if not val:
            return ""
        if len(val) <= 4:
            return "****"
        return "*" * (len(val) - 4) + val[-4:]

    return {
        "keys": [
            {
                "id": "tavily",
                "label": "Tavily API Key",
                "masked_value": mask(tavily_key),
                "is_set": bool(tavily_key),
                "source": "database" if await get_tavily_key_from_db(db) else "env",
            },
        ],
    }


@router.put(
    "/api-keys/tavily",
    summary="Update the Tavily API key",
    responses={
        200: {"description": "Key updated"},
        403: {"description": "Admin only"},
    },
)
async def update_tavily_key(
    body: UpdateApiKeyRequest,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Store or update the Tavily API key in MongoDB. Admin only."""
    await _require_admin(current_user, db)

    await db.settings.update_one(
        {"key": "tavily_api_key"},
        {
            "$set": {
                "key": "tavily_api_key",
                "value": body.key,
                "updated_at": datetime.utcnow(),
                "updated_by": current_user["email"],
            }
        },
        upsert=True,
    )

    logger.info("Tavily API key updated by %s", current_user["email"])
    return {"message": "Tavily API key updated successfully"}
