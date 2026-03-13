# main.py
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.core.config import settings
from app.core.rate_limit import limiter
from app.core.websocket import ws_manager
from app.database.connection import connect_to_mongo, close_mongo_connection, get_database
from app.database.repositories.document_repo import DocumentRepository
from app.api import auth, upload, processing, analysis, admin, export, sessions

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Backend API for the AI Book Update System. "
        "Handles user authentication, DOCX upload, and document parsing "
        "(text, figures, tables extraction)."
    ),
    version="1.0.0",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Rate Limiting ─────────────────────────────────────────────────────────────
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lifecycle ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()
    print(f"{settings.PROJECT_NAME} started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router, prefix=settings.API_V1_PREFIX)
app.include_router(upload.router, prefix=settings.API_V1_PREFIX)
app.include_router(processing.router, prefix=settings.API_V1_PREFIX)
app.include_router(analysis.router, prefix=settings.API_V1_PREFIX)
app.include_router(admin.router, prefix=settings.API_V1_PREFIX)
app.include_router(export.router, prefix=settings.API_V1_PREFIX)
app.include_router(sessions.router, prefix=settings.API_V1_PREFIX)

# ── Utility endpoints ─────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "service": settings.PROJECT_NAME}


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs": "/docs",
        "redoc": "/redoc",
        "version": "1.0.0",
    }


# ── WebSocket for real-time processing status ────────────────────────────────
@app.websocket("/ws/documents/{document_id}/status")
async def websocket_document_status(websocket: WebSocket, document_id: str):
    """
    WebSocket endpoint for real-time processing status updates.
    Clients connect here instead of polling GET /documents/{id}/status.
    """
    await ws_manager.connect(document_id, websocket)
    try:
        # Send current status immediately on connect
        db = get_database()
        repo = DocumentRepository(db)
        doc = await repo.find_by_id(document_id)
        if doc:
            history = []
            for h in doc.get("processing_history", []):
                entry = dict(h)
                if hasattr(entry.get("timestamp"), "isoformat"):
                    entry["timestamp"] = entry["timestamp"].isoformat()
                history.append(entry)
            await websocket.send_json({
                "type": "status",
                "document_id": document_id,
                "status": doc.get("status", ""),
                "progress": doc.get("progress", 0),
                "current_stage": doc.get("current_stage", ""),
                "processing_history": history,
            })
        # Keep connection alive, wait for client disconnect
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(document_id, websocket)
    except Exception:
        ws_manager.disconnect(document_id, websocket)


# ── Custom OpenAPI schema (adds Bearer security scheme) ───────────────────────
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    schema.setdefault("components", {})
    schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": (
                "Enter the JWT access token obtained from `POST /api/v1/auth/login`. "
                "Format: **Bearer &lt;token&gt;**"
            ),
        }
    }

    # Apply security globally to all operations that are not auth endpoints
    for path, path_item in schema.get("paths", {}).items():
        if path.startswith(f"{settings.API_V1_PREFIX}/auth"):
            continue
        for operation in path_item.values():
            if isinstance(operation, dict):
                operation.setdefault("security", [{"BearerAuth": []}])

    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi
