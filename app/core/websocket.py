"""
Simple WebSocket connection manager for broadcasting document status updates.
"""
from typing import Dict, List
from fastapi import WebSocket
import json
import asyncio
from app.core.logger import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections grouped by document_id."""

    def __init__(self):
        self._connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, document_id: str, websocket: WebSocket):
        await websocket.accept()
        if document_id not in self._connections:
            self._connections[document_id] = []
        self._connections[document_id].append(websocket)
        logger.info("WS connected for document %s (%d clients)", document_id, len(self._connections[document_id]))

    def disconnect(self, document_id: str, websocket: WebSocket):
        if document_id in self._connections:
            self._connections[document_id] = [
                ws for ws in self._connections[document_id] if ws is not websocket
            ]
            if not self._connections[document_id]:
                del self._connections[document_id]

    async def broadcast(self, document_id: str, data: dict):
        """Send a status update to all clients watching this document."""
        if document_id not in self._connections:
            return
        message = json.dumps(data)
        dead = []
        for ws in self._connections[document_id]:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(document_id, ws)


# Singleton instance
ws_manager = ConnectionManager()
