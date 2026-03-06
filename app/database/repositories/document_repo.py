from typing import Optional, List
from bson import ObjectId
from app.core.logger import get_logger

logger = get_logger(__name__)


class DocumentRepository:
    """Repository for document CRUD operations"""

    def __init__(self, db):
        self.collection = db.documents

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _serialize(doc: dict) -> dict:
        """Convert ObjectId _id to string id"""
        if doc and "_id" in doc:
            doc["id"] = str(doc.pop("_id"))
        return doc

    # ------------------------------------------------------------------ #
    # Read                                                                 #
    # ------------------------------------------------------------------ #
    # Heavy fields to exclude for summary/detail views
    _HEAVY_FIELDS_PROJECTION = {
        "text_content": 0,
        "equations": 0,
        "figures": 0,
        "tables": 0,
        "para_to_page": 0,
    }

    # For analysis pipeline: only exclude binary-heavy fields, keep text_content + para_to_page
    _ANALYSIS_PROJECTION = {
        "equations": 0,
        "figures": 0,
        "tables": 0,
    }

    async def find_by_id(
        self, document_id: str, lightweight: bool = False,
        analysis_mode: bool = False,
    ) -> Optional[dict]:
        """Fetch a single document by its ObjectId string.

        When lightweight=True, excludes heavy fields (text_content,
        equations, figures, tables, para_to_page) for faster loading.
        When analysis_mode=True, keeps text_content and para_to_page
        but excludes binary-heavy fields (figures, equations, tables).
        """
        try:
            if analysis_mode:
                projection = self._ANALYSIS_PROJECTION
            elif lightweight:
                projection = self._HEAVY_FIELDS_PROJECTION
            else:
                projection = None
            doc = await self.collection.find_one(
                {"_id": ObjectId(document_id)}, projection,
            )
            return self._serialize(doc) if doc else None
        except Exception:
            return None

    # Fields to exclude when listing documents (they can be huge)
    _LIST_PROJECTION = {
        "text_content": 0,
        "equations": 0,
        "figures": 0,
        "tables": 0,
        "para_to_page": 0,
        "processing_history": 0,
    }

    async def find_by_user(
        self,
        user_id: str,
        skip: int = 0,
        limit: int = 20,
        lightweight: bool = True,
    ) -> List[dict]:
        """Return paginated documents owned by a user (newest first).

        When lightweight=True (default), heavy fields like text_content,
        equations, figures, and tables are excluded for fast listing.
        """
        projection = self._LIST_PROJECTION if lightweight else None
        cursor = (
            self.collection.find({"user_id": user_id}, projection)
            .sort("uploaded_at", -1)
            .skip(skip)
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        return [self._serialize(d) for d in docs]

    async def count_by_user(self, user_id: str) -> int:
        """Count how many documents belong to a user"""
        return await self.collection.count_documents({"user_id": user_id})

    # ------------------------------------------------------------------ #
    # Write                                                                #
    # ------------------------------------------------------------------ #
    async def create(self, document_data: dict) -> str:
        """Insert a new document record and return its id"""
        result = await self.collection.insert_one(document_data)
        logger.info("Document created: %s", result.inserted_id)
        return str(result.inserted_id)

    async def update_fields(self, document_id: str, fields: dict) -> bool:
        """Partial update (set) on a document. Returns True if modified."""
        result = await self.collection.update_one(
            {"_id": ObjectId(document_id)},
            {"$set": fields},
        )
        return result.modified_count > 0

    async def push_history_entry(self, document_id: str, entry: dict) -> bool:
        """Append a processing history entry to the document"""
        result = await self.collection.update_one(
            {"_id": ObjectId(document_id)},
            {"$push": {"processing_history": entry}},
        )
        return result.modified_count > 0

    async def delete(self, document_id: str) -> bool:
        """Delete a document by id. Returns True if deleted."""
        result = await self.collection.delete_one({"_id": ObjectId(document_id)})
        logger.info("Document deleted: %s", document_id)
        return result.deleted_count > 0
