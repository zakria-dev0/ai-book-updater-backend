from typing import Optional, List
from bson import ObjectId
from datetime import datetime
from app.core.logger import get_logger

logger = get_logger(__name__)


class ChangeRepository:
    """Repository for change proposal and analysis CRUD operations"""

    def __init__(self, db):
        self.changes = db.changes
        self.changelogs = db.changelogs

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _serialize(doc: dict) -> dict:
        if doc and "_id" in doc:
            doc["id"] = str(doc.pop("_id"))
        return doc

    # ------------------------------------------------------------------ #
    # Change Proposals                                                     #
    # ------------------------------------------------------------------ #
    async def create(self, change_data: dict) -> str:
        result = await self.changes.insert_one(change_data)
        logger.info("Change proposal created: %s", result.inserted_id)
        return str(result.inserted_id)

    async def create_many(self, changes: List[dict]) -> List[str]:
        if not changes:
            return []
        result = await self.changes.insert_many(changes)
        ids = [str(oid) for oid in result.inserted_ids]
        logger.info("Created %d change proposals", len(ids))
        return ids

    async def find_by_id(self, change_id: str) -> Optional[dict]:
        try:
            doc = await self.changes.find_one({"_id": ObjectId(change_id)})
            return self._serialize(doc) if doc else None
        except Exception:
            return None

    async def find_by_document(
        self, document_id: str, skip: int = 0, limit: int = 50
    ) -> List[dict]:
        cursor = (
            self.changes.find({"document_id": document_id})
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        return [self._serialize(d) for d in docs]

    async def find_by_status(
        self, document_id: str, status: str, skip: int = 0, limit: int = 50
    ) -> List[dict]:
        cursor = (
            self.changes.find({"document_id": document_id, "status": status})
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        return [self._serialize(d) for d in docs]

    async def count_by_document(self, document_id: str) -> int:
        return await self.changes.count_documents({"document_id": document_id})

    async def count_by_status(self, document_id: str, status: str) -> int:
        return await self.changes.count_documents(
            {"document_id": document_id, "status": status}
        )

    async def update_status(
        self, change_id: str, status: str, reviewer_note: str = "",
        approval_action: str = "", user_edited_content: str = "",
    ) -> bool:
        update_fields = {
            "status": status,
            "reviewed_at": datetime.utcnow(),
            "reviewer_note": reviewer_note,
        }
        if approval_action:
            update_fields["approval_action"] = approval_action
        if user_edited_content:
            update_fields["user_edited_content"] = user_edited_content

        result = await self.changes.update_one(
            {"_id": ObjectId(change_id)},
            {"$set": update_fields},
        )
        return result.modified_count > 0

    async def delete_by_document(self, document_id: str) -> int:
        result = await self.changes.delete_many({"document_id": document_id})
        logger.info("Deleted %d changes for document %s", result.deleted_count, document_id)
        return result.deleted_count

    # ------------------------------------------------------------------ #
    # Change Logs                                                          #
    # ------------------------------------------------------------------ #
    async def save_changelog(self, changelog_data: dict) -> str:
        result = await self.changelogs.insert_one(changelog_data)
        logger.info("Changelog saved: %s", result.inserted_id)
        return str(result.inserted_id)

    async def find_changelog_by_document(self, document_id: str) -> Optional[dict]:
        doc = await self.changelogs.find_one(
            {"document_id": document_id},
            sort=[("created_at", -1)],
        )
        return self._serialize(doc) if doc else None
