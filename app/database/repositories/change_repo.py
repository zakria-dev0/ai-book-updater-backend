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
        # Try by MongoDB _id first, then by the change_id field
        try:
            doc = await self.changes.find_one({"_id": ObjectId(change_id)})
            if doc:
                return self._serialize(doc)
        except Exception:
            pass
        doc = await self.changes.find_one({"change_id": change_id})
        return self._serialize(doc) if doc else None

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

        # Try by MongoDB _id first, then by the change_id field
        try:
            result = await self.changes.update_one(
                {"_id": ObjectId(change_id)},
                {"$set": update_fields},
            )
            if result.matched_count > 0:
                return result.modified_count > 0
        except Exception:
            pass
        result = await self.changes.update_one(
            {"change_id": change_id},
            {"$set": update_fields},
        )
        return result.modified_count > 0

    async def batch_update_status(
        self, document_id: str, change_ids: List[str],
        status: str, reviewer_note: str = "", approval_action: str = "",
    ) -> int:
        """Bulk-update status for multiple change proposals at once."""
        oids = []
        string_ids = []
        for cid in change_ids:
            try:
                oids.append(ObjectId(cid))
            except Exception:
                string_ids.append(cid)

        update_fields = {
            "status": status,
            "reviewed_at": datetime.utcnow(),
            "reviewer_note": reviewer_note,
        }
        if approval_action:
            update_fields["approval_action"] = approval_action

        total_modified = 0
        # Match by MongoDB _id
        if oids:
            result = await self.changes.update_many(
                {"_id": {"$in": oids}, "document_id": document_id},
                {"$set": update_fields},
            )
            total_modified += result.modified_count
        # Match by change_id field
        if string_ids:
            result = await self.changes.update_many(
                {"change_id": {"$in": string_ids}, "document_id": document_id},
                {"$set": update_fields},
            )
            total_modified += result.modified_count

        logger.info("Batch updated %d changes for document %s", total_modified, document_id)
        return total_modified

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

    async def delete_changelogs_by_document(self, document_id: str) -> int:
        """Delete all previous changelogs for a document so stale data is not returned."""
        result = await self.changelogs.delete_many({"document_id": document_id})
        if result.deleted_count > 0:
            logger.info("Deleted %d old changelogs for document %s", result.deleted_count, document_id)
        return result.deleted_count

    # Summary-only projection: excludes heavy claims/changes arrays
    _CHANGELOG_SUMMARY_PROJECTION = {
        "claims": 0,
        "changes": 0,
    }

    async def find_changelog_by_document(
        self, document_id: str, summary_only: bool = False,
    ) -> Optional[dict]:
        """Fetch the most recent changelog for a document.

        When summary_only=True, excludes the claims and changes arrays
        for fast status checks (only returns counts and metadata).
        """
        projection = self._CHANGELOG_SUMMARY_PROJECTION if summary_only else None
        doc = await self.changelogs.find_one(
            {"document_id": document_id},
            projection=projection,
            sort=[("created_at", -1)],
        )
        if not doc:
            return None

        doc = self._serialize(doc)

        # If claims were stored separately (large changelog), load them back
        if not summary_only and doc.get("claims_stored_separately"):
            claims = await self.find_claims_by_document(doc.get("document_id", ""))
            doc["claims"] = claims
            logger.info(
                "Loaded %d claims from separate storage for document %s",
                len(claims), doc.get("document_id"),
            )

        return doc

    # ------------------------------------------------------------------ #
    # Claims (separate storage for large documents)                        #
    # ------------------------------------------------------------------ #
    async def save_claims_batch(self, document_id: str, claims: List[dict]) -> int:
        """Save claims to a separate collection when changelog exceeds size limits."""
        if not claims:
            return 0
        # Delete any previously stored claims for this document
        await self.changes.database.claims.delete_many({"document_id": document_id})
        # Tag each claim with the document_id
        for claim in claims:
            claim["document_id"] = document_id
        result = await self.changes.database.claims.insert_many(claims)
        logger.info("Saved %d claims separately for document %s", len(result.inserted_ids), document_id)
        return len(result.inserted_ids)

    async def find_claims_by_document(self, document_id: str) -> List[dict]:
        """Load claims from separate collection."""
        cursor = self.changes.database.claims.find({"document_id": document_id})
        docs = await cursor.to_list(length=5000)
        return [self._serialize(d) for d in docs]
