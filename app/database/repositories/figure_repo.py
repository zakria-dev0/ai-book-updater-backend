from typing import Optional, List
from bson import ObjectId
from app.core.logger import get_logger

logger = get_logger(__name__)


class FigureRepository:
    """Repository for storing figures in a separate collection to avoid
    MongoDB's 16 MB document-size limit when documents contain many images."""

    def __init__(self, db):
        self.collection = db.figures

    @staticmethod
    def _serialize(doc: dict) -> dict:
        if doc and "_id" in doc:
            doc["id"] = str(doc.pop("_id"))
        return doc

    # ------------------------------------------------------------------ #
    # Write                                                                #
    # ------------------------------------------------------------------ #

    async def insert_many(self, document_id: str, figures: List[dict]) -> int:
        """Bulk-insert figures for a document. Returns count inserted."""
        if not figures:
            return 0
        for fig in figures:
            fig["document_id"] = document_id
        result = await self.collection.insert_many(figures)
        count = len(result.inserted_ids)
        logger.info("Inserted %d figures for document %s", count, document_id)
        return count

    async def replace_all(self, document_id: str, figures: List[dict]) -> int:
        """Delete existing figures for a document and insert new ones."""
        await self.delete_by_document(document_id)
        return await self.insert_many(document_id, figures)

    # ------------------------------------------------------------------ #
    # Read                                                                 #
    # ------------------------------------------------------------------ #

    async def find_by_document(
        self,
        document_id: str,
        include_image: bool = True,
    ) -> List[dict]:
        """Return all figures for a document.

        When include_image=False, the heavy image_base64 field is excluded
        (useful for listing metadata only).
        """
        projection = None
        if not include_image:
            projection = {"image_base64": 0}
        cursor = self.collection.find({"document_id": document_id}, projection)
        docs = await cursor.to_list(length=1000)
        return [self._serialize(d) for d in docs]

    async def find_one_figure(
        self, document_id: str, figure_id: str
    ) -> Optional[dict]:
        """Fetch a single figure by its figure_id within a document."""
        doc = await self.collection.find_one(
            {"document_id": document_id, "figure_id": figure_id}
        )
        return self._serialize(doc) if doc else None

    # ------------------------------------------------------------------ #
    # Delete                                                               #
    # ------------------------------------------------------------------ #

    async def delete_by_document(self, document_id: str) -> int:
        """Delete all figures belonging to a document."""
        result = await self.collection.delete_many({"document_id": document_id})
        if result.deleted_count:
            logger.info(
                "Deleted %d figures for document %s",
                result.deleted_count,
                document_id,
            )
        return result.deleted_count

    # ------------------------------------------------------------------ #
    # Clone                                                                #
    # ------------------------------------------------------------------ #

    async def clone_figures(self, src_document_id: str, dest_document_id: str) -> int:
        """Copy all figures from one document to another (for clone feature)."""
        figures = await self.find_by_document(src_document_id, include_image=True)
        for fig in figures:
            fig.pop("id", None)
            fig.pop("_id", None)
            fig["document_id"] = dest_document_id
        return await self.insert_many(dest_document_id, figures)
