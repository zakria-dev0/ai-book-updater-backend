from typing import Optional, List
from bson import ObjectId
from datetime import datetime
from app.core.logger import get_logger

logger = get_logger(__name__)


class SessionRepository:
    """Repository for editorial session pipeline CRUD operations"""

    def __init__(self, db):
        self.sessions = db.editorial_sessions
        self.opportunities = db.update_opportunities
        self.research_plans = db.research_plans
        self.evidence_items = db.evidence_items
        self.patches = db.patches
        self.dated_statements = db.dated_statements

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _serialize(doc: dict) -> dict:
        if doc and "_id" in doc:
            doc["id"] = str(doc.pop("_id"))
        return doc

    # ------------------------------------------------------------------ #
    # Sessions                                                             #
    # ------------------------------------------------------------------ #
    async def create_session(self, data: dict) -> str:
        result = await self.sessions.insert_one(data)
        logger.info("Editorial session created: %s", result.inserted_id)
        return str(result.inserted_id)

    async def find_session(self, session_id: str) -> Optional[dict]:
        try:
            doc = await self.sessions.find_one({"_id": ObjectId(session_id)})
            return self._serialize(doc) if doc else None
        except Exception:
            return None

    async def find_sessions_by_document(self, document_id: str) -> List[dict]:
        cursor = self.sessions.find({"document_id": document_id}).sort("created_at", -1)
        docs = await cursor.to_list(length=50)
        return [self._serialize(d) for d in docs]

    async def find_sessions_by_user(self, user_id: str, skip: int = 0, limit: int = 20) -> List[dict]:
        cursor = (
            self.sessions.find({"user_id": user_id})
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        return [self._serialize(d) for d in docs]

    async def update_session(self, session_id: str, fields: dict) -> bool:
        fields["updated_at"] = datetime.utcnow()
        result = await self.sessions.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": fields},
        )
        return result.modified_count > 0

    async def delete_session(self, session_id: str) -> bool:
        # Delete all related data
        await self.opportunities.delete_many({"session_id": session_id})
        await self.research_plans.delete_many({"session_id": session_id})
        await self.evidence_items.delete_many({"session_id": session_id})
        await self.patches.delete_many({"session_id": session_id})
        await self.dated_statements.delete_many({"session_id": session_id})
        result = await self.sessions.delete_one({"_id": ObjectId(session_id)})
        logger.info("Session deleted: %s", session_id)
        return result.deleted_count > 0

    # ------------------------------------------------------------------ #
    # Update Opportunities                                                 #
    # ------------------------------------------------------------------ #
    async def create_opportunities(self, opportunities: List[dict]) -> List[str]:
        if not opportunities:
            return []
        result = await self.opportunities.insert_many(opportunities)
        return [str(oid) for oid in result.inserted_ids]

    async def find_opportunities(self, session_id: str) -> List[dict]:
        cursor = self.opportunities.find({"session_id": session_id})
        docs = await cursor.to_list(length=500)
        return [self._serialize(d) for d in docs]

    async def find_opportunity(self, opportunity_id: str) -> Optional[dict]:
        try:
            doc = await self.opportunities.find_one({"_id": ObjectId(opportunity_id)})
            return self._serialize(doc) if doc else None
        except Exception:
            doc = await self.opportunities.find_one({"opportunity_id": opportunity_id})
            return self._serialize(doc) if doc else None

    async def update_opportunity_selections(self, session_id: str, selections: dict) -> int:
        """Bulk update selected field for opportunities. selections = {opportunity_id: bool}"""
        total = 0
        for opp_id, selected in selections.items():
            try:
                result = await self.opportunities.update_one(
                    {"_id": ObjectId(opp_id), "session_id": session_id},
                    {"$set": {"selected": selected}},
                )
            except Exception:
                result = await self.opportunities.update_one(
                    {"opportunity_id": opp_id, "session_id": session_id},
                    {"$set": {"selected": selected}},
                )
            total += result.modified_count
        return total

    async def find_selected_opportunities(self, session_id: str) -> List[dict]:
        cursor = self.opportunities.find({"session_id": session_id, "selected": True})
        docs = await cursor.to_list(length=500)
        return [self._serialize(d) for d in docs]

    async def delete_opportunities(self, session_id: str) -> int:
        result = await self.opportunities.delete_many({"session_id": session_id})
        return result.deleted_count

    # ------------------------------------------------------------------ #
    # Research Plans                                                       #
    # ------------------------------------------------------------------ #
    async def create_research_plans(self, plans: List[dict]) -> List[str]:
        if not plans:
            return []
        result = await self.research_plans.insert_many(plans)
        return [str(oid) for oid in result.inserted_ids]

    async def find_research_plans(self, session_id: str) -> List[dict]:
        cursor = self.research_plans.find({"session_id": session_id})
        docs = await cursor.to_list(length=500)
        return [self._serialize(d) for d in docs]

    async def find_research_plan(self, plan_id: str) -> Optional[dict]:
        try:
            doc = await self.research_plans.find_one({"_id": ObjectId(plan_id)})
            return self._serialize(doc) if doc else None
        except Exception:
            doc = await self.research_plans.find_one({"plan_id": plan_id})
            return self._serialize(doc) if doc else None

    async def approve_research_plan(self, plan_id: str, approved: bool = True) -> bool:
        try:
            result = await self.research_plans.update_one(
                {"_id": ObjectId(plan_id)},
                {"$set": {"approved": approved}},
            )
            if result.matched_count > 0:
                return result.modified_count > 0
        except Exception:
            pass
        result = await self.research_plans.update_one(
            {"plan_id": plan_id},
            {"$set": {"approved": approved}},
        )
        return result.modified_count > 0

    async def find_approved_plans(self, session_id: str) -> List[dict]:
        cursor = self.research_plans.find({"session_id": session_id, "approved": True})
        docs = await cursor.to_list(length=500)
        return [self._serialize(d) for d in docs]

    async def delete_research_plans(self, session_id: str) -> int:
        result = await self.research_plans.delete_many({"session_id": session_id})
        return result.deleted_count

    # ------------------------------------------------------------------ #
    # Evidence Items                                                       #
    # ------------------------------------------------------------------ #
    async def create_evidence_items(self, items: List[dict]) -> List[str]:
        if not items:
            return []
        result = await self.evidence_items.insert_many(items)
        return [str(oid) for oid in result.inserted_ids]

    async def find_evidence_items(self, session_id: str) -> List[dict]:
        cursor = self.evidence_items.find({"session_id": session_id})
        docs = await cursor.to_list(length=1000)
        return [self._serialize(d) for d in docs]

    async def find_evidence_by_plan(self, plan_id: str) -> List[dict]:
        cursor = self.evidence_items.find({"research_plan_id": plan_id})
        docs = await cursor.to_list(length=50)
        return [self._serialize(d) for d in docs]

    async def decide_evidence(self, evidence_id: str, accepted: bool) -> bool:
        try:
            result = await self.evidence_items.update_one(
                {"_id": ObjectId(evidence_id)},
                {"$set": {"accepted": accepted}},
            )
            if result.matched_count > 0:
                return result.modified_count > 0
        except Exception:
            pass
        result = await self.evidence_items.update_one(
            {"evidence_id": evidence_id},
            {"$set": {"accepted": accepted}},
        )
        return result.modified_count > 0

    async def find_accepted_evidence_for_opportunity(self, opportunity_id: str) -> List[dict]:
        """Find all accepted evidence items linked to an opportunity through its research plans."""
        plan_cursor = self.research_plans.find({"opportunity_id": opportunity_id})
        plans = await plan_cursor.to_list(length=50)
        plan_ids = [p.get("plan_id", str(p.get("_id", ""))) for p in plans]

        if not plan_ids:
            return []

        cursor = self.evidence_items.find({
            "research_plan_id": {"$in": plan_ids},
            "accepted": True,
        })
        docs = await cursor.to_list(length=100)
        return [self._serialize(d) for d in docs]

    async def delete_evidence_items(self, session_id: str) -> int:
        result = await self.evidence_items.delete_many({"session_id": session_id})
        return result.deleted_count

    # ------------------------------------------------------------------ #
    # Patches                                                              #
    # ------------------------------------------------------------------ #
    async def create_patches(self, patches: List[dict]) -> List[str]:
        if not patches:
            return []
        result = await self.patches.insert_many(patches)
        return [str(oid) for oid in result.inserted_ids]

    async def find_patches(self, session_id: str) -> List[dict]:
        cursor = self.patches.find({"session_id": session_id})
        docs = await cursor.to_list(length=500)
        return [self._serialize(d) for d in docs]

    async def find_patch(self, patch_id: str) -> Optional[dict]:
        try:
            doc = await self.patches.find_one({"_id": ObjectId(patch_id)})
            return self._serialize(doc) if doc else None
        except Exception:
            doc = await self.patches.find_one({"patch_id": patch_id})
            return self._serialize(doc) if doc else None

    async def update_patch(self, patch_id: str, fields: dict) -> bool:
        try:
            result = await self.patches.update_one(
                {"_id": ObjectId(patch_id)},
                {"$set": fields},
            )
            if result.matched_count > 0:
                return result.modified_count > 0
        except Exception:
            pass
        result = await self.patches.update_one(
            {"patch_id": patch_id},
            {"$set": fields},
        )
        return result.modified_count > 0

    async def find_approved_patches(self, session_id: str) -> List[dict]:
        cursor = self.patches.find({
            "session_id": session_id,
            "status": {"$in": ["approved", "edited"]},
        })
        docs = await cursor.to_list(length=500)
        return [self._serialize(d) for d in docs]

    async def delete_patches(self, session_id: str) -> int:
        result = await self.patches.delete_many({"session_id": session_id})
        return result.deleted_count

    # ------------------------------------------------------------------ #
    # Dated Statements                                                     #
    # ------------------------------------------------------------------ #
    async def create_dated_statements(self, statements: List[dict]) -> List[str]:
        if not statements:
            return []
        result = await self.dated_statements.insert_many(statements)
        return [str(oid) for oid in result.inserted_ids]

    async def find_dated_statements(self, session_id: str) -> List[dict]:
        cursor = self.dated_statements.find({"session_id": session_id})
        docs = await cursor.to_list(length=500)
        return [self._serialize(d) for d in docs]

    async def resolve_dated_statement(self, statement_id: str, resolution_note: str) -> bool:
        try:
            result = await self.dated_statements.update_one(
                {"_id": ObjectId(statement_id)},
                {"$set": {"resolved": True, "resolution_note": resolution_note}},
            )
            if result.matched_count > 0:
                return result.modified_count > 0
        except Exception:
            pass
        result = await self.dated_statements.update_one(
            {"statement_id": statement_id},
            {"$set": {"resolved": True, "resolution_note": resolution_note}},
        )
        return result.modified_count > 0

    async def delete_dated_statements(self, session_id: str) -> int:
        result = await self.dated_statements.delete_many({"session_id": session_id})
        return result.deleted_count
