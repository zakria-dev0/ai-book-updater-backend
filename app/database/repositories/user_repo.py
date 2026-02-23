from typing import Optional
from app.core.logger import get_logger

logger = get_logger(__name__)


class UserRepository:
    """Repository for user CRUD operations"""

    def __init__(self, db):
        self.collection = db.users

    async def find_by_email(self, email: str) -> Optional[dict]:
        """Find a user by email address"""
        return await self.collection.find_one({"email": email})

    async def create(self, user_data: dict) -> str:
        """Insert a new user and return the inserted id as string"""
        result = await self.collection.insert_one(user_data)
        logger.info("Created user: %s", user_data.get("email"))
        return str(result.inserted_id)

    async def update(self, email: str, update_data: dict) -> bool:
        """Update user fields by email. Returns True if a document was modified."""
        result = await self.collection.update_one(
            {"email": email}, {"$set": update_data}
        )
        return result.modified_count > 0

    async def delete(self, email: str) -> bool:
        """Delete a user by email. Returns True if deleted."""
        result = await self.collection.delete_one({"email": email})
        return result.deleted_count > 0
