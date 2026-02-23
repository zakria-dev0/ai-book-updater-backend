from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings


class Database:
    client: AsyncIOMotorClient = None


db = Database()


async def connect_to_mongo():
    """Connect to MongoDB and create indexes"""
    db.client = AsyncIOMotorClient(settings.MONGODB_URL)
    database = db.client[settings.MONGODB_DB_NAME]

    # users: unique email
    await database.users.create_index("email", unique=True)

    # documents: fast lookup by owner and status
    await database.documents.create_index("user_id")
    await database.documents.create_index([("user_id", 1), ("status", 1)])
    await database.documents.create_index([("uploaded_at", -1)])

    # token_blacklist: auto-expire entries using MongoDB TTL index
    # Tokens are stored with an "expires_at" field; MongoDB removes them automatically.
    await database.token_blacklist.create_index(
        "expires_at", expireAfterSeconds=0
    )

    print(f"Connected to MongoDB – indexes ensured")


async def close_mongo_connection():
    """Close MongoDB connection"""
    if db.client:
        db.client.close()
        print("MongoDB connection closed")


def get_database():
    """Get database instance"""
    return db.client[settings.MONGODB_DB_NAME]
