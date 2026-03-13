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

    # changes: fast lookup by document and status
    await database.changes.create_index("document_id")
    await database.changes.create_index([("document_id", 1), ("status", 1)])
    await database.changes.create_index([("created_at", -1)])

    # changelogs: one per document analysis run
    await database.changelogs.create_index("document_id")
    await database.changelogs.create_index([("created_at", -1)])

    # editorial_sessions: pipeline sessions per document
    await database.editorial_sessions.create_index("document_id")
    await database.editorial_sessions.create_index("user_id")
    await database.editorial_sessions.create_index([("created_at", -1)])

    # update_opportunities: issues found per session
    await database.update_opportunities.create_index("session_id")
    await database.update_opportunities.create_index([("session_id", 1), ("selected", 1)])

    # research_plans: per opportunity
    await database.research_plans.create_index("session_id")
    await database.research_plans.create_index("opportunity_id")

    # evidence_items: per research plan
    await database.evidence_items.create_index("session_id")
    await database.evidence_items.create_index("research_plan_id")

    # patches: per opportunity
    await database.patches.create_index("session_id")
    await database.patches.create_index([("session_id", 1), ("status", 1)])

    # dated_statements: temporal audit per session
    await database.dated_statements.create_index("session_id")

    print(f"Connected to MongoDB – indexes ensured")


async def close_mongo_connection():
    """Close MongoDB connection"""
    if db.client:
        db.client.close()
        print("MongoDB connection closed")


def get_database():
    """Get database instance"""
    return db.client[settings.MONGODB_DB_NAME]
