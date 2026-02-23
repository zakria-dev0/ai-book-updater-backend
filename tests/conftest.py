# conftest.py
import os
import pytest
import pytest_asyncio
from dotenv import load_dotenv

# Load .env before anything else
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from httpx import AsyncClient, ASGITransport
from app.main import app
from app.database.connection import connect_to_mongo, close_mongo_connection

@pytest_asyncio.fixture(autouse=True)
async def startup_db():
    """Connect to MongoDB before each test and close after"""
    await connect_to_mongo()
    yield
    await close_mongo_connection()

@pytest_asyncio.fixture
async def client():
    """Provide an async test client"""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
