import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_upload_docx():
    async with AsyncClient(app=app, base_url="http://test") as client:
        # First, create a user and login
        await client.post("/api/v1/auth/register", json={
            "email": "test@example.com",
            "password": "testpass123"
        })
        
        login_response = await client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
            "password": "testpass123"
        })
        token = login_response.json()["access_token"]
        
        # Upload a file
        with open("tests/sample.docx", "rb") as f:
            response = await client.post(
                "/api/v1/upload/",
                files={"file": ("test.docx", f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                headers={"Authorization": f"Bearer {token}"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert data["filename"] == "test.docx"

@pytest.mark.asyncio
async def test_process_document():
    # Similar test for processing endpoint
    pass