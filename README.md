# AI Book Chapter Update System - Backend

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Setup MongoDB:

```bash
mongod
```

3. Configure environment:

```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Run server:

```bash
uvicorn app.main:app --reload
```

5. API documentation:

```
http://localhost:8000/docs
```

## API Endpoints

### Authentication

- POST `/api/v1/auth/register` - Register new user
- POST `/api/v1/auth/login` - Login and get JWT token

### Document Management

- POST `/api/v1/upload` - Upload DOCX/PDF
- GET `/api/v1/documents` - List all documents
- GET `/api/v1/documents/{id}` - Get document details
- POST `/api/v1/documents/{id}/process` - Start processing
- GET `/api/v1/documents/{id}/status` - Get processing status

## Testing

Run tests:

```bash
pytest tests/ -v
```

## Milestone 1 Completion Criteria

✅ Upload DOCX and PDF files
✅ Extract text with 95%+ accuracy
✅ Extract equations with 90%+ accuracy
✅ Detect figures and tables
✅ Store in MongoDB
✅ JWT authentication working
✅ API documentation available
