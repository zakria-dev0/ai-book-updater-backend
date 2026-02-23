# API Documentation

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

All endpoints except `/auth/register` and `/auth/login` require Bearer token.

### Headers

```
Authorization: Bearer <your_jwt_token>
```

## Endpoints

### 1. Register User

**POST** `/auth/register`

**Request:**

```json
{
  "email": "user@example.com",
  "password": "securepass123"
}
```

**Response:**

```json
{
  "message": "User registered successfully"
}
```

### 2. Login

**POST** `/auth/login`

**Request:**

```json
{
  "email": "user@example.com",
  "password": "securepass123"
}
```

**Response:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### 3. Upload Document

**POST** `/upload`

**Request:**

- Content-Type: `multipart/form-data`
- Body: `file` (DOCX file)

**Response:**

```json
{
  "document_id": "69987faa0ebbabe77df0b42c",
  "filename": "chapter-1.docx",
  "status": "uploaded",
  "uploaded_at": "2026-02-20T15:37:14.549414"
}
```

### 4. Start Processing

**POST** `/documents/{document_id}/process`

**Response:**

```json
{
  "document_id": "69987faa0ebbabe77df0b42c",
  "status": "processing",
  "message": "Document processing started"
}
```

### 5. Get Processing Status

**GET** `/documents/{document_id}/status`

**Response:**

```json
{
  "document_id": "69987faa0ebbabe77df0b42c",
  "status": "processing",
  "progress": 60,
  "current_stage": "extracting_tables",
  "message": "Extracting tables from document",
  "changes_count": 0
}
```

### 6. List All Documents

**GET** `/documents`

**Response:**

```json
{
  "documents": [
    {
      "id": "69987faa0ebbabe77df0b42c",
      "filename": "chapter-1.docx",
      "status": "completed",
      "uploaded_at": "2026-02-20T15:37:14.549414"
    }
  ]
}
```

### 7. Get Document Details

**GET** `/documents/{document_id}`

**Response:** Full document object with extracted content
