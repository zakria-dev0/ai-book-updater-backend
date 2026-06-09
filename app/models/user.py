from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime

class User(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    email: EmailStr
    hashed_password: str
    role: str = "user"  # "user", "admin", or "super_admin"
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class AdminCreateUser(BaseModel):
    email: EmailStr
    password: str
    role: str = "user"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None