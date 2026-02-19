from fastapi import APIRouter, Depends, HTTPException, status
from datetime import timedelta
from app.models.user import UserCreate, UserLogin, Token, User
from app.core.security import get_password_hash, verify_password, create_access_token
from app.core.config import settings
from app.database.connection import get_database

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/register", response_model=dict)
async def register(
    user_data: UserCreate,
    db = Depends(get_database)
):
    """
    Register a new user
    
    - **email**: User email
    - **password**: User password
    - Returns: Success message
    """
    
    # Check if user already exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    user = User(
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password)
    )
    
    await db.users.insert_one(user.model_dump(by_alias=True, exclude={"id"}))
    
    return {"message": "User registered successfully"}

@router.post("/login", response_model=Token)
async def login(
    user_credentials: UserLogin,
    db = Depends(get_database)
):
    """
    Login and get access token
    
    - **email**: User email
    - **password**: User password
    - Returns: JWT access token
    """
    
    # Find user
    user = await db.users.find_one({"email": user_credentials.email})
    
    if not user or not verify_password(user_credentials.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }