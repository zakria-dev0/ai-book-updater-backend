from fastapi import APIRouter, Depends, HTTPException, status, Request
from datetime import timedelta
from app.models.user import UserCreate, UserLogin, Token, User
from app.core.security import (
    get_password_hash, verify_password,
    create_access_token, create_refresh_token,
    decode_token, get_current_user_dep,
)
from app.core.config import settings
from app.database.connection import get_database
from app.core.logger import get_logger
from app.core.rate_limit import limiter

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=dict,
    summary="Register a new user",
    responses={
        200: {"description": "User registered successfully"},
        400: {"description": "Email already registered"},
    },
)
@limiter.limit("5/minute")
async def register(
    request: Request,
    user_data: UserCreate,
    db=Depends(get_database),
):
    """
    Register a new user account.

    - **email**: A valid email address
    - **password**: Minimum 6 characters
    """
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    user = User(
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
    )
    await db.users.insert_one(user.model_dump(by_alias=True, exclude={"id"}))
    logger.info("New user registered: %s", user_data.email)
    return {"message": "User registered successfully"}


@router.post(
    "/login",
    response_model=Token,
    summary="Login and receive tokens",
    responses={
        200: {"description": "Login successful, returns access and refresh tokens"},
        401: {"description": "Invalid credentials"},
    },
)
@limiter.limit("10/minute")
async def login(
    request: Request,
    user_credentials: UserLogin,
    db=Depends(get_database),
):
    """
    Authenticate and receive a JWT access token + refresh token.

    - **email**: Registered email address
    - **password**: Account password
    """
    user = await db.users.find_one({"email": user_credentials.email})

    if not user or not verify_password(user_credentials.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user["email"]},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    refresh_token = create_refresh_token(data={"sub": user["email"]})

    logger.info("User logged in: %s", user_credentials.email)
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


@router.get(
    "/me",
    response_model=dict,
    summary="Get current authenticated user info",
    responses={
        200: {"description": "Current user info"},
        401: {"description": "Not authenticated"},
    },
)
async def get_me(
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Return the profile of the currently authenticated user.
    """
    user = await db.users.find_one({"email": current_user["email"]})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return {
        "email": user["email"],
        "role": user.get("role", "user"),
        "created_at": user.get("created_at", "").isoformat() if user.get("created_at") else None,
    }


@router.post(
    "/logout",
    response_model=dict,
    summary="Logout and revoke the current token",
    responses={
        200: {"description": "Logged out successfully"},
        401: {"description": "Not authenticated"},
    },
)
async def logout(
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Revoke the current access token so it can no longer be used.
    The token is stored in a blacklist until it naturally expires.
    """
    token = current_user.get("token")
    if token:
        await db.token_blacklist.insert_one({"token": token})
    logger.info("User logged out: %s", current_user["email"])
    return {"message": "Logged out successfully"}


@router.post(
    "/refresh",
    response_model=Token,
    summary="Refresh access token using a refresh token",
    responses={
        200: {"description": "New access token issued"},
        401: {"description": "Invalid or expired refresh token"},
    },
)
async def refresh_token(
    body: dict,
    db=Depends(get_database),
):
    """
    Exchange a valid refresh token for a new access token.

    Request body:
    - **refresh_token**: The refresh token received during login
    """
    token = body.get("refresh_token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="refresh_token is required",
        )

    payload = decode_token(token)
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )

    # Check blacklist
    blacklisted = await db.token_blacklist.find_one({"token": token})
    if blacklisted:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked",
        )

    email = payload.get("sub")
    new_access_token = create_access_token(
        data={"sub": email},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    new_refresh_token = create_refresh_token(data={"sub": email})

    logger.info("Token refreshed for: %s", email)
    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
    }
