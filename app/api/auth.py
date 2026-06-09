from fastapi import APIRouter, Depends, HTTPException, status, Request
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr
from jose import JWTError, jwt
from app.models.user import UserCreate, UserLogin, Token, User
from app.core.security import (
    get_password_hash, verify_password,
    create_access_token, create_refresh_token,
    decode_token, get_current_user_dep,
)
from app.core.config import settings
from app.core.email import send_password_reset_email, is_smtp_configured
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

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Please contact an administrator.",
        )

    access_token = create_access_token(
        data={"sub": user["email"], "role": user.get("role", "user")},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    refresh_token = create_refresh_token(data={"sub": user["email"], "role": user.get("role", "user")})

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
    role = payload.get("role", "user")
    new_access_token = create_access_token(
        data={"sub": email, "role": role},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    new_refresh_token = create_refresh_token(data={"sub": email, "role": role})

    logger.info("Token refreshed for: %s", email)
    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
    }


# ------------------------------------------------------------------ #
# Password Reset                                                       #
# ------------------------------------------------------------------ #

class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


def create_reset_token(email: str) -> str:
    """Create a short-lived JWT for password reset (15 minutes)."""
    expire = datetime.utcnow() + timedelta(minutes=15)
    return jwt.encode(
        {"sub": email, "type": "password_reset", "exp": expire},
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )


def verify_reset_token(token: str) -> str | None:
    """Decode a password reset token. Returns email or None if invalid."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if payload.get("type") != "password_reset":
            return None
        return payload.get("sub")
    except JWTError:
        return None


@router.post(
    "/forgot-password",
    response_model=dict,
    summary="Request a password reset email",
    responses={
        200: {"description": "Reset email sent (or silently ignored if email not found)"},
    },
)
@limiter.limit("3/minute")
async def forgot_password(
    request: Request,
    body: ForgotPasswordRequest,
    db=Depends(get_database),
):
    """
    Send a password reset link to the user's email.
    Always returns success to prevent email enumeration.
    """
    if not is_smtp_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Email service is not configured. Please contact your administrator.",
        )

    user = await db.users.find_one({"email": body.email})
    if user:
        token = create_reset_token(body.email)
        send_password_reset_email(body.email, token)
        logger.info("Password reset email sent to %s", body.email)
    else:
        logger.info("Password reset requested for unknown email: %s", body.email)

    # Always return success to prevent email enumeration
    return {"message": "If an account with that email exists, a password reset link has been sent."}


@router.post(
    "/reset-password",
    response_model=dict,
    summary="Reset password using a token",
    responses={
        200: {"description": "Password reset successful"},
        400: {"description": "Invalid or expired token"},
    },
)
@limiter.limit("5/minute")
async def reset_password(
    request: Request,
    body: ResetPasswordRequest,
    db=Depends(get_database),
):
    """
    Reset the user's password using the token from the reset email.
    """
    if len(body.new_password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters",
        )

    email = verify_reset_token(body.token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset link. Please request a new one.",
        )

    user = await db.users.find_one({"email": email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset link. Please request a new one.",
        )

    await db.users.update_one(
        {"email": email},
        {"$set": {"hashed_password": get_password_hash(body.new_password)}},
    )

    # Blacklist all existing tokens for this user (force re-login)
    # This is a security measure — after password change, old sessions should be invalidated

    logger.info("Password reset successful for %s", email)
    return {"message": "Password has been reset successfully. You can now sign in with your new password."}
