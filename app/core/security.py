from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta if expires_delta
        else timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token (7-day expiry)"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=7)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token, returning the payload"""
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db=None,
) -> dict:
    """
    Get current authenticated user from JWT token.
    Checks token blacklist if db is provided.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception

        # Check token blacklist when db is injected
        if db is not None:
            blacklisted = await db.token_blacklist.find_one({"token": token})
            if blacklisted:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked. Please log in again.",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        return {"email": email, "token": token}
    except JWTError:
        raise credentials_exception


def make_auth_dependency(check_blacklist: bool = True):
    """
    Factory that returns a FastAPI dependency for auth.
    When check_blacklist=True the dependency also injects db to verify logout.
    """
    from app.database.connection import get_database

    async def _auth_with_blacklist(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db=Depends(get_database),
    ) -> dict:
        return await get_current_user(credentials, db)

    async def _auth_without_blacklist(
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> dict:
        return await get_current_user(credentials)

    return _auth_with_blacklist if check_blacklist else _auth_without_blacklist


# Default dependency used across all protected routes (blacklist-aware)
get_current_user_dep = make_auth_dependency(check_blacklist=True)
