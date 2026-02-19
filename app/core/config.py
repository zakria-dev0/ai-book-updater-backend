from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "AI Book Update System"
    DEBUG: bool = True
    
    # MongoDB Configuration
    MONGODB_URL: str
    MONGODB_DB_NAME: str
    
    # JWT Authentication
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    
    # File Upload Configuration
    MAX_UPLOAD_SIZE: int = 52428800  # 50MB
    ALLOWED_EXTENSIONS: List[str] = [".docx", ".pdf"]
    
    # Storage Paths
    UPLOAD_DIR: str = "./storage/uploads"
    PROCESSING_DIR: str = "./storage/processing"
    OUTPUT_DIR: str = "./storage/outputs"
    LOG_DIR: str = "./storage/logs"
    
    # External APIs
    MATHPIX_APP_ID: str = ""
    MATHPIX_APP_KEY: str = ""
    OPENAI_API_KEY: str = ""
    TAVILY_API_KEY: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()