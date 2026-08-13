import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central Application Settings using Pydantic Settings.
    Reads environment variables with fallbacks.
    """
    PROJECT_NAME: str = "Smart Road Damage Detection and Analysis System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security & Auth
    SECRET_KEY: str = "supersecretjwtkey_road_damage_detection_system_2026_change_in_prod"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgrespassword@localhost:5432/road_damage_db"
    
    # Redis / Celery
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # File Paths & Directories
    UPLOAD_DIR: str = os.path.join(os.getcwd(), "uploads")
    PROCESSED_DIR: str = os.path.join(os.getcwd(), "processed")
    REPORTS_DIR: str = os.path.join(os.getcwd(), "reports")
    WEIGHTS_DIR: str = os.path.join(os.getcwd(), "weights")
    
    # YOLO Settings
    YOLO_MODEL_PATH: str = os.path.join(WEIGHTS_DIR, "yolov11x-pothole.pt")
    FALLBACK_YOLO_MODEL: str = os.path.join(WEIGHTS_DIR, "yolov8n-damage.pt")
    CONFIDENCE_THRESHOLD: float = 0.35
    IOU_THRESHOLD: float = 0.45
    FRAME_SKIP: int = 5  # Process every 5th frame for performance
    
    # Severity Formula Weights
    WEIGHT_AREA: float = 0.40
    WEIGHT_CONFIDENCE: float = 0.20
    WEIGHT_DENSITY: float = 0.25
    WEIGHT_CLASS_SEVERITY: float = 0.15

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

# Ensure required local directories exist
for path in [settings.UPLOAD_DIR, settings.PROCESSED_DIR, settings.REPORTS_DIR, settings.WEIGHTS_DIR]:
    os.makedirs(path, exist_ok=True)
