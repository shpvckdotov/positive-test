from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://loguser:logpassword@db:5432/logclustering"
    DATABASE_URL_SYNC: str = "postgresql://loguser:logpassword@db:5432/logclustering"

    # Model
    MODEL_DIR: str = "models"
    TFIDF_MAX_FEATURES: int = 5000
    DBSCAN_EPS: float = 0.35
    DBSCAN_MIN_SAMPLES: int = 3
    ANOMALY_THRESHOLD: float = 0.6  # cosine distance threshold for "new cluster"

    # App
    APP_NAME: str = "Log Clustering Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Training data paths
    LINUX_LOG_PATH: str = "data/sample_logs/linux.log"
    WINDOWS_LOG_PATH: str = "data/sample_logs/windows.log"
    HDFS_LOG_PATH: str = "data/sample_logs/hdfs.log"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
