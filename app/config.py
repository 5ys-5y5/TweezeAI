"""Global configuration for the platform."""
from pydantic import BaseSettings

class Settings(BaseSettings):
    env: str = "development"
    debug: bool = True

settings = Settings()
