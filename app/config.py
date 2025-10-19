"""Global configuration for the platform."""
from pydantic import BaseSettings

class Settings(BaseSettings):
    env: str = "development"
    debug: bool = True
    sec_user_agent: str = "fiveyyyyy sungjy@fiveyyyyy.com"

settings = Settings()
