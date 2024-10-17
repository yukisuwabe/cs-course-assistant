from dotenv import load_dotenv

class Settings:
    """Handles application configuration and settings."""

    @staticmethod
    def load_environment():
        load_dotenv()
