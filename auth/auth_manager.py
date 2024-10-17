import os

class AuthManager:
    """Manages API key authentication."""

    @staticmethod
    def get_api_key():
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key not found. Ensure it is set in the environment variables.")
        return api_key
