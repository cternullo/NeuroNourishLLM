import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env in the project root
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
VAULT_PATH = os.getenv("VAULT_PATH")

if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY is not set in .env")
if not VAULT_PATH:
    raise EnvironmentError("VAULT_PATH is not set in .env")

MODEL = "llama-3.3-70b-versatile"

# Wiki notes are written into a /wiki subdirectory of the vault
VAULT_WIKI_PATH = os.path.join(VAULT_PATH, "wiki")

# Ensure the wiki directory exists on import
os.makedirs(VAULT_WIKI_PATH, exist_ok=True)
