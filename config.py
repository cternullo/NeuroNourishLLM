import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
VAULT_PATH = os.environ.get("VAULT_PATH", "/app")

if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY environment variable is not set")

MODEL = "llama-3.3-70b-versatile"

# Wiki notes are written into a /wiki subdirectory of the vault
VAULT_WIKI_PATH = os.path.join(VAULT_PATH, "wiki")

# Ensure the wiki directory exists on import
os.makedirs(VAULT_WIKI_PATH, exist_ok=True)
