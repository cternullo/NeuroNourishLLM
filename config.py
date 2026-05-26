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

# Google OAuth (Gmail)
GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.environ.get(
    "GOOGLE_REDIRECT_URI", "http://localhost:5001/auth/gmail/callback"
)
GMAIL_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/gmail.send",
]
