"""
Bootstrap an admin user in the database.

Usage:
    python create_admin.py

Reads credentials from environment (or .env file):
    DEFAULT_ADMIN_USERNAME  — defaults to "admin" if not set
    DEFAULT_ADMIN_PASSWORD  — required; exits with error if missing
    DATABASE_URL            — defaults to sqlite:///local.db

Behaviour:
    - If any user with role="admin" already exists, prints their username and exits.
    - Otherwise creates the user and prints confirmation.
"""

import os
import sys

# Load .env before any module that reads DATABASE_URL at import time
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[warn] python-dotenv not installed — reading environment variables directly")

import bcrypt as _bcrypt

# Import after load_dotenv so DATABASE_URL is already in os.environ
from database import Base, SessionLocal, User, engine


def create_admin() -> None:
    username = os.environ.get("DEFAULT_ADMIN_USERNAME", "admin")
    password = os.environ.get("DEFAULT_ADMIN_PASSWORD")

    if not password:
        print("ERROR: DEFAULT_ADMIN_PASSWORD environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    Base.metadata.create_all(engine)

    db = SessionLocal()
    try:
        existing = db.query(User).filter_by(role="admin").first()
        if existing:
            print(f"Admin already exists: {existing.username}")
            return

        pw_hash = _bcrypt.hashpw(password.encode("utf-8"), _bcrypt.gensalt(12)).decode("utf-8")
        db.add(User(username=username, password_hash=pw_hash, role="admin"))
        db.commit()
        print(f"Admin created: {username}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    create_admin()
