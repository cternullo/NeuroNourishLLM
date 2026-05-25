"""
Standalone script — creates all database tables.

Usage:
    python init_db.py
"""

from database import Base, engine

if __name__ == "__main__":
    Base.metadata.create_all(engine)
    print("Database tables created successfully.")
