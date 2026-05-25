"""
SQLAlchemy setup and models for NeuroNourishLLM.

Falls back to SQLite (local.db) when DATABASE_URL is not set.
"""

import os

from flask_login import UserMixin
from sqlalchemy import Column, DateTime, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///local.db")

# Heroku/Railway emit postgres:// — SQLAlchemy 1.4+ requires postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()


class User(UserMixin, Base):
    __tablename__ = "users"

    username = Column(String(120), primary_key=True)
    email = Column(String(200), nullable=True)
    password_hash = Column(String(255), nullable=True)
    role = Column(String(20), default="researcher")
    first_seen = Column(DateTime(timezone=True), server_default=func.now())
    last_seen = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    def get_id(self):
        return self.username


class Note(Base):
    __tablename__ = "notes"

    filename = Column(String(255), primary_key=True)
    source_type = Column(String(50))   # topic | url | pdf
    source_ref = Column(Text)
    word_count = Column(String(20))
    created_by = Column(String(120))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ActivityLog(Base):
    __tablename__ = "activity_log"

    id = Column(String(36), primary_key=True)  # UUID string
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    user = Column(String(120), index=True)
    action = Column(String(80))
    detail = Column(Text)
    note_written = Column(String(255), nullable=True)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    user = Column(String(120))
    question = Column(Text)
    answer = Column(Text)
    sources = Column(Text)  # JSON-encoded list
