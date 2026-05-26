"""
SQLAlchemy setup and models for NeuroNourishLLM.

Falls back to SQLite (local.db) when DATABASE_URL is not set.
"""

import json
import os

from flask_login import UserMixin
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, Text, UniqueConstraint, create_engine
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
    source_type = Column(String(50))
    source_ref = Column(Text)
    word_count = Column(String(20))
    created_by = Column(String(120))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ActivityLog(Base):
    __tablename__ = "activity_log"

    id = Column(String(36), primary_key=True)
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


class TeamNote(Base):
    __tablename__ = "team_notes"

    id = Column(String(36), primary_key=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, default="")
    created_by = Column(String(120), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    is_public = Column(Boolean, default=True, nullable=False)
    needs_help = Column(Boolean, default=False, nullable=False)
    help_resolved = Column(Boolean, default=False, nullable=False)
    tags = Column(Text, default="[]")          # JSON list of strings
    linked_notes = Column(Text, default="[]")  # JSON list of vault Note filenames

    def tags_list(self):
        try:
            return json.loads(self.tags or "[]")
        except Exception:
            return []

    def linked_notes_list(self):
        try:
            return json.loads(self.linked_notes or "[]")
        except Exception:
            return []


class TeamComment(Base):
    __tablename__ = "team_comments"

    id = Column(String(36), primary_key=True)
    team_note_id = Column(
        String(36), ForeignKey("team_notes.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    content = Column(Text, nullable=False)
    created_by = Column(String(120), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    is_edited = Column(Boolean, default=False, nullable=False)


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(120), nullable=False, index=True)  # username
    type = Column(String(50), nullable=False)                   # new_help | new_comment | help_resolved
    team_note_id = Column(String(36), nullable=True)
    message = Column(Text, nullable=False)
    is_read = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class GmailToken(Base):
    __tablename__ = "gmail_tokens"

    id = Column(String(36), primary_key=True)
    user_id = Column(
        String(120), ForeignKey("users.username", ondelete="CASCADE"),
        nullable=False, unique=True, index=True,
    )
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text, nullable=True)
    token_expiry = Column(DateTime(timezone=True), nullable=True)
    email_address = Column(String(200), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AuthorOutreach(Base):
    __tablename__ = "author_outreach"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(120), nullable=False, index=True)
    author_name = Column(String(500), nullable=False)
    author_email = Column(String(200), nullable=True)
    paper_title = Column(Text, nullable=False)
    journal = Column(String(500), nullable=True)
    year = Column(String(10), nullable=True)
    doi = Column(String(200), nullable=True)
    pmid = Column(String(50), nullable=True)
    topic = Column(String(500), nullable=True)
    email_subject = Column(Text, nullable=True)
    email_body = Column(Text, nullable=True)
    status = Column(String(20), default="draft", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    sent_at = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, nullable=True)
