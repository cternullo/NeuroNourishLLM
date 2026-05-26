"""
Gmail API helpers for NeuroNourishLLM.
Loads stored OAuth tokens, refreshes when expired, and sends email.
"""

import base64
import datetime
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from config import GMAIL_SCOPES, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
from database import GmailToken, SessionLocal


def _load_credentials(user_id: str):
    """Return (Credentials, error_str). Refreshes token if expired."""
    db = SessionLocal()
    row = db.query(GmailToken).filter_by(user_id=user_id).first()
    if not row:
        db.close()
        return None, "Gmail not connected"

    creds = Credentials(
        token=row.access_token,
        refresh_token=row.refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=GMAIL_SCOPES,
    )

    now = datetime.datetime.utcnow()
    needs_refresh = row.token_expiry and now >= row.token_expiry.replace(tzinfo=None)
    if needs_refresh or not creds.valid:
        try:
            creds.refresh(Request())
            row.access_token = creds.token
            if creds.expiry:
                row.token_expiry = creds.expiry
            db.commit()
        except Exception as e:
            db.close()
            return None, f"Token refresh failed — please reconnect Gmail ({e})"

    db.close()
    return creds, None


def get_gmail_service(user_id: str):
    """Return (service, error_str). service is None on failure."""
    creds, err = _load_credentials(user_id)
    if err:
        return None, err
    try:
        service = build("gmail", "v1", credentials=creds)
        return service, None
    except Exception as e:
        return None, str(e)


def send_email(user_id: str, to_email: str, subject: str, body: str) -> dict:
    """Send an email via Gmail API. Returns {success, message_id?, error?}."""
    service, err = get_gmail_service(user_id)
    if err:
        return {"success": False, "error": err}

    msg = MIMEText(body, "plain", "utf-8")
    msg["to"] = to_email
    msg["subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")

    try:
        sent = service.users().messages().send(
            userId="me", body={"raw": raw}
        ).execute()
        return {"success": True, "message_id": sent.get("id", "")}
    except Exception as e:
        return {"success": False, "error": str(e)}
