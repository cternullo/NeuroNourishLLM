import datetime
import os
import uuid

from flask import Blueprint, jsonify, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user

from config import GMAIL_SCOPES, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI
from database import GmailToken, SessionLocal, User
from extensions import bcrypt

# Allow HTTP for local dev; oauthlib requires HTTPS otherwise
if not GOOGLE_REDIRECT_URI.startswith("https"):
    os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/auth/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        db = SessionLocal()
        user = db.query(User).filter_by(username=username).first()
        if user and user.password_hash and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user, remember=True)
            db.close()
            return redirect(url_for("index"))
        db.close()
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@auth_bp.route("/auth/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip() or None
        password = request.form.get("password", "")
        if not username or not password:
            error = "Username and password are required."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        else:
            db = SessionLocal()
            existing = db.query(User).filter_by(username=username).first()
            if existing:
                db.close()
                error = "Username already taken."
            else:
                user = User(
                    username=username,
                    email=email,
                    password_hash=bcrypt.generate_password_hash(password).decode("utf-8"),
                    role="researcher",
                )
                db.add(user)
                db.commit()
                login_user(user, remember=True)
                db.close()
                return redirect(url_for("index"))
    return render_template("register.html", error=error)


@auth_bp.route("/auth/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))


# ── Gmail OAuth ───────────────────────────────────────────────────────────────
# Uses requests_oauthlib.OAuth2Session directly rather than google_auth_oauthlib
# Flow, which in newer versions auto-injects a code_challenge (PKCE) into the
# authorization URL. Because the callback constructs a fresh session object the
# code_verifier is lost, causing invalid_grant. OAuth2Session never adds PKCE
# unless you explicitly pass pkce='S256', so this is the safest approach.

_GOOGLE_AUTH_URI  = "https://accounts.google.com/o/oauth2/auth"
_GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"
_GOOGLE_REVOKE_URI = "https://oauth2.googleapis.com/revoke"
_GOOGLE_USERINFO_URI = "https://www.googleapis.com/oauth2/v3/userinfo"


def _make_oauth_session(state=None):
    from requests_oauthlib import OAuth2Session
    return OAuth2Session(
        client_id=GOOGLE_CLIENT_ID,
        scope=GMAIL_SCOPES,
        redirect_uri=GOOGLE_REDIRECT_URI,
        state=state,
    )


@auth_bp.route("/auth/gmail")
@login_required
def gmail_oauth_start():
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return redirect("/?gmail_error=not_configured")
    oauth = _make_oauth_session()
    # No code_challenge / code_verifier — plain authorization code grant
    auth_url, state = oauth.authorization_url(
        _GOOGLE_AUTH_URI,
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    session["gmail_oauth_state"] = state
    return redirect(auth_url)


@auth_bp.route("/auth/gmail/callback")
@login_required
def gmail_oauth_callback():
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return redirect("/?gmail_error=not_configured")
    state = session.pop("gmail_oauth_state", None)
    try:
        oauth = _make_oauth_session(state=state)
        # Railway reverse proxy strips HTTPS; normalize so redirect URI matches
        auth_response = request.url
        if not auth_response.startswith("https") and GOOGLE_REDIRECT_URI.startswith("https"):
            auth_response = auth_response.replace("http://", "https://", 1)
        # Plain code exchange — no code_verifier
        token_data = oauth.fetch_token(
            _GOOGLE_TOKEN_URI,
            authorization_response=auth_response,
            client_secret=GOOGLE_CLIENT_SECRET,
        )

        access_token = token_data.get("access_token", "")
        refresh_token = token_data.get("refresh_token")
        expiry = None
        if token_data.get("expires_at"):
            expiry = datetime.datetime.utcfromtimestamp(token_data["expires_at"])

        import requests as http_req
        ui = http_req.get(
            _GOOGLE_USERINFO_URI,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=8,
        )
        email_address = ui.json().get("email", "") if ui.ok else ""

        db = SessionLocal()
        row = db.query(GmailToken).filter_by(user_id=current_user.username).first()
        now = datetime.datetime.utcnow()
        if row:
            row.access_token = access_token
            if refresh_token:
                row.refresh_token = refresh_token
            row.token_expiry = expiry
            row.email_address = email_address
            row.updated_at = now
        else:
            db.add(GmailToken(
                id=str(uuid.uuid4()),
                user_id=current_user.username,
                access_token=access_token,
                refresh_token=refresh_token,
                token_expiry=expiry,
                email_address=email_address,
            ))
        db.commit()
        db.close()
        return redirect("/?gmail_connected=true")
    except Exception as e:
        return redirect(f"/?gmail_error={str(e)[:60]}")


@auth_bp.route("/auth/gmail/status")
@login_required
def gmail_status():
    db = SessionLocal()
    token = db.query(GmailToken).filter_by(user_id=current_user.username).first()
    db.close()
    if token:
        return jsonify({"connected": True, "email": token.email_address or ""})
    return jsonify({"connected": False, "email": ""})


@auth_bp.route("/auth/gmail/disconnect")
@login_required
def gmail_disconnect():
    db = SessionLocal()
    token = db.query(GmailToken).filter_by(user_id=current_user.username).first()
    if token:
        try:
            import requests as http_req
            http_req.post(
                "https://oauth2.googleapis.com/revoke",
                params={"token": token.access_token},
                timeout=6,
            )
        except Exception:
            pass
        db.delete(token)
        db.commit()
    db.close()
    return redirect("/?gmail_disconnected=true")
