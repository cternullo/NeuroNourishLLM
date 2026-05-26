"""
NeuroNourish Flask server — Phase 8

Routes:
  GET  /                         → serve index.html (login required)
  POST /ingest/topic             → ingest Wikipedia topic (researcher+admin)
  POST /ingest/url               → ingest URL (researcher+admin)
  POST /ingest/pdf               → ingest uploaded PDF (researcher+admin)
  POST /ingest/pubmed            → ingest PubMed record by PMID (researcher+admin)
  POST /ingest/doi               → ingest paper by DOI via CrossRef (researcher+admin)
  POST /chat                     → RAG chat with structured response (all authenticated)
  GET  /notes                    → list vault notes for sidebar (all authenticated)
  GET  /notes/<name>             → fetch note content (all authenticated)
  GET  /api/notes                → list all notes with full metadata (all authenticated)
  GET  /api/notes/<id>           → get single note content + metadata (all authenticated)
  PUT  /api/notes/<id>           → update note content (researcher+admin)
  DELETE /api/notes/<id>         → delete note (admin only)
  POST /api/notes/<id>/reindex   → re-embed a single note (researcher+admin)
  POST /index                    → regenerate INDEX.md (admin only)
  GET  /activity                 → last 50 activity log entries (all authenticated)
  GET/POST /auth/login           → login page
  GET/POST /auth/register        → register page
  GET  /auth/logout              → logout
"""

import datetime
import json
import os
import re
import shutil
import tempfile
import uuid
from functools import wraps

from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_cors import CORS
from flask_login import current_user, login_required
from groq import Groq

from auth import auth_bp
from config import GROQ_API_KEY, MODEL, VAULT_PATH, VAULT_WIKI_PATH
from database import (
    ActivityLog, AuthorOutreach, Base, ChatMessage, GmailToken,
    Note, Notification, SessionLocal, TeamComment, TeamNote, User, engine,
)
import gmail_service
import outreach as outreach_module
from embed import build_index, query_index, reindex_note
from extensions import bcrypt, login_manager
from ingest import ingest_doi, ingest_pdf, ingest_pubmed, ingest_topic, ingest_url, write_note

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
CORS(app)

bcrypt.init_app(app)
login_manager.init_app(app)
login_manager.login_view = "auth.login"

app.register_blueprint(auth_bp)

client = Groq(api_key=GROQ_API_KEY)

try:
    Base.metadata.create_all(engine)
except Exception as _db_init_err:
    print(f"[db] Warning: could not create tables: {_db_init_err}")

ACTIVITY_LOG = os.path.join(VAULT_PATH, "activity_log.json")

CHAT_SYSTEM = (
    "You are a medical research assistant. You summarise research evidence only. "
    "You do not give medical advice. Always state uncertainty. Never make clinical recommendations.\n\n"
    "Answer ONLY from the provided wiki context. If the answer is not in the context, say so clearly.\n\n"
    "Format your response exactly as follows (use these exact bold headers):\n\n"
    "**Answer:**\n"
    "[direct answer here]\n\n"
    "**Evidence from the vault:**\n"
    "[what the notes say, with specific details]\n\n"
    "**Conflicting or uncertain findings:**\n"
    "[any contradictions, gaps, or areas of uncertainty]\n\n"
    "**Limitations:**\n"
    "[key limitations of the evidence]\n\n"
    "**Suggested next questions:**\n"
    "1. [follow-up question]\n"
    "2. [follow-up question]\n"
    "3. [follow-up question]\n\n"
    "**Sources used:**\n"
    "[list the note filenames you drew from]"
)


# ── auth helpers ──────────────────────────────────────────────────────────────

@login_manager.user_loader
def load_user(username):
    db = SessionLocal()
    user = db.query(User).filter_by(username=username).first()
    db.close()
    return user


def require_role(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not current_user.is_authenticated:
                return jsonify({"error": "Authentication required"}), 401
            if current_user.role not in roles:
                return jsonify({"error": "Insufficient permissions"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator


def _create_default_admin():
    admin_username = os.environ.get("DEFAULT_ADMIN_USERNAME")
    admin_password = os.environ.get("DEFAULT_ADMIN_PASSWORD")
    try:
        db = SessionLocal()
        if db.query(User).count() == 0:
            if not admin_username or not admin_password:
                print(
                    "[auth] WARNING: No users exist. Set DEFAULT_ADMIN_USERNAME and "
                    "DEFAULT_ADMIN_PASSWORD env vars to auto-create an admin account."
                )
            else:
                pw_hash = bcrypt.generate_password_hash(admin_password).decode("utf-8")
                db.add(User(username=admin_username, password_hash=pw_hash, role="admin"))
                db.commit()
                print(f"[auth] Default admin created: {admin_username}")
        db.close()
    except Exception as e:
        print(f"[auth] Could not create default admin: {e}")


# ── helpers ──────────────────────────────────────────────────────────────────

def _upsert_note(filename, source_type, source_ref, word_count, created_by):
    try:
        db = SessionLocal()
        n = db.query(Note).filter_by(filename=filename).first()
        if n:
            n.source_type = source_type
            n.source_ref = source_ref
            n.word_count = str(word_count)
        else:
            db.add(Note(
                filename=filename, source_type=source_type,
                source_ref=source_ref, word_count=str(word_count),
                created_by=created_by,
            ))
        db.commit()
        db.close()
    except Exception:
        pass


def _log(action: str, detail: str, note_written: str | None = None) -> None:
    user = current_user.username if current_user.is_authenticated else "anonymous"
    now = datetime.datetime.utcnow()
    entry = {
        "timestamp": now.isoformat() + "Z",
        "user": user,
        "action": action,
        "detail": detail,
        "note_written": note_written,
    }
    os.makedirs(os.path.dirname(ACTIVITY_LOG), exist_ok=True)
    with open(ACTIVITY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    try:
        db = SessionLocal()
        db.add(ActivityLog(
            id=str(uuid.uuid4()),
            timestamp=now,
            user=user,
            action=action,
            detail=detail,
            note_written=note_written,
        ))
        db.commit()
        db.close()
    except Exception as _db_err:
        print(f"[db] _log write failed: {_db_err}")


def _parse_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter block from a note. Returns {} if absent."""
    if not content.startswith("---\n"):
        return {}
    end = content.find("\n---\n", 4)
    if end == -1:
        return {}
    result = {}
    for line in content[4:end].split("\n"):
        if ":" not in line:
            continue
        key, _, raw = line.partition(":")
        key = key.strip()
        raw = raw.strip()
        if not key:
            continue
        if raw.startswith("[") and raw.endswith("]"):
            inner = raw[1:-1].strip()
            if not inner:
                result[key] = []
            else:
                items = re.findall(r'"([^"]*)"', inner)
                result[key] = items if items else [x.strip() for x in inner.split(",") if x.strip()]
        elif raw == "null":
            result[key] = ""
        elif raw.startswith('"') and raw.endswith('"'):
            result[key] = raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        else:
            result[key] = raw
    return result


def get_suggestions(context: str) -> list:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "Based on this content, suggest exactly 3 specific research "
                    "topics or questions this person should explore next. "
                    'Return only a JSON array of 3 short strings, nothing else. '
                    'Example: ["Topic A", "Topic B", "Topic C"]\n\n'
                    + context[:2000]
                ),
            }],
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(s) for s in parsed[:3]]
        return []
    except Exception:
        return []


# ── main page ─────────────────────────────────────────────────────────────────

@app.route("/")
@login_required
def index():
    return render_template("index.html",
                           username=current_user.username,
                           role=current_user.role)


# ── ingest routes ─────────────────────────────────────────────────────────────

@app.route("/ingest/topic", methods=["POST"])
@login_required
@require_role("researcher", "admin")
def route_ingest_topic():
    data = request.get_json(force=True)
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"success": False, "error": "topic is required"}), 400
    try:
        path = ingest_topic(topic, created_by=current_user.username)
        filename = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            note = f.read()
        word_count = len(note.split())
        suggestions = get_suggestions(note)
        _upsert_note(filename, "topic", topic, word_count, current_user.username)
        _log("ingest_topic", topic, filename)
        return jsonify({"success": True, "filename": filename,
                        "word_count": word_count, "suggestions": suggestions})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/ingest/url", methods=["POST"])
@login_required
@require_role("researcher", "admin")
def route_ingest_url():
    data = request.get_json(force=True)
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"success": False, "error": "url is required"}), 400
    try:
        path = ingest_url(url, created_by=current_user.username)
        filename = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            note = f.read()
        word_count = len(note.split())
        suggestions = get_suggestions(note)
        _upsert_note(filename, "url", url, word_count, current_user.username)
        _log("ingest_url", url, filename)
        return jsonify({"success": True, "filename": filename,
                        "word_count": word_count, "suggestions": suggestions})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/ingest/pdf", methods=["POST"])
@login_required
@require_role("researcher", "admin")
def route_ingest_pdf():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"success": False, "error": "File must be a PDF"}), 400

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)
    try:
        file.save(tmp_path)
        path = ingest_pdf(tmp_path, created_by=current_user.username)
        filename = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            note = f.read()
        word_count = len(note.split())
        suggestions = get_suggestions(note)
        _upsert_note(filename, "pdf", file.filename, word_count, current_user.username)
        _log("ingest_pdf", file.filename, filename)
        return jsonify({"success": True, "filename": filename,
                        "word_count": word_count, "suggestions": suggestions})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── pubmed / doi ingest ───────────────────────────────────────────────────────

@app.route("/ingest/pubmed", methods=["POST"])
@login_required
@require_role("researcher", "admin")
def route_ingest_pubmed():
    data = request.get_json(force=True)
    pmid = (data.get("pmid") or "").strip()
    if not pmid:
        return jsonify({"success": False, "error": "pmid is required"}), 400
    try:
        path = ingest_pubmed(pmid, created_by=current_user.username)
        filename = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            note = f.read()
        word_count = len(note.split())
        suggestions = get_suggestions(note)
        _upsert_note(filename, "pubmed", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                     word_count, current_user.username)
        _log("ingest_pubmed", f"PMID:{pmid}", filename)
        return jsonify({"success": True, "filename": filename,
                        "word_count": word_count, "suggestions": suggestions})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/ingest/doi", methods=["POST"])
@login_required
@require_role("researcher", "admin")
def route_ingest_doi():
    data = request.get_json(force=True)
    doi = (data.get("doi") or "").strip()
    if not doi:
        return jsonify({"success": False, "error": "doi is required"}), 400
    try:
        path = ingest_doi(doi, created_by=current_user.username)
        filename = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            note = f.read()
        word_count = len(note.split())
        suggestions = get_suggestions(note)
        _upsert_note(filename, "doi", f"https://doi.org/{doi}",
                     word_count, current_user.username)
        _log("ingest_doi", f"DOI:{doi}", filename)
        return jsonify({"success": True, "filename": filename,
                        "word_count": word_count, "suggestions": suggestions})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── chat ──────────────────────────────────────────────────────────────────────

def _search_team_notes(query: str, limit: int = 3) -> list:
    """Keyword search across public team notes for chat context."""
    tokens = list({t.lower() for t in re.findall(r"\b\w{3,}\b", query)})
    if not tokens:
        return []
    try:
        db = SessionLocal()
        notes = db.query(TeamNote).filter(TeamNote.is_public == True).all()
        db.close()
    except Exception:
        return []
    scored = []
    for note in notes:
        text = (note.title + " " + (note.content or "")).lower()
        score = sum(1 for t in tokens if t in text)
        if score > 0:
            scored.append((score, note))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [n for _, n in scored[:limit]]


@app.route("/chat", methods=["POST"])
@login_required
def route_chat():
    data = request.get_json(force=True)
    message = (data.get("message") or "").strip()
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "message is required"}), 400

    try:
        build_index()

        # Vault RAG: fetch top-20, rescore, return best 7
        chunks = query_index(message)
        sources = sorted({c["source"] for c in chunks})

        vault_ctx = (
            "\n\n---\n\n".join(
                f"[{c['source']}{'  §' + c['section'] if c.get('section') else ''}]\n{c['text']}"
                for c in chunks
            )
            if chunks
            else "No relevant vault notes found."
        )

        # Team notes keyword search
        team_matches = _search_team_notes(message)
        team_ctx_parts = [
            f"[Team note by {n.created_by}: \"{n.title}\"]\n{(n.content or '')[:800]}"
            for n in team_matches
        ]
        team_sources = [{"id": n.id, "title": n.title} for n in team_matches]

        full_context = vault_ctx
        if team_ctx_parts:
            full_context += "\n\n=== Team Notes ===\n\n" + "\n\n---\n\n".join(team_ctx_parts)

        messages = [{"role": "system", "content": CHAT_SYSTEM}]
        messages.extend(history)
        messages.append({
            "role": "user",
            "content": f"Context:\n\n{full_context}\n\nQuestion: {message}",
        })

        resp = client.chat.completions.create(model=MODEL, messages=messages)
        answer = resp.choices[0].message.content
        suggestions = get_suggestions(answer)

        _log("chat", message)
        return jsonify({
            "response": answer,
            "sources": sources,
            "team_sources": team_sources,
            "suggestions": suggestions,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── notes ─────────────────────────────────────────────────────────────────────

@app.route("/notes")
@login_required
def route_notes():
    try:
        files = sorted(f for f in os.listdir(VAULT_WIKI_PATH) if f.endswith(".md"))
        try:
            db = SessionLocal()
            note_types = {n.filename: n.source_type for n in db.query(Note).all()}
            db.close()
        except Exception:
            note_types = {}
        notes = []
        for f in files:
            path = os.path.join(VAULT_WIKI_PATH, f)
            mtime = datetime.datetime.utcfromtimestamp(os.path.getmtime(path))
            notes.append({
                "filename": f,
                "modified": mtime.isoformat() + "Z",
                "source_type": note_types.get(f, "wiki"),
            })
        return jsonify({"notes": notes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/notes/<filename>")
@login_required
def route_note(filename):
    safe = os.path.basename(filename)
    path = os.path.join(VAULT_WIKI_PATH, safe)
    if not os.path.isfile(path):
        return jsonify({"error": "Note not found"}), 404
    with open(path, encoding="utf-8") as f:
        return jsonify({"content": f.read()})


# ── API: notes CRUD + reindex ─────────────────────────────────────────────────

@app.route("/api/notes")
@login_required
def api_notes_list():
    try:
        files = sorted(f for f in os.listdir(VAULT_WIKI_PATH) if f.endswith(".md"))
        try:
            db = SessionLocal()
            db_notes = {n.filename: n for n in db.query(Note).all()}
            db.close()
        except Exception:
            db_notes = {}

        notes = []
        for f in files:
            path = os.path.join(VAULT_WIKI_PATH, f)
            with open(path, encoding="utf-8") as fp:
                content = fp.read()
            fm = _parse_frontmatter(content)
            db_n = db_notes.get(f)
            mtime = datetime.datetime.utcfromtimestamp(os.path.getmtime(path))
            word_count = len(content.split())
            notes.append({
                "id": f,
                "filename": f,
                "title": fm.get("title") or f.replace(".md", "").replace("_", " "),
                "source_type": fm.get("source_type") or (db_n.source_type if db_n else "wiki"),
                "source_url": fm.get("source_url", ""),
                "doi": fm.get("doi", ""),
                "pmid": fm.get("pmid", ""),
                "authors": fm.get("authors") or [],
                "journal": fm.get("journal", ""),
                "year": fm.get("year", ""),
                "biomarkers": fm.get("biomarkers") or [],
                "conditions": fm.get("conditions") or [],
                "created_by": fm.get("created_by") or (db_n.created_by if db_n else ""),
                "created_at": (
                    db_n.created_at.isoformat() + "Z"
                    if db_n and db_n.created_at
                    else mtime.isoformat() + "Z"
                ),
                "word_count": word_count,
                "modified": mtime.isoformat() + "Z",
            })
        return jsonify({"notes": notes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/notes/<path:note_id>")
@login_required
def api_note_get(note_id):
    safe = os.path.basename(note_id)
    path = os.path.join(VAULT_WIKI_PATH, safe)
    if not os.path.isfile(path):
        return jsonify({"error": "Note not found"}), 404
    with open(path, encoding="utf-8") as f:
        content = f.read()
    fm = _parse_frontmatter(content)
    return jsonify({"id": safe, "filename": safe, "content": content, **fm})


@app.route("/api/notes/<path:note_id>", methods=["PUT"])
@login_required
@require_role("researcher", "admin")
def api_note_update(note_id):
    safe = os.path.basename(note_id)
    path = os.path.join(VAULT_WIKI_PATH, safe)
    if not os.path.isfile(path):
        return jsonify({"error": "Note not found"}), 404
    data = request.get_json(force=True)
    content = data.get("content", "")
    if not content.strip():
        return jsonify({"error": "content is required"}), 400
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    word_count = len(content.split())
    try:
        db = SessionLocal()
        n = db.query(Note).filter_by(filename=safe).first()
        if n:
            n.word_count = str(word_count)
            db.commit()
        db.close()
    except Exception:
        pass
    _log("edit_note", safe)
    return jsonify({"success": True, "word_count": word_count})


@app.route("/api/notes/<path:note_id>", methods=["DELETE"])
@login_required
@require_role("admin")
def api_note_delete(note_id):
    safe = os.path.basename(note_id)
    path = os.path.join(VAULT_WIKI_PATH, safe)
    if not os.path.isfile(path):
        return jsonify({"error": "Note not found"}), 404
    os.remove(path)
    try:
        db = SessionLocal()
        db.query(Note).filter_by(filename=safe).delete()
        db.commit()
        db.close()
    except Exception:
        pass
    # purge from chroma (reindex_note handles missing file gracefully)
    try:
        reindex_note(safe)
    except Exception:
        pass
    _log("delete_note", safe)
    return jsonify({"success": True})


@app.route("/api/notes/<path:note_id>/reindex", methods=["POST"])
@login_required
@require_role("researcher", "admin")
def api_note_reindex(note_id):
    safe = os.path.basename(note_id)
    path = os.path.join(VAULT_WIKI_PATH, safe)
    if not os.path.isfile(path):
        return jsonify({"error": "Note not found"}), 404
    reindex_note(safe)
    _log("reindex_note", safe)
    return jsonify({"success": True})


# ── index regeneration ────────────────────────────────────────────────────────

@app.route("/index", methods=["POST"])
@login_required
@require_role("admin")
def route_index():
    try:
        md_files = sorted(
            f for f in os.listdir(VAULT_WIKI_PATH)
            if f.endswith(".md") and f != "INDEX.md"
        )
        entries = []
        for filename in md_files:
            path = os.path.join(VAULT_WIKI_PATH, filename)
            with open(path, encoding="utf-8") as f:
                snippet = f.read()[:1500]

            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return exactly one sentence (20 words or fewer) summarising the "
                            "topic of this wiki note. No preamble — just the sentence."
                        ),
                    },
                    {"role": "user", "content": snippet},
                ],
            )
            summary = resp.choices[0].message.content.strip().rstrip(".")
            entries.append(f"- [[{os.path.splitext(filename)[0]}]] — {summary}")

        write_note("# Wiki Index\n\n" + "\n".join(entries) + "\n", "INDEX.md")
        return jsonify({"success": True, "note_count": len(entries)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── activity log ──────────────────────────────────────────────────────────────

@app.route("/activity")
@login_required
def route_activity():
    try:
        db = SessionLocal()
        rows = (
            db.query(ActivityLog)
            .order_by(ActivityLog.timestamp.desc())
            .limit(50)
            .all()
        )
        db.close()
        entries = [
            {
                "timestamp": r.timestamp.isoformat() + "Z",
                "user": r.user,
                "action": r.action,
                "detail": r.detail,
                "note_written": r.note_written,
            }
            for r in rows
        ]
        return jsonify({"activity": entries})
    except Exception as _db_err:
        print(f"[db] /activity fallback to JSON: {_db_err}")
        if not os.path.exists(ACTIVITY_LOG):
            return jsonify({"activity": []})
        with open(ACTIVITY_LOG, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        entries = []
        for line in lines[-50:]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return jsonify({"activity": entries})


# ── team-note helpers ─────────────────────────────────────────────────────────

def _user_role(username, db):
    u = db.query(User).filter_by(username=username).first()
    return u.role if u else ""


def _tn_dict(note, db, comment_count=None):
    if comment_count is None:
        comment_count = db.query(TeamComment).filter_by(team_note_id=note.id).count()
    return {
        "id": note.id,
        "title": note.title,
        "created_by": note.created_by,
        "creator_role": _user_role(note.created_by, db),
        "created_at": note.created_at.isoformat() + "Z" if note.created_at else "",
        "updated_at": note.updated_at.isoformat() + "Z" if note.updated_at else "",
        "is_public": note.is_public,
        "needs_help": note.needs_help,
        "help_resolved": note.help_resolved,
        "tags": note.tags_list(),
        "linked_notes": note.linked_notes_list(),
        "comment_count": comment_count,
    }


def _tn_full(note, db):
    d = _tn_dict(note, db)
    d["content"] = note.content or ""
    comments = (
        db.query(TeamComment)
        .filter_by(team_note_id=note.id)
        .order_by(TeamComment.created_at)
        .all()
    )
    d["comments"] = [_comment_dict(c, db) for c in comments]
    return d


def _comment_dict(c, db):
    return {
        "id": c.id,
        "team_note_id": c.team_note_id,
        "content": c.content,
        "created_by": c.created_by,
        "creator_role": _user_role(c.created_by, db),
        "created_at": c.created_at.isoformat() + "Z" if c.created_at else "",
        "updated_at": c.updated_at.isoformat() + "Z" if c.updated_at else "",
        "is_edited": c.is_edited,
    }


def _notify(user_id, type_, team_note_id, message):
    if not user_id:
        return
    try:
        db = SessionLocal()
        db.add(Notification(
            id=str(uuid.uuid4()),
            user_id=user_id,
            type=type_,
            team_note_id=team_note_id,
            message=message,
        ))
        db.commit()
        db.close()
    except Exception as e:
        print(f"[notify] {e}")


# ── team-note routes ──────────────────────────────────────────────────────────

@app.route("/api/team-notes")
@login_required
def api_team_notes_list():
    needs_help_only = request.args.get("needs_help") == "true"
    tag_filter = request.args.get("tag", "").strip()
    creator_filter = request.args.get("created_by", "").strip()
    try:
        from sqlalchemy import func as sqla_func, or_
        db = SessionLocal()
        q = db.query(TeamNote).filter(
            or_(TeamNote.is_public == True, TeamNote.created_by == current_user.username)
        )
        if needs_help_only:
            q = q.filter(TeamNote.needs_help == True, TeamNote.help_resolved == False)
        if creator_filter:
            q = q.filter(TeamNote.created_by == creator_filter)
        notes = q.order_by(TeamNote.updated_at.desc()).all()

        note_ids = [n.id for n in notes]
        counts = dict(
            db.query(TeamComment.team_note_id, sqla_func.count(TeamComment.id))
            .filter(TeamComment.team_note_id.in_(note_ids))
            .group_by(TeamComment.team_note_id)
            .all()
        ) if note_ids else {}

        result = []
        for n in notes:
            if tag_filter and tag_filter not in n.tags_list():
                continue
            result.append(_tn_dict(n, db, counts.get(n.id, 0)))
        db.close()
        return jsonify({"notes": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/team-notes", methods=["POST"])
@login_required
@require_role("researcher", "admin")
def api_team_note_create():
    data = request.get_json(force=True)
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "title is required"}), 400
    tags = data.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    try:
        db = SessionLocal()
        note = TeamNote(
            id=str(uuid.uuid4()),
            title=title,
            content=data.get("content", ""),
            created_by=current_user.username,
            is_public=bool(data.get("is_public", True)),
            needs_help=bool(data.get("needs_help", False)),
            help_resolved=False,
            tags=json.dumps(tags),
            linked_notes=json.dumps(data.get("linked_notes", [])),
        )
        db.add(note)
        db.commit()
        result = _tn_full(note, db)
        note_id = note.id
        flag_help = note.needs_help
        db.close()

        _log("team_note_create", title)

        if flag_help:
            try:
                db2 = SessionLocal()
                users = db2.query(User).filter(
                    User.username != current_user.username,
                    User.role.in_(["researcher", "admin"]),
                ).all()
                db2.close()
                for u in users:
                    _notify(u.username, "new_help", note_id,
                            f"{current_user.username} needs help with: {title}")
            except Exception:
                pass

        return jsonify({"success": True, "note": result}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Static segment — registered before /<note_id> so Flask routes it correctly
@app.route("/api/team-notes/needs-help")
@login_required
def api_team_notes_needs_help():
    try:
        db = SessionLocal()
        notes = (
            db.query(TeamNote)
            .filter(TeamNote.needs_help == True,
                    TeamNote.help_resolved == False,
                    TeamNote.is_public == True)
            .order_by(TeamNote.updated_at.desc())
            .limit(20)
            .all()
        )
        result = [_tn_dict(n, db) for n in notes]
        db.close()
        return jsonify({"notes": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/team-notes/<note_id>")
@login_required
def api_team_note_get(note_id):
    try:
        db = SessionLocal()
        note = db.query(TeamNote).filter_by(id=note_id).first()
        if not note:
            db.close()
            return jsonify({"error": "Note not found"}), 404
        if (not note.is_public
                and note.created_by != current_user.username
                and current_user.role != "admin"):
            db.close()
            return jsonify({"error": "Access denied"}), 403
        result = _tn_full(note, db)
        db.close()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/team-notes/<note_id>", methods=["PUT"])
@login_required
def api_team_note_update(note_id):
    try:
        db = SessionLocal()
        note = db.query(TeamNote).filter_by(id=note_id).first()
        if not note:
            db.close()
            return jsonify({"error": "Note not found"}), 404
        if note.created_by != current_user.username and current_user.role != "admin":
            db.close()
            return jsonify({"error": "Not authorized"}), 403

        data = request.get_json(force=True)
        was_help = note.needs_help

        if "title" in data:
            note.title = (data["title"] or "").strip() or note.title
        if "content" in data:
            note.content = data["content"]
        if "is_public" in data:
            note.is_public = bool(data["is_public"])
        if "tags" in data:
            tags = data["tags"]
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            note.tags = json.dumps(tags)
        if "linked_notes" in data:
            note.linked_notes = json.dumps(data["linked_notes"])
        if "needs_help" in data:
            note.needs_help = bool(data["needs_help"])
        if "help_resolved" in data:
            note.help_resolved = bool(data["help_resolved"])

        note.updated_at = datetime.datetime.utcnow()
        db.commit()
        result = _tn_full(note, db)
        newly_help = not was_help and note.needs_help
        db.close()

        if newly_help:
            try:
                db2 = SessionLocal()
                users = db2.query(User).filter(
                    User.username != current_user.username,
                    User.role.in_(["researcher", "admin"]),
                ).all()
                db2.close()
                for u in users:
                    _notify(u.username, "new_help", note_id,
                            f"{current_user.username} needs help with: {note.title}")
            except Exception:
                pass

        _log("team_note_edit", note.title)
        return jsonify({"success": True, "note": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/team-notes/<note_id>", methods=["DELETE"])
@login_required
@require_role("admin")
def api_team_note_delete(note_id):
    try:
        db = SessionLocal()
        note = db.query(TeamNote).filter_by(id=note_id).first()
        if not note:
            db.close()
            return jsonify({"error": "Note not found"}), 404
        title = note.title
        db.query(TeamComment).filter_by(team_note_id=note_id).delete()
        db.query(Notification).filter_by(team_note_id=note_id).delete()
        db.delete(note)
        db.commit()
        db.close()
        _log("team_note_delete", title)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/team-notes/<note_id>/comments", methods=["POST"])
@login_required
def api_add_comment(note_id):
    try:
        db = SessionLocal()
        note = db.query(TeamNote).filter_by(id=note_id).first()
        if not note:
            db.close()
            return jsonify({"error": "Note not found"}), 404
        if (not note.is_public
                and note.created_by != current_user.username
                and current_user.role != "admin"):
            db.close()
            return jsonify({"error": "Access denied"}), 403

        data = request.get_json(force=True)
        content = (data.get("content") or "").strip()
        if not content:
            db.close()
            return jsonify({"error": "content is required"}), 400

        comment = TeamComment(
            id=str(uuid.uuid4()),
            team_note_id=note_id,
            content=content,
            created_by=current_user.username,
        )
        db.add(comment)
        note.updated_at = datetime.datetime.utcnow()
        db.commit()
        result = _comment_dict(comment, db)
        note_title = note.title
        note_creator = note.created_by
        db.close()

        if note_creator != current_user.username:
            _notify(note_creator, "new_comment", note_id,
                    f"{current_user.username} commented on: {note_title}")

        _log("team_note_comment", f"on: {note_title}")
        return jsonify({"success": True, "comment": result}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/team-notes/<note_id>/comments/<comment_id>", methods=["PUT"])
@login_required
def api_edit_comment(note_id, comment_id):
    try:
        db = SessionLocal()
        comment = db.query(TeamComment).filter_by(id=comment_id, team_note_id=note_id).first()
        if not comment:
            db.close()
            return jsonify({"error": "Comment not found"}), 404
        if comment.created_by != current_user.username and current_user.role != "admin":
            db.close()
            return jsonify({"error": "Not authorized"}), 403

        data = request.get_json(force=True)
        content = (data.get("content") or "").strip()
        if not content:
            db.close()
            return jsonify({"error": "content is required"}), 400

        comment.content = content
        comment.is_edited = True
        comment.updated_at = datetime.datetime.utcnow()
        db.commit()
        result = _comment_dict(comment, db)
        db.close()
        return jsonify({"success": True, "comment": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/team-notes/<note_id>/comments/<comment_id>", methods=["DELETE"])
@login_required
def api_delete_comment(note_id, comment_id):
    try:
        db = SessionLocal()
        comment = db.query(TeamComment).filter_by(id=comment_id, team_note_id=note_id).first()
        if not comment:
            db.close()
            return jsonify({"error": "Comment not found"}), 404
        if comment.created_by != current_user.username and current_user.role != "admin":
            db.close()
            return jsonify({"error": "Not authorized"}), 403
        db.delete(comment)
        db.commit()
        db.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/team-notes/<note_id>/resolve-help", methods=["POST"])
@login_required
@require_role("researcher", "admin")
def api_resolve_help(note_id):
    try:
        db = SessionLocal()
        note = db.query(TeamNote).filter_by(id=note_id).first()
        if not note:
            db.close()
            return jsonify({"error": "Note not found"}), 404
        note.needs_help = False
        note.help_resolved = True
        note.updated_at = datetime.datetime.utcnow()
        db.commit()
        note_title = note.title
        note_creator = note.created_by
        db.close()

        if note_creator != current_user.username:
            _notify(note_creator, "help_resolved", note_id,
                    f"{current_user.username} resolved your help request: {note_title}")

        _log("team_note_resolve", note_title)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── notifications ─────────────────────────────────────────────────────────────

@app.route("/api/notifications")
@login_required
def api_notifications():
    try:
        db = SessionLocal()
        notifs = (
            db.query(Notification)
            .filter_by(user_id=current_user.username, is_read=False)
            .order_by(Notification.created_at.desc())
            .limit(20)
            .all()
        )
        result = [
            {
                "id": n.id,
                "type": n.type,
                "team_note_id": n.team_note_id,
                "message": n.message,
                "created_at": n.created_at.isoformat() + "Z" if n.created_at else "",
            }
            for n in notifs
        ]
        db.close()
        return jsonify({"notifications": result})
    except Exception as e:
        return jsonify({"notifications": [], "error": str(e)})


@app.route("/api/notifications/<notif_id>/read", methods=["POST"])
@login_required
def api_notification_read(notif_id):
    try:
        db = SessionLocal()
        n = db.query(Notification).filter_by(
            id=notif_id, user_id=current_user.username
        ).first()
        if n:
            n.is_read = True
            db.commit()
        db.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/notifications/read-all", methods=["POST"])
@login_required
def api_notifications_read_all():
    try:
        db = SessionLocal()
        db.query(Notification).filter_by(
            user_id=current_user.username, is_read=False
        ).update({"is_read": True})
        db.commit()
        db.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Gmail send (via stored OAuth token) ──────────────────────────────────────

@app.route("/api/gmail/send", methods=["POST"])
@login_required
def api_gmail_send():
    data = request.get_json(force=True)
    to_email   = (data.get("to_email") or "").strip()
    subject    = (data.get("subject") or "").strip()
    body       = (data.get("body") or "").strip()
    outreach_id = data.get("outreach_id", "")

    if not to_email or not subject or not body:
        return jsonify({"error": "to_email, subject, and body are required"}), 400

    result = gmail_service.send_email(current_user.username, to_email, subject, body)
    if not result["success"]:
        return jsonify({"error": result["error"]}), 400

    if outreach_id:
        try:
            db = SessionLocal()
            rec = db.query(AuthorOutreach).filter_by(
                id=outreach_id, user_id=current_user.username
            ).first()
            if rec:
                rec.status = "sent"
                rec.sent_at = datetime.datetime.utcnow()
                db.commit()
            db.close()
        except Exception:
            pass

    _log("outreach_sent", f"to {to_email} — {subject[:60]}")
    return jsonify({"success": True, "message_id": result.get("message_id", "")})


# ── Author outreach routes ────────────────────────────────────────────────────

@app.route("/api/outreach/search", methods=["POST"])
@login_required
@require_role("researcher", "admin")
def api_outreach_search():
    data = request.get_json(force=True)
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "topic is required"}), 400
    max_results = min(int(data.get("max_results", 10)), 20)
    try:
        papers = outreach_module.search_papers(topic, max_results)
        return jsonify({"papers": papers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/outreach/draft", methods=["POST"])
@login_required
@require_role("researcher", "admin")
def api_outreach_draft():
    data = request.get_json(force=True)
    author_name  = (data.get("author_name") or "").strip()
    author_email = (data.get("author_email") or "").strip() or None
    paper_title  = (data.get("paper_title") or "").strip()
    journal      = (data.get("journal") or "").strip()
    year         = (data.get("year") or "").strip()
    doi          = (data.get("doi") or "").strip() or None
    pmid         = (data.get("pmid") or "").strip() or None
    topic        = (data.get("topic") or "").strip()

    if not author_name or not paper_title:
        return jsonify({"error": "author_name and paper_title are required"}), 400

    # Sender identity: username + Gmail address if connected
    db = SessionLocal()
    gmail_row = db.query(GmailToken).filter_by(user_id=current_user.username).first()
    sender_email = gmail_row.email_address if gmail_row else f"{current_user.username}@unknown"
    db.close()

    try:
        draft = outreach_module.draft_outreach_email(
            author_name=author_name,
            paper_title=paper_title,
            journal=journal,
            year=year,
            topic=topic,
            sender_name=current_user.username,
            sender_email=sender_email,
        )
        db = SessionLocal()
        rec = AuthorOutreach(
            id=str(uuid.uuid4()),
            user_id=current_user.username,
            author_name=author_name,
            author_email=author_email,
            paper_title=paper_title,
            journal=journal,
            year=year,
            doi=doi,
            pmid=pmid,
            topic=topic,
            email_subject=draft["subject"],
            email_body=draft["body"],
            status="draft",
        )
        db.add(rec)
        db.commit()
        result = {
            "id": rec.id,
            "author_name": rec.author_name,
            "author_email": rec.author_email or "",
            "paper_title": rec.paper_title,
            "journal": rec.journal or "",
            "year": rec.year or "",
            "doi": rec.doi or "",
            "pmid": rec.pmid or "",
            "topic": rec.topic or "",
            "subject": rec.email_subject,
            "body": rec.email_body,
            "status": rec.status,
            "created_at": rec.created_at.isoformat() + "Z" if rec.created_at else "",
        }
        db.close()
        _log("outreach_draft", f"{author_name} — {paper_title[:60]}")
        return jsonify({"success": True, "draft": result}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/outreach")
@login_required
def api_outreach_list():
    try:
        db = SessionLocal()
        recs = (
            db.query(AuthorOutreach)
            .filter_by(user_id=current_user.username)
            .order_by(AuthorOutreach.created_at.desc())
            .all()
        )
        result = [
            {
                "id": r.id,
                "author_name": r.author_name,
                "author_email": r.author_email or "",
                "paper_title": r.paper_title,
                "journal": r.journal or "",
                "year": r.year or "",
                "doi": r.doi or "",
                "pmid": r.pmid or "",
                "topic": r.topic or "",
                "subject": r.email_subject or "",
                "body": r.email_body or "",
                "status": r.status,
                "created_at": r.created_at.isoformat() + "Z" if r.created_at else "",
                "sent_at": r.sent_at.isoformat() + "Z" if r.sent_at else "",
            }
            for r in recs
        ]
        db.close()
        return jsonify({"drafts": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/outreach/<outreach_id>", methods=["PUT"])
@login_required
def api_outreach_update(outreach_id):
    try:
        db = SessionLocal()
        rec = db.query(AuthorOutreach).filter_by(
            id=outreach_id, user_id=current_user.username
        ).first()
        if not rec:
            db.close()
            return jsonify({"error": "Draft not found"}), 404
        data = request.get_json(force=True)
        if "subject" in data:
            rec.email_subject = data["subject"]
        if "body" in data:
            rec.email_body = data["body"]
        if "author_email" in data:
            rec.author_email = (data["author_email"] or "").strip() or None
        db.commit()
        result = {
            "id": rec.id, "author_name": rec.author_name,
            "author_email": rec.author_email or "",
            "paper_title": rec.paper_title, "status": rec.status,
            "subject": rec.email_subject or "", "body": rec.email_body or "",
        }
        db.close()
        return jsonify({"success": True, "draft": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/outreach/<outreach_id>", methods=["DELETE"])
@login_required
@require_role("researcher", "admin")
def api_outreach_delete(outreach_id):
    try:
        db = SessionLocal()
        rec = db.query(AuthorOutreach).filter_by(
            id=outreach_id, user_id=current_user.username
        ).first()
        if not rec:
            db.close()
            return jsonify({"error": "Draft not found"}), 404
        db.delete(rec)
        db.commit()
        db.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/outreach/<outreach_id>/send", methods=["POST"])
@login_required
def api_outreach_send(outreach_id):
    try:
        db = SessionLocal()
        rec = db.query(AuthorOutreach).filter_by(
            id=outreach_id, user_id=current_user.username
        ).first()
        if not rec:
            db.close()
            return jsonify({"error": "Draft not found"}), 404
        if not rec.author_email:
            db.close()
            return jsonify({"error": "No recipient email address on this draft"}), 400

        result = gmail_service.send_email(
            current_user.username,
            rec.author_email,
            rec.email_subject or "(no subject)",
            rec.email_body or "",
        )
        if not result["success"]:
            db.close()
            return jsonify({"error": result["error"]}), 400

        rec.status = "sent"
        rec.sent_at = datetime.datetime.utcnow()
        db.commit()
        db.close()
        _log("outreach_sent", f"{rec.author_name} — {rec.paper_title[:60]}")
        return jsonify({"success": True, "message_id": result.get("message_id", "")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _create_default_admin()
    print("\n  NeuroNourish running at http://localhost:5001\n")
    app.run(debug=True, host="0.0.0.0", port=5001, use_reloader=False)
