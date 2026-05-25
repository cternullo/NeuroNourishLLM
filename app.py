"""
NeuroNourish Flask server — Phase 3

Routes:
  GET  /                → serve index.html
  POST /ingest/topic    → ingest Wikipedia topic
  POST /ingest/url      → ingest URL
  POST /ingest/pdf      → ingest uploaded PDF
  POST /chat            → RAG chat with conversation history
  GET  /notes           → list vault notes
  GET  /notes/<name>    → fetch note content
  POST /index           → regenerate INDEX.md
  GET  /activity        → last 50 activity log entries
"""

import datetime
import json
import os
import re
import shutil
import tempfile
import uuid

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from groq import Groq

from config import GROQ_API_KEY, MODEL, VAULT_PATH, VAULT_WIKI_PATH
from database import ActivityLog, Base, ChatMessage, SessionLocal, engine
from embed import build_index, query_index
from ingest import ingest_pdf, ingest_topic, ingest_url, write_note

app = Flask(__name__)
CORS(app)

client = Groq(api_key=GROQ_API_KEY)

try:
    Base.metadata.create_all(engine)
except Exception as _db_init_err:
    print(f"[db] Warning: could not create tables: {_db_init_err}")

ACTIVITY_LOG = os.path.join(VAULT_PATH, "activity_log.json")


# ── helpers ──────────────────────────────────────────────────────────────────

def _word_count(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        return len(f.read().split())


def _log(action: str, detail: str, note_written: str | None = None) -> None:
    user = request.headers.get("X-Username", "anonymous")
    now = datetime.datetime.utcnow()
    entry = {
        "timestamp": now.isoformat() + "Z",
        "user": user,
        "action": action,
        "detail": detail,
        "note_written": note_written,
    }
    # JSON file (for git sync)
    os.makedirs(os.path.dirname(ACTIVITY_LOG), exist_ok=True)
    with open(ACTIVITY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    # Database (wrapped so a DB failure never breaks the request)
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


def get_suggestions(context: str) -> list:
    """Ask Groq for 3 related research directions based on *context*.
    Returns a list of up to 3 strings; returns [] if Groq fails or JSON is malformed."""
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
        # Strip markdown code fences Groq sometimes wraps around JSON output
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(s) for s in parsed[:3]]
        return []
    except Exception:
        return []


# ── main page ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── ingest routes ─────────────────────────────────────────────────────────────

@app.route("/ingest/topic", methods=["POST"])
def route_ingest_topic():
    data = request.get_json(force=True)
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"success": False, "error": "topic is required"}), 400
    try:
        path = ingest_topic(topic)
        filename = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            note = f.read()
        suggestions = get_suggestions(note)
        _log("ingest_topic", topic, filename)
        return jsonify({"success": True, "filename": filename,
                        "word_count": len(note.split()), "suggestions": suggestions})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/ingest/url", methods=["POST"])
def route_ingest_url():
    data = request.get_json(force=True)
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"success": False, "error": "url is required"}), 400
    try:
        path = ingest_url(url)
        filename = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            note = f.read()
        suggestions = get_suggestions(note)
        _log("ingest_url", url, filename)
        return jsonify({"success": True, "filename": filename,
                        "word_count": len(note.split()), "suggestions": suggestions})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/ingest/pdf", methods=["POST"])
def route_ingest_pdf():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"success": False, "error": "File must be a PDF"}), 400

    # Preserve original filename so ingest_pdf derives a clean note name
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)
    try:
        file.save(tmp_path)
        path = ingest_pdf(tmp_path)
        filename = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            note = f.read()
        suggestions = get_suggestions(note)
        _log("ingest_pdf", file.filename, filename)
        return jsonify({"success": True, "filename": filename,
                        "word_count": len(note.split()), "suggestions": suggestions})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── chat ──────────────────────────────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def route_chat():
    data = request.get_json(force=True)
    message = (data.get("message") or "").strip()
    history = data.get("history", [])  # [{role, content}, ...]

    if not message:
        return jsonify({"error": "message is required"}), 400

    try:
        # Keep index fresh (incremental — only re-embeds changed notes)
        build_index()

        chunks = query_index(message, n_results=5)
        sources = sorted({c["source"] for c in chunks})

        context = (
            "\n\n---\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
            if chunks
            else "No relevant context found in the wiki."
        )

        system = (
            "You are a precise assistant that answers questions using only the "
            "provided wiki context. If the answer is not present in the context, "
            "say so clearly. Cite the source file(s) at the end of your answer."
        )

        # Inject fresh context only on the current user turn; prior history is plain
        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({
            "role": "user",
            "content": f"Wiki context:\n\n{context}\n\nQuestion: {message}",
        })

        resp = client.chat.completions.create(model=MODEL, messages=messages)
        answer = resp.choices[0].message.content
        suggestions = get_suggestions(answer)

        _log("chat", message)
        return jsonify({"response": answer, "sources": sources, "suggestions": suggestions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── notes ─────────────────────────────────────────────────────────────────────

@app.route("/notes")
def route_notes():
    try:
        files = sorted(f for f in os.listdir(VAULT_WIKI_PATH) if f.endswith(".md"))
        notes = []
        for f in files:
            path = os.path.join(VAULT_WIKI_PATH, f)
            mtime = datetime.datetime.utcfromtimestamp(os.path.getmtime(path))
            notes.append({"filename": f, "modified": mtime.isoformat() + "Z"})
        return jsonify({"notes": notes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/notes/<filename>")
def route_note(filename):
    # Guard against path traversal
    safe = os.path.basename(filename)
    path = os.path.join(VAULT_WIKI_PATH, safe)
    if not os.path.isfile(path):
        return jsonify({"error": "Note not found"}), 404
    with open(path, encoding="utf-8") as f:
        return jsonify({"content": f.read()})


# ── index regeneration ────────────────────────────────────────────────────────

@app.route("/index", methods=["POST"])
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
        # JSON file fallback
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


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  NeuroNourish running at http://localhost:5001\n")
    app.run(debug=True, host="0.0.0.0", port=5001, use_reloader=False)
