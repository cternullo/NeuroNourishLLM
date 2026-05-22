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
import shutil
import tempfile

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from groq import Groq

from config import GROQ_API_KEY, MODEL, VAULT_PATH, VAULT_WIKI_PATH
from embed import build_index, query_index
from ingest import ingest_pdf, ingest_topic, ingest_url, write_note

app = Flask(__name__)
CORS(app)

client = Groq(api_key=GROQ_API_KEY)

ACTIVITY_LOG = os.path.join(VAULT_PATH, "activity_log.json")


# ── helpers ──────────────────────────────────────────────────────────────────

def _word_count(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        return len(f.read().split())


def _log(action: str, detail: str, note_written: str | None = None) -> None:
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "user": request.headers.get("X-Username", "anonymous"),
        "action": action,
        "detail": detail,
        "note_written": note_written,
    }
    os.makedirs(os.path.dirname(ACTIVITY_LOG), exist_ok=True)
    with open(ACTIVITY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


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
        _log("ingest_topic", topic, filename)
        return jsonify({"success": True, "filename": filename, "word_count": _word_count(path)})
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
        _log("ingest_url", url, filename)
        return jsonify({"success": True, "filename": filename, "word_count": _word_count(path)})
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
        _log("ingest_pdf", file.filename, filename)
        return jsonify({"success": True, "filename": filename, "word_count": _word_count(path)})
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

        _log("chat", message)
        return jsonify({"response": answer, "sources": sources})

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
    if not os.path.exists(ACTIVITY_LOG):
        return jsonify({"entries": []})
    with open(ACTIVITY_LOG, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    entries = []
    for line in lines[-50:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return jsonify({"entries": entries})


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  NeuroNourish running at http://localhost:5001\n")
    app.run(debug=True, host="0.0.0.0", port=5001, use_reloader=False)
