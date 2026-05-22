"""
wiki-brain CLI — Phase 2

Commands:
  python cli.py ingest topic "Machine learning"
  python cli.py ingest url https://example.com/article
  python cli.py ingest pdf /path/to/paper.pdf
  python cli.py chat "What is apolipoprotein B?"
  python cli.py index
"""
import argparse
import os
import sys

from groq import Groq

from config import GROQ_API_KEY, VAULT_WIKI_PATH, MODEL
from ingest import ingest_topic, ingest_url, ingest_pdf, write_note
from embed import build_index, query_index

client = Groq(api_key=GROQ_API_KEY)


# ── helpers ──────────────────────────────────────────────────────────────────

def _reindex():
    """Re-embed any new/changed notes and report the result."""
    print("[cli] Syncing vector index…")
    n = build_index()
    label = f"{n} file(s) re-indexed" if n else "index already up to date"
    print(f"[cli] {label}")


# ── ingest commands ──────────────────────────────────────────────────────────

def cmd_topic(args):
    ingest_topic(args.name)
    _reindex()


def cmd_url(args):
    ingest_url(args.url)
    _reindex()


def cmd_pdf(args):
    ingest_pdf(args.path)
    _reindex()


# ── chat (RAG) ───────────────────────────────────────────────────────────────

def cmd_chat(args):
    query = " ".join(args.query)

    # Keep the index fresh before answering
    _reindex()

    print(f"[chat] Searching wiki for: {query!r}")
    chunks = query_index(query, n_results=5)

    if not chunks:
        print(
            "[chat] The vector index is empty. "
            "Ingest some notes first with 'python cli.py ingest topic <name>'."
        )
        sys.exit(1)

    # Build context block with per-chunk source labels
    context_blocks = []
    seen_sources: set[str] = set()
    for chunk in chunks:
        context_blocks.append(f"[source: {chunk['source']}]\n{chunk['text']}")
        seen_sources.add(chunk["source"])

    context = "\n\n---\n\n".join(context_blocks)

    system = (
        "You are a precise assistant that answers questions using only the "
        "provided wiki context. If the answer is not present, say so clearly. "
        "At the end of your answer, list the source file(s) you drew from."
    )
    user_msg = f"Wiki context:\n\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
    )

    answer = response.choices[0].message.content
    divider = "─" * 60
    print(f"\n{divider}\n{answer}\n{divider}")


# ── index command ─────────────────────────────────────────────────────────────

def cmd_index(args):
    """Regenerate INDEX.md with a Groq-generated one-line summary per note."""
    md_files = sorted(
        f for f in os.listdir(VAULT_WIKI_PATH)
        if f.endswith(".md") and f != "INDEX.md"
    )

    if not md_files:
        print("[index] No notes found in the vault wiki directory.")
        return

    print(f"[index] Summarising {len(md_files)} note(s) via Groq…")
    entries = []

    for filename in md_files:
        path = os.path.join(VAULT_WIKI_PATH, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Trim to first 1 500 chars so the call is cheap and fast
        snippet = content[:1500]

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return exactly one sentence (20 words or fewer) that summarises "
                        "the topic of the provided wiki note. "
                        "No preamble, no quotation marks — just the sentence."
                    ),
                },
                {"role": "user", "content": snippet},
            ],
        )

        summary = response.choices[0].message.content.strip().rstrip(".")
        note_name = os.path.splitext(filename)[0]
        entries.append(f"- [[{note_name}]] — {summary}")
        print(f"  + {note_name}")

    index_md = "# Wiki Index\n\n" + "\n".join(entries) + "\n"
    path = write_note(index_md, "INDEX.md")
    print(f"\n[index] Written → {path}  ({len(entries)} entries)")


# ── argument parser ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="wiki-brain",
        description="Groq-powered personal wiki CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # ── ingest ──
    ingest_p = sub.add_parser("ingest", help="Add content to the wiki")
    ingest_sub = ingest_p.add_subparsers(dest="source", required=True, metavar="SOURCE")

    tp = ingest_sub.add_parser("topic", help="Ingest a Wikipedia topic")
    tp.add_argument("name", help='Topic name, e.g. "Apolipoprotein B"')
    tp.set_defaults(func=cmd_topic)

    up = ingest_sub.add_parser("url", help="Ingest content from a URL")
    up.add_argument("url", help="URL to fetch")
    up.set_defaults(func=cmd_url)

    pp = ingest_sub.add_parser("pdf", help="Ingest a local PDF")
    pp.add_argument("path", help="Absolute or relative path to the PDF")
    pp.set_defaults(func=cmd_pdf)

    # ── chat ──
    cp = sub.add_parser("chat", help="Ask a question answered from your wiki (RAG)")
    cp.add_argument("query", nargs="+", help="Your question (no quotes needed)")
    cp.set_defaults(func=cmd_chat)

    # ── index ──
    ip = sub.add_parser("index", help="Regenerate INDEX.md with one-line summaries")
    ip.set_defaults(func=cmd_index)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
