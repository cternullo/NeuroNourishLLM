"""
Vault cleanup script — removes notes whose frontmatter title contains
any of the keywords in IRRELEVANT_KEYWORDS.

Usage:
    python cleanup_vault.py

Add keywords to IRRELEVANT_KEYWORDS to flag more notes for deletion.
The match is case-insensitive and checks the frontmatter title field.
"""

import os
import re
import sys

# ── Configuration ──────────────────────────────────────────────────────────────

IRRELEVANT_KEYWORDS = [
    "gallbladder",
    "microbiology",
    "malaria",
    "mobile media",
    "penile",
    "osteomyelitis",
    "biotin",
    "vancomycin",
    "oat starch",
    "phytoplankton",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_vault_wiki_path() -> str:
    vault = os.environ.get("VAULT_PATH", "/app")
    return os.path.join(vault, "wiki")


def _parse_frontmatter_title(content: str) -> str:
    """Extract the title field from YAML frontmatter, or return ''."""
    if not content.startswith("---\n"):
        return ""
    end = content.find("\n---\n", 4)
    if end == -1:
        return ""
    for line in content[4:end].split("\n"):
        if not line.startswith("title:"):
            continue
        raw = line.partition(":")[2].strip()
        if raw.startswith('"') and raw.endswith('"'):
            return raw[1:-1].replace('\\"', '"')
        if raw != "null":
            return raw
    return ""


def _matches_keyword(title: str, keywords: list) -> str | None:
    """Return the first matching keyword, or None."""
    lower = title.lower()
    return next((kw for kw in keywords if kw.lower() in lower), None)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_cleanup(vault_wiki_path: str, keywords: list, dry_run: bool = False) -> list:
    """
    Walk all .md files under vault_wiki_path and delete those whose
    frontmatter title contains any of the given keywords.
    Returns list of deleted filenames.
    """
    if not os.path.isdir(vault_wiki_path):
        print(f"[cleanup] Vault wiki path not found: {vault_wiki_path}", file=sys.stderr)
        return []

    deleted = []
    for fname in sorted(os.listdir(vault_wiki_path)):
        if not fname.endswith(".md"):
            continue
        fpath = os.path.join(vault_wiki_path, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue
        title = _parse_frontmatter_title(content)
        matched = _matches_keyword(title, keywords)
        if matched:
            if dry_run:
                print(f"  [dry-run] Would delete: {fname!r}  (title: {title!r}, keyword: {matched!r})")
            else:
                os.remove(fpath)
                print(f"  Deleted: {fname!r}  (title: {title!r}, keyword: {matched!r})")
            deleted.append(fname)

    return deleted


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    vault_wiki = _get_vault_wiki_path()

    print(f"Vault wiki path: {vault_wiki}")
    print(f"Keywords ({len(IRRELEVANT_KEYWORDS)}): {', '.join(IRRELEVANT_KEYWORDS)}")
    if dry_run:
        print("DRY RUN — no files will be deleted\n")
    else:
        print()

    deleted = run_cleanup(vault_wiki, IRRELEVANT_KEYWORDS, dry_run=dry_run)

    if not deleted:
        print("No matching notes found.")
    else:
        print(f"\n{'Would delete' if dry_run else 'Deleted'} {len(deleted)} note(s).")

    # Invalidate biomarker cache (if app is running in the same process space)
    try:
        import app as _app
        _app._biomarker_cache = None
        print("[cleanup] Biomarker cache invalidated.")
    except Exception:
        pass
