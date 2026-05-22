import os
import re
import sys
import requests
from bs4 import BeautifulSoup
import PyPDF2
from groq import Groq

from config import GROQ_API_KEY, VAULT_WIKI_PATH, MODEL

# Initialise Groq client once at module level
client = Groq(api_key=GROQ_API_KEY)

WIKI_HEADERS = {"User-Agent": "wiki-brain/1.0 (educational project; cternullo2229@gmail.com)"}

SYSTEM_PROMPT = (
    "You are a knowledge base writer. Given raw source content, "
    "produce a clean structured Markdown wiki note with:\n"
    "- # Title\n"
    "- ## Summary (3-5 sentences)\n"
    "- ## Key Concepts (bullet points)\n"
    "- ## Details (structured prose)\n"
    "- ## Sources (origin reference)\n"
    "Do not invent facts. Only use what is in the provided content."
)


def _llm_to_markdown(raw_content: str) -> str:
    """Send raw text to Groq and return the structured Markdown response."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw_content},
        ],
    )
    return response.choices[0].message.content


def _to_snake_case(text: str) -> str:
    """Convert arbitrary text to a safe snake_case filename (no extension)."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)   # strip punctuation
    text = re.sub(r"[\s-]+", "_", text)    # spaces/hyphens → underscore
    return text


def write_note(content: str, filename: str) -> str:
    """
    Write *content* to VAULT_WIKI_PATH/filename.
    Creates the directory if it doesn't exist.
    Returns the full file path.
    """
    os.makedirs(VAULT_WIKI_PATH, exist_ok=True)
    file_path = os.path.join(VAULT_WIKI_PATH, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path


def _fetch_wikipedia(topic: str) -> str:
    """
    Fetch article text from the Wikipedia API with a proper User-Agent.
    Returns the full plain-text extract for *topic*.
    """
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": topic,
        "format": "json",
        "explaintext": True,   # plain text, not HTML
        "redirects": 1,
    }
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params=params,
        headers=WIKI_HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page:
        raise ValueError(f"Wikipedia page not found: {topic}")
    return page.get("extract", "")


def ingest_topic(topic: str) -> str:
    """
    Fetch the Wikipedia article for *topic*, pass it through Groq,
    and save the resulting Markdown note to the vault.
    """
    print(f"[ingest_topic] Fetching Wikipedia page for: {topic}")

    raw = _fetch_wikipedia(topic)
    if not raw.strip():
        raise ValueError(f"Empty Wikipedia extract for: {topic}")

    print(f"[ingest_topic] Raw content fetched ({len(raw.split())} words). Sending to Groq…")
    markdown = _llm_to_markdown(f"Source: Wikipedia – {topic}\n\n{raw}")

    filename = _to_snake_case(topic) + ".md"
    path = write_note(markdown, filename)

    word_count = len(markdown.split())
    print(f"[ingest_topic] Written → {path}  ({word_count} words)")
    return path


def ingest_url(url: str) -> str:
    """
    Download *url*, extract paragraph text with BeautifulSoup,
    pass it through Groq, and save the note to the vault.
    """
    print(f"[ingest_url] Fetching URL: {url}")

    response = requests.get(url, headers=WIKI_HEADERS, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Collect text from every <p> tag; skip near-empty ones
    paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
    raw = "\n\n".join(p for p in paragraphs if len(p) > 60)

    if not raw:
        raise ValueError("No usable paragraph text found at the given URL.")

    print(f"[ingest_url] Extracted {len(raw.split())} words. Sending to Groq…")
    markdown = _llm_to_markdown(f"Source: {url}\n\n{raw}")

    # Build a safe filename from the URL path
    slug = _to_snake_case(re.sub(r"https?://", "", url)[:80])
    filename = slug + ".md"
    path = write_note(markdown, filename)

    word_count = len(markdown.split())
    print(f"[ingest_url] Written → {path}  ({word_count} words)")
    return path


def ingest_pdf(pdf_path: str) -> str:
    """
    Extract text from a local PDF at *pdf_path*, pass it through Groq,
    and save the note to the vault.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"[ingest_pdf] Reading PDF: {pdf_path}")

    pages = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)

    raw = "\n\n".join(pages)
    if not raw.strip():
        raise ValueError("No extractable text found in the PDF.")

    print(f"[ingest_pdf] Extracted {len(raw.split())} words. Sending to Groq…")
    markdown = _llm_to_markdown(f"Source: {os.path.basename(pdf_path)}\n\n{raw}")

    pdf_stem = _to_snake_case(os.path.splitext(os.path.basename(pdf_path))[0])
    filename = pdf_stem + ".md"
    path = write_note(markdown, filename)

    word_count = len(markdown.split())
    print(f"[ingest_pdf] Written → {path}  ({word_count} words)")
    return path


# ---------------------------------------------------------------------------
# Smoke test – run with: python ingest.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_topic = "Apolipoprotein B"
    print(f"\n{'='*60}")
    print(f"Smoke test: ingesting Wikipedia topic '{test_topic}'")
    print(f"{'='*60}\n")

    try:
        result_path = ingest_topic(test_topic)
        print(f"\nSUCCESS — note written to: {result_path}")
        sys.exit(0)
    except Exception as exc:
        print(f"\nERROR — {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
