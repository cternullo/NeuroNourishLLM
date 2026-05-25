import datetime
import os
import re
import sys
import requests
from bs4 import BeautifulSoup
import PyPDF2
from groq import Groq

from config import GROQ_API_KEY, VAULT_WIKI_PATH, MODEL

# Optional Biopython — only needed for PubMed ingestion
try:
    from Bio import Entrez
    Entrez.email = "cternullo2229@gmail.com"
    Entrez.tool = "NeuroNourishLLM"
    _BIOPYTHON_AVAILABLE = True
except ImportError:
    _BIOPYTHON_AVAILABLE = False

client = Groq(api_key=GROQ_API_KEY)

WIKI_HEADERS = {"User-Agent": "wiki-brain/1.0 (educational project; cternullo2229@gmail.com)"}

SYSTEM_PROMPT = (
    "You are a medical research summariser. You summarise published research evidence only. "
    "You do not give medical advice. Always state uncertainty clearly. "
    "Never make clinical recommendations.\n\n"
    "Given raw source content, produce a clean structured Markdown wiki note with:\n"
    "- # Title\n"
    "- ## Summary (3-5 sentences)\n"
    "- ## Key Concepts (bullet points)\n"
    "- ## Details (structured prose)\n"
    "- ## Evidence Quality (assess study design, sample size, and strength of evidence)\n"
    "- ## Limitations (key limitations of the evidence)\n"
    "- ## Sources (origin reference)\n"
    "Do not invent facts. Only use what is in the provided content."
)

BIOMARKER_TAGS = [
    "ApoB", "Lp(a)", "LDL-C", "HDL-C", "total cholesterol", "triglycerides",
]

CONDITION_TAGS = [
    "Alzheimer's disease", "dementia", "cognitive decline",
    "vascular dementia", "cardiovascular disease",
]

MAX_WORDS = 3000


# ── Tag detection ─────────────────────────────────────────────────────────────

def _detect_tags(text: str) -> dict:
    """Return lists of matched biomarker and condition tags (case-insensitive)."""
    lower = text.lower()
    return {
        "biomarkers": [t for t in BIOMARKER_TAGS if t.lower() in lower],
        "conditions": [t for t in CONDITION_TAGS if t.lower() in lower],
    }


# ── YAML frontmatter ──────────────────────────────────────────────────────────

def _yaml_str(v) -> str:
    if v is None or str(v).strip() == "":
        return "null"
    escaped = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _yaml_list(items: list) -> str:
    if not items:
        return "[]"
    parts = [f'"{str(i).replace(chr(92), chr(92)*2).replace(chr(34), chr(92)+chr(34))}"' for i in items]
    return "[" + ", ".join(parts) + "]"


def _build_frontmatter(meta: dict) -> str:
    lines = [
        "---",
        f"title: {_yaml_str(meta.get('title', ''))}",
        f"source_type: {meta.get('source_type', '')}",
        f"source_url: {_yaml_str(meta.get('source_url', ''))}",
        f"doi: {_yaml_str(meta.get('doi', ''))}",
        f"pmid: {_yaml_str(meta.get('pmid', ''))}",
        f"authors: {_yaml_list(meta.get('authors', []))}",
        f"journal: {_yaml_str(meta.get('journal', ''))}",
        f"year: {_yaml_str(meta.get('year', ''))}",
        f"biomarkers: {_yaml_list(meta.get('biomarkers', []))}",
        f"conditions: {_yaml_list(meta.get('conditions', []))}",
        f"created_by: {_yaml_str(meta.get('created_by', ''))}",
        f"created_at: {datetime.datetime.utcnow().isoformat()}",
        "---",
        "",
    ]
    return "\n".join(lines) + "\n"


# ── LLM helpers ───────────────────────────────────────────────────────────────

def chunk_text(text: str, max_words: int = MAX_WORDS) -> list:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _llm_to_markdown(raw_content: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw_content},
        ],
    )
    return response.choices[0].message.content


def _llm_to_markdown_chunked(raw_content: str, source_label: str) -> tuple[str, int]:
    chunks = chunk_text(raw_content)
    if len(chunks) == 1:
        return _llm_to_markdown(f"Source: {source_label}\n\n{raw_content}"), 1
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"  chunk {i}/{len(chunks)} ({len(chunk.split())} words)…")
        result = _llm_to_markdown(f"Source: {source_label} (part {i} of {len(chunks)})\n\n{chunk}")
        parts.append(f"## Part {i}\n\n{result}")
    return "\n\n---\n\n".join(parts), len(chunks)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _to_snake_case(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    return text


def write_note(content: str, filename: str) -> str:
    os.makedirs(VAULT_WIKI_PATH, exist_ok=True)
    file_path = os.path.join(VAULT_WIKI_PATH, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path


# ── Wikipedia fetch ───────────────────────────────────────────────────────────

def _fetch_wikipedia(topic: str) -> str:
    params = {
        "action": "query", "prop": "extracts", "titles": topic,
        "format": "json", "explaintext": True, "redirects": 1,
    }
    r = requests.get("https://en.wikipedia.org/w/api.php",
                     params=params, headers=WIKI_HEADERS, timeout=15)
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page:
        raise ValueError(f"Wikipedia page not found: {topic}")
    return page.get("extract", "")


# ── PubMed fetch ──────────────────────────────────────────────────────────────

def _fetch_pubmed_meta(pmid: str) -> dict:
    if not _BIOPYTHON_AVAILABLE:
        raise ImportError("biopython is required: pip install biopython")

    handle = Entrez.efetch(db="pubmed", id=str(pmid), rettype="xml", retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    if not records.get("PubmedArticle"):
        raise ValueError(f"No PubMed article found for PMID: {pmid}")

    article = records["PubmedArticle"][0]["MedlineCitation"]["Article"]

    title = str(article.get("ArticleTitle", "")).strip()

    abstract_parts = article.get("Abstract", {}).get("AbstractText", [])
    if isinstance(abstract_parts, list):
        abstract = " ".join(str(p) for p in abstract_parts)
    else:
        abstract = str(abstract_parts)
    abstract = abstract.strip()

    journal_info = article.get("Journal", {})
    journal = str(journal_info.get("Title", "")).strip()
    pub_date = journal_info.get("JournalIssue", {}).get("PubDate", {})
    year = str(pub_date.get("Year", pub_date.get("MedlineDate", "")[:4])).strip()

    authors = []
    for a in article.get("AuthorList", [])[:6]:
        last = str(a.get("LastName", "")).strip()
        initials = str(a.get("Initials", a.get("ForeName", ""))).strip()
        if last:
            authors.append(f"{last} {initials}".strip())

    return {
        "title": title, "abstract": abstract, "journal": journal,
        "year": year, "authors": authors, "pmid": str(pmid),
    }


# ── CrossRef / DOI fetch ──────────────────────────────────────────────────────

def _fetch_doi_meta(doi: str) -> dict:
    url = f"https://api.crossref.org/works/{doi}"
    r = requests.get(url, headers=WIKI_HEADERS, timeout=15)
    r.raise_for_status()
    work = r.json().get("message", {})

    title = work.get("title", [""])[0] if work.get("title") else ""

    authors = []
    for a in work.get("author", [])[:6]:
        family = a.get("family", "").strip()
        given = a.get("given", "").strip()
        if family:
            authors.append(f"{family} {given}".strip())

    containers = work.get("container-title", [])
    journal = containers[0] if containers else ""

    year = ""
    for key in ("published-print", "published-online", "created"):
        parts = work.get(key, {}).get("date-parts", [[]])
        if parts and parts[0]:
            year = str(parts[0][0])
            break

    abstract = work.get("abstract", "")
    abstract = re.sub(r"<[^>]+>", " ", abstract)
    for entity, char in [("&lt;","<"),("&gt;",">"),("&amp;","&"),("&nbsp;"," ")]:
        abstract = abstract.replace(entity, char)
    abstract = re.sub(r"\s+", " ", abstract).strip()

    return {
        "title": title, "abstract": abstract, "journal": journal,
        "year": year, "authors": authors, "doi": doi,
    }


# ── Public ingest functions ───────────────────────────────────────────────────

def ingest_topic(topic: str, created_by: str = "") -> str:
    print(f"[ingest_topic] Fetching Wikipedia: {topic}")
    raw = _fetch_wikipedia(topic)
    if not raw.strip():
        raise ValueError(f"Empty Wikipedia extract for: {topic}")

    print(f"[ingest_topic] {len(raw.split())} words → Groq…")
    markdown, n_chunks = _llm_to_markdown_chunked(raw, f"Wikipedia – {topic}")

    tags = _detect_tags(raw + " " + markdown)
    frontmatter = _build_frontmatter({
        "title": topic,
        "source_type": "topic",
        "source_url": f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
        "created_by": created_by,
        **tags,
    })

    filename = _to_snake_case(topic) + ".md"
    path = write_note(frontmatter + markdown, filename)
    print(f"[ingest_topic] Written → {path} ({n_chunks} chunk(s))")
    return path


def ingest_url(url: str, created_by: str = "") -> str:
    print(f"[ingest_url] Fetching: {url}")
    response = requests.get(url, headers=WIKI_HEADERS, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    page_title_tag = soup.find("title")
    page_title = page_title_tag.get_text(strip=True) if page_title_tag else url

    paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
    raw = "\n\n".join(p for p in paragraphs if len(p) > 60)
    if not raw:
        raise ValueError("No usable paragraph text found at the given URL.")

    print(f"[ingest_url] {len(raw.split())} words → Groq…")
    markdown, n_chunks = _llm_to_markdown_chunked(raw, url)

    tags = _detect_tags(raw + " " + markdown)
    frontmatter = _build_frontmatter({
        "title": page_title[:200],
        "source_type": "url",
        "source_url": url,
        "created_by": created_by,
        **tags,
    })

    slug = _to_snake_case(re.sub(r"https?://", "", url)[:80])
    filename = slug + ".md"
    path = write_note(frontmatter + markdown, filename)
    print(f"[ingest_url] Written → {path} ({n_chunks} chunk(s))")
    return path


def ingest_pdf(pdf_path: str, created_by: str = "") -> str:
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"[ingest_pdf] Reading: {pdf_path}")
    pages_text = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

    raw = "\n\n".join(pages_text)
    if not raw.strip():
        raise ValueError("No extractable text found in the PDF.")

    print(f"[ingest_pdf] {len(raw.split())} words → Groq…")
    markdown, n_chunks = _llm_to_markdown_chunked(raw, os.path.basename(pdf_path))

    tags = _detect_tags(raw + " " + markdown)
    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    frontmatter = _build_frontmatter({
        "title": pdf_stem.replace("_", " ").replace("-", " "),
        "source_type": "pdf",
        "source_url": os.path.basename(pdf_path),
        "created_by": created_by,
        **tags,
    })

    filename = _to_snake_case(pdf_stem) + ".md"
    path = write_note(frontmatter + markdown, filename)
    print(f"[ingest_pdf] Written → {path} ({n_chunks} chunk(s))")
    return path


def ingest_pubmed(pmid: str, created_by: str = "") -> str:
    print(f"[ingest_pubmed] Fetching PMID: {pmid}")
    meta = _fetch_pubmed_meta(pmid)

    content = f"Title: {meta['title']}\n\nAbstract: {meta['abstract']}"
    if not meta["abstract"].strip():
        print("[ingest_pubmed] No abstract — using title/journal only.")
        content = f"Title: {meta['title']}\nJournal: {meta['journal']}\nYear: {meta['year']}"

    print(f"[ingest_pubmed] Sending to Groq…")
    markdown, n_chunks = _llm_to_markdown_chunked(
        content, f"PubMed PMID {pmid} — {meta['title']}"
    )

    tags = _detect_tags(content + " " + markdown)
    frontmatter = _build_frontmatter({
        "title": meta["title"],
        "source_type": "pubmed",
        "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "pmid": meta["pmid"],
        "authors": meta["authors"],
        "journal": meta["journal"],
        "year": meta["year"],
        "created_by": created_by,
        **tags,
    })

    filename = f"pmid_{pmid}.md"
    path = write_note(frontmatter + markdown, filename)
    print(f"[ingest_pubmed] Written → {path} ({n_chunks} chunk(s))")
    return path


def ingest_doi(doi: str, created_by: str = "") -> str:
    # Strip doi.org URL prefix if pasted in full
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi).strip()
    print(f"[ingest_doi] Fetching DOI: {doi}")

    meta = _fetch_doi_meta(doi)

    content = f"Title: {meta['title']}\n\nAbstract: {meta['abstract']}"
    if not meta["abstract"].strip():
        print("[ingest_doi] No abstract — using title/journal only.")
        content = f"Title: {meta['title']}\nJournal: {meta['journal']}\nYear: {meta['year']}"

    print(f"[ingest_doi] Sending to Groq…")
    markdown, n_chunks = _llm_to_markdown_chunked(
        content, f"DOI {doi} — {meta['title']}"
    )

    tags = _detect_tags(content + " " + markdown)
    frontmatter = _build_frontmatter({
        "title": meta["title"],
        "source_type": "doi",
        "source_url": f"https://doi.org/{doi}",
        "doi": doi,
        "authors": meta["authors"],
        "journal": meta["journal"],
        "year": meta["year"],
        "created_by": created_by,
        **tags,
    })

    slug = "doi_" + _to_snake_case(doi[:60])
    filename = slug + ".md"
    path = write_note(frontmatter + markdown, filename)
    print(f"[ingest_doi] Written → {path} ({n_chunks} chunk(s))")
    return path


# ---------------------------------------------------------------------------
# Smoke test — run with: python ingest.py
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
