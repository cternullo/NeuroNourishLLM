"""
Author outreach helpers for NeuroNourishLLM.
Searches PubMed + CrossRef for papers and drafts professional outreach emails.
"""

import json
import re

import requests
from groq import Groq

from config import GROQ_API_KEY, MODEL

try:
    from Bio import Entrez
    Entrez.email = "cternullo2229@gmail.com"
    Entrez.tool = "NeuroNourishLLM"
    _BIO_OK = True
except ImportError:
    _BIO_OK = False

client = Groq(api_key=GROQ_API_KEY)

_EMAIL_RE = re.compile(r"[\w.+\-]+@[\w\-]+\.[a-zA-Z]{2,}")
_CR_HEADERS = {"User-Agent": "NeuroNourishLLM/1.0 (cternullo2229@gmail.com)"}


def _email_from_text(text: str):
    m = _EMAIL_RE.search(text or "")
    return m.group(0) if m else None


def _pubmed_search(topic: str, max_results: int) -> list:
    if not _BIO_OK:
        return []
    papers = []
    try:
        handle = Entrez.esearch(db="pubmed", term=f"{topic}[Title/Abstract]", retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        ids = record.get("IdList", [])
        if not ids:
            return []

        fetch = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="xml", retmode="xml")
        fetch_rec = Entrez.read(fetch)
        fetch.close()

        for art_rec in fetch_rec.get("PubmedArticle", []):
            medline = art_rec.get("MedlineCitation", {})
            art = medline.get("Article", {})
            pmid = str(medline.get("PMID", ""))
            title = str(art.get("ArticleTitle", ""))
            journal = str(art.get("Journal", {}).get("Title", ""))
            pub_date = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            year = str(pub_date.get("Year", pub_date.get("MedlineDate", "")[:4]))

            abstract_parts = art.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(str(p) for p in abstract_parts).strip()[:600]

            doi = ""
            for id_item in art_rec.get("PubmedData", {}).get("ArticleIdList", []):
                if str(getattr(id_item, "attributes", {}).get("IdType", "")) == "doi":
                    doi = str(id_item)
                    break

            authors = []
            email = None
            affiliations = []
            for auth in art.get("AuthorList", []):
                last = str(auth.get("LastName", ""))
                fore = str(auth.get("ForeName", ""))
                name = (fore + " " + last).strip() if fore else last
                if name:
                    authors.append(name)
                for aff_info in auth.get("AffiliationInfo", []):
                    aff_str = str(aff_info.get("Affiliation", ""))
                    affiliations.append(aff_str)
                    if not email:
                        email = _email_from_text(aff_str)

            papers.append({
                "pmid": pmid, "doi": doi,
                "title": title, "authors": authors,
                "journal": journal, "year": year,
                "abstract": abstract, "email": email,
                "affiliations": affiliations[:2],
                "source": "pubmed",
            })
    except Exception as e:
        print(f"[outreach] PubMed error: {e}")
    return papers


def _crossref_search(topic: str, max_results: int) -> list:
    papers = []
    try:
        resp = requests.get(
            "https://api.crossref.org/works",
            params={
                "query": topic, "rows": max_results,
                "select": "DOI,title,author,container-title,published,abstract",
            },
            headers=_CR_HEADERS, timeout=12,
        )
        if not resp.ok:
            return []
        for item in resp.json().get("message", {}).get("items", []):
            doi = item.get("DOI", "")
            if not doi:
                continue
            titles = item.get("title", [])
            title = titles[0] if titles else ""
            journals = item.get("container-title", [])
            journal = journals[0] if journals else ""
            dp = item.get("published", {}).get("date-parts", [[""]])[0]
            year = str(dp[0]) if dp else ""
            authors = []
            email = None
            for auth in item.get("author", []):
                given = auth.get("given", "")
                family = auth.get("family", "")
                name = (given + " " + family).strip() if given else family
                if name:
                    authors.append(name)
                for aff in auth.get("affiliation", []):
                    aff_str = aff.get("name", "")
                    if not email:
                        email = _email_from_text(aff_str)
            abstract = re.sub(r"<[^>]+>", "", item.get("abstract", ""))[:600]
            papers.append({
                "pmid": "", "doi": doi,
                "title": title, "authors": authors,
                "journal": journal, "year": year,
                "abstract": abstract, "email": email,
                "affiliations": [], "source": "crossref",
            })
    except Exception as e:
        print(f"[outreach] CrossRef error: {e}")
    return papers


def search_papers(topic: str, max_results: int = 10) -> list:
    """
    Search PubMed then CrossRef. Deduplicate by DOI.
    Returns up to max_results paper dicts with keys:
    pmid, doi, title, authors, journal, year, abstract, email, affiliations, source
    """
    seen_dois = set()
    results = []

    for paper in _pubmed_search(topic, max_results):
        key = paper["doi"] or f"pmid:{paper['pmid']}"
        if key not in seen_dois:
            seen_dois.add(key)
            results.append(paper)

    for paper in _crossref_search(topic, max_results):
        if paper["doi"] not in seen_dois:
            seen_dois.add(paper["doi"])
            results.append(paper)

    return results[:max_results]


def draft_outreach_email(
    author_name: str,
    paper_title: str,
    journal: str,
    year: str,
    topic: str,
    sender_name: str,
    sender_email: str,
) -> dict:
    """
    Use Groq to draft a professional outreach email.
    Returns {subject, body}. Never fabricates emails or credentials.
    """
    prompt = (
        f"Draft a professional email from a university research student "
        f"to a researcher requesting access to their data.\n\n"
        f"Researcher name: {author_name}\n"
        f'Their paper: "{paper_title}"'
        + (f" ({journal}, {year})" if journal else "")
        + f"\nResearch topic: {topic}\n"
        f"Student sender: {sender_name} ({sender_email})\n\n"
        f"Requirements:\n"
        f"- Polite and professional tone\n"
        f"- Clearly state the sender is a student researcher\n"
        f"- Reference the specific paper by its exact title\n"
        f"- Explain the research purpose clearly\n"
        f"- Request access to dataset or methodology\n"
        f"- Under 200 words in the body\n"
        f"- Never misrepresent credentials\n\n"
        f'Respond with ONLY valid JSON: {{"subject": "...", "body": "..."}}'
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You help university research students draft professional emails "
                        "to researchers. Be honest, concise, and respectful. "
                        "Always respond with a JSON object only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        return {
            "subject": str(parsed.get("subject", f"Research Inquiry — {paper_title[:60]}")),
            "body": str(parsed.get("body", "")),
        }
    except Exception as e:
        print(f"[outreach] draft error: {e}")
        return {
            "subject": f"Research Data Inquiry — {paper_title[:60]}",
            "body": (
                f"Dear {author_name},\n\n"
                f"I am a student researcher and came across your paper \"{paper_title}\". "
                f"I am researching {topic} and would appreciate access to your dataset "
                f"or methodology.\n\n"
                f"Best regards,\n{sender_name}"
            ),
        }
