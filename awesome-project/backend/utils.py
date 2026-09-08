"""
Semantic Scholar returns fields like `paperId`, `citationCount`, and
`authors: [{"authorId": ..., "name": ...}]`. Our schemas use snake_case and
flat author name lists. This is the single place that reconciles the two,
so pipeline code can work with raw S2 dicts and only normalize at the edge
(right before building the API response).
"""
import re
from typing import List

# A DOI is "10." + registrant code + "/" + an opaque suffix. This is the
# common practical pattern (Crossref's own recommendation), not the full spec.
_DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:a-z0-9]+$", re.IGNORECASE)
_DOI_PREFIXES = (
    "https://doi.org/", "http://doi.org/",
    "https://dx.doi.org/", "http://dx.doi.org/",
    "doi:",
)


def normalize_doi(value: str) -> str:
    """Strip a URL / `doi:` wrapper so just the bare `10.x/...` identifier is left."""
    s = (value or "").strip()
    low = s.lower()
    for prefix in _DOI_PREFIXES:
        if low.startswith(prefix):
            return s[len(prefix):]
    return s


def is_doi(value: str) -> bool:
    """True if `value` (optionally URL/`doi:`-wrapped) is a well-formed DOI."""
    return bool(_DOI_RE.match(normalize_doi(value)))


def looks_like_doi_attempt(value: str) -> bool:
    """True if the user clearly meant this to be a DOI (so a malformed one
    should be rejected rather than passed through as a title search)."""
    low = (value or "").strip().lower()
    return low.startswith("10.") or low.startswith("doi:") or "doi.org/" in low


def normalize_paper(raw: dict) -> dict:
    embedding = raw.get("embedding")
    vector = embedding.get("vector") if isinstance(embedding, dict) else embedding

    authors_raw = raw.get("authors") or []
    authors: List[str] = [
        a.get("name") for a in authors_raw if isinstance(a, dict) and a.get("name")
    ] or [a for a in authors_raw if isinstance(a, str)]

    external_ids = raw.get("externalIds") or {}

    return {
        "paper_id": raw.get("paperId") or raw.get("paper_id") or "",
        "title": raw.get("title") or "",
        "authors": authors,
        "abstract": raw.get("abstract"),
        "year": raw.get("year"),
        "citation_count": raw.get("citationCount", raw.get("citation_count", 0)) or 0,
        "url": raw.get("url") or (f"https://doi.org/{external_ids['DOI']}" if external_ids.get("DOI") else None),
        "provenance": raw.get("provenance", "query"),
        "score": raw.get("score", 0.0) or 0.0,
        "summary": raw.get("summary"),
        "limitations": raw.get("limitations"),
        "future_work": raw.get("future_work"),
        "_vector": vector,  # not part of the response schema, dropped by pydantic; kept for debugging
    }
