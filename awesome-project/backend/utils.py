"""
Semantic Scholar returns fields like `paperId`, `citationCount`, and
`authors: [{"authorId": ..., "name": ...}]`. Our schemas use snake_case and
flat author name lists. This is the single place that reconciles the two,
so pipeline code can work with raw S2 dicts and only normalize at the edge
(right before building the API response).
"""
import re
from typing import List

import requests

# A DOI is "10." + registrant code + "/" + an opaque suffix.
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
_DOI_RESOLVER = "https://doi.org/"
_DOI_PREFIXES = (
    "https://doi.org/", "http://doi.org/",
    "https://dx.doi.org/", "http://dx.doi.org/",
    "doi:",
)

# Shown to the user whenever a DOI fails validation.
DOI_FORMAT_HINT = (
    'A DOI must look like "10.<registrant>/<suffix>" — for example '
    "10.1145/3292500.3330701 or 10.48550/arXiv.2301.04104. A "
    "https://doi.org/10.... link is also accepted."
)


def valid_doi(value: str) -> List[str]:
    """Extract every DOI in `value` and confirm each is registered by resolving
    it through the canonical DOI resolver at doi.org (the International DOI
    Foundation's proxy -- knows every registered DOI, not just papers in one
    corpus). Returns the DOI list on success; raises ValueError (with a
    user-facing message) listing every DOI that failed."""
    matches = _DOI_RE.findall(normalize_doi(value))

    # Clean up trailing punctuation if the text had a period at the end of a sentence.
    dois = [doi.rstrip(".") for doi in matches]
    if not dois:
        raise ValueError(f"Not a DOI: {value!r}\n\n{DOI_FORMAT_HINT}")

    errors = []
    for doi in dois:
        # Ask doi.org whether it has a redirect for this DOI, but DON'T follow
        # it -- publishers (ACM, IEEE, ...) 403 non-browser clients, so
        # resolving all the way to the landing page is unreliable. A registered
        # DOI => 3xx with a Location; an unregistered one => 404.
        try:
            resp = requests.head(_DOI_RESOLVER + doi, allow_redirects=False, timeout=10)
        except requests.RequestException:
            raise ValueError(
                f"Could not verify {doi} — the DOI resolver is unreachable. "
                "Try again in a moment."
            )
        if resp.status_code == 404:
            errors.append(doi)
        elif resp.status_code not in (200, 301, 302, 303, 307, 308):
            raise ValueError(
                f"Could not verify {doi} — the DOI resolver returned "
                f"{resp.status_code}. Try again in a moment."
            )

    if errors:
        listed = "\n".join(f"  • {d}" for d in errors)
        raise ValueError(f"The following DOI(s) are invalid:\n{listed}\n\n{DOI_FORMAT_HINT}")
    return dois


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
