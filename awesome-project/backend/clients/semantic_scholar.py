"""
Thin wrapper around the Semantic Scholar Graph API + Recommendations API.
Docs: https://api.semanticscholar.org/api-docs/graph

All requests go through a shared Session with automatic retry/backoff on 429
(rate limit) and 5xx (transient) responses -- S2 throttles even with an API
key, and a single un-retried 429 used to crash a whole pipeline run. Where S2
exposes a batch endpoint we use it instead of a per-item loop.
"""
from typing import List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend import config

FIELDS = "title,abstract,year,authors,citationCount,externalIds,embedding.specter_v2"

_TIMEOUT = 20

# 429 / 5xx -> back off and retry. backoff_factor=1 => waits ~2s, 4s, 8s, 16s
# between attempts; Retry-After on a 429 response overrides that when present.
_retry = Retry(
    total=4,
    backoff_factor=1,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset({"GET", "POST"}),
    respect_retry_after_header=True,
    raise_on_status=False,
)

_session = requests.Session()
_adapter = HTTPAdapter(max_retries=_retry)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


def _headers() -> dict:
    if config.SEMANTIC_SCHOLAR_API_KEY:
        return {"x-api-key": config.SEMANTIC_SCHOLAR_API_KEY}
    return {}


def _get(path_or_url: str, params: dict) -> requests.Response:
    url = path_or_url if path_or_url.startswith("http") else f"{config.S2_BASE_URL}{path_or_url}"
    return _session.get(url, params=params, headers=_headers(), timeout=_TIMEOUT)


def search_papers(query: str, year_min: Optional[int] = None, year_max: Optional[int] = None,
                   limit: int = 20) -> List[dict]:
    params = {"query": query, "fields": FIELDS, "limit": limit}
    if year_min or year_max:
        params["year"] = f"{year_min or ''}-{year_max or ''}"
    resp = _get("/paper/search", params)
    resp.raise_for_status()
    return resp.json().get("data", [])


def get_papers_batch(ids: List[str], fields: str = FIELDS) -> List[Optional[dict]]:
    """Resolve up to 500 paper identifiers in ONE call via /paper/batch.
    Accepts S2 ids, `DOI:...`, `ARXIV:...`, `CorpusId:...`, etc. Returns a list
    positionally aligned with `ids`; entries S2 can't find come back as None.
    """
    ids = [i for i in ids if i]
    if not ids:
        return []
    resp = _session.post(
        f"{config.S2_BASE_URL}/paper/batch",
        params={"fields": fields},
        json={"ids": ids},
        headers=_headers(),
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()  # list aligned with ids, None for misses


def _normalize_identifier(identifier: str) -> Optional[str]:
    """Map a raw user identifier to an S2 batch-endpoint id, or None if it
    needs a title search instead."""
    ident = identifier.strip()
    low = ident.lower()
    if low.startswith("arxiv:"):
        return f"ARXIV:{ident.split(':', 1)[1]}"
    if ident.replace(".", "").isdigit() and "." in ident:  # bare arXiv number e.g. 2301.04104
        return f"ARXIV:{ident}"
    if low.startswith("10.") or "doi.org" in low:
        doi = ident.split("doi.org/")[-1]
        # arXiv DOIs (10.48550/arXiv.NNNN) resolve more reliably as ARXIV ids
        prefix = "10.48550/arxiv."
        if doi.lower().startswith(prefix):
            return f"ARXIV:{doi[len(prefix):]}"
        return f"DOI:{doi}"
    if low.startswith(("doi:", "corpusid:", "mag:", "pmid:", "pmcid:", "url:", "dblp:", "acl:")):
        return ident
    return None  # free text -> title search


def resolve_papers(identifiers: List[str]) -> List[dict]:
    """Batch-resolve a mix of DOIs / arXiv IDs / titles to S2 records.
    Direct ids/DOIs go through /paper/batch in one call; free-text titles fall
    back to a per-item top-hit search. Order is not preserved; misses dropped.
    """
    direct: dict = {}   # position -> normalized id
    text: dict = {}      # position -> raw identifier
    for pos, ident in enumerate(identifiers or []):
        norm = _normalize_identifier(ident)
        (direct if norm else text)[pos] = norm or ident

    out: List[dict] = []
    if direct:
        fetched = get_papers_batch(list(direct.values()))
        out.extend(p for p in fetched if p)
    for raw in text.values():
        hits = search_papers(raw, limit=1)
        if hits:
            out.append(hits[0])
    return out


def resolve_paper(identifier: str) -> Optional[dict]:
    """Single-identifier convenience wrapper around resolve_papers()."""
    resolved = resolve_papers([identifier])
    return resolved[0] if resolved else None


def get_recommendations(seed_paper_ids: List[str], limit: int = 20) -> List[dict]:
    """S2's learned recommender -- folds in citation-graph signal, not just
    embedding similarity. Note: this endpoint doesn't expose a per-item
    relevance score, so callers should treat all results as roughly equally
    relevant and let citation count / recency differentiate them further."""
    if not seed_paper_ids:
        return []
    resp = _session.post(
        config.S2_RECOMMENDATIONS_URL,
        params={"fields": FIELDS, "limit": limit},
        json={"positivePaperIds": seed_paper_ids},
        headers=_headers(),
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("recommendedPapers", [])


def get_references(paper_id: str, limit: int = 50) -> List[dict]:
    """Papers THIS paper cites (backward -- foundational work it builds on)."""
    resp = _get(f"/paper/{paper_id}/references", {"fields": FIELDS, "limit": limit})
    resp.raise_for_status()
    return [r["citedPaper"] for r in resp.json().get("data", []) if r.get("citedPaper")]


def get_citations(paper_id: str, limit: int = 50) -> List[dict]:
    """Papers that cite THIS paper (forward -- newer work building on it)."""
    resp = _get(f"/paper/{paper_id}/citations", {"fields": FIELDS, "limit": limit})
    resp.raise_for_status()
    return [c["citingPaper"] for c in resp.json().get("data", []) if c.get("citingPaper")]
