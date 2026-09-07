"""
Thin wrapper around the Semantic Scholar Graph API + Recommendations API.
Docs: https://api.semanticscholar.org/api-docs/graph
"""
from typing import List, Optional
import requests
from backend import config
import json

FIELDS = "title,abstract,year,authors,citationCount,externalIds,embedding.specter_v2"


def _headers() -> dict:
    if config.SEMANTIC_SCHOLAR_API_KEY:
        print("Using api key...")
        return {"x-api-key": config.SEMANTIC_SCHOLAR_API_KEY}
    else:
        return {}


def search_papers(query: str, year_min: Optional[int] = None, year_max: Optional[int] = None,
                   limit: int = 20) -> List[dict]:
    params = {"query": query, "fields": FIELDS, "limit": limit}
    if year_min or year_max:
        params["year"] = f"{year_min or ''}-{year_max or ''}"
    resp = requests.get(f"{config.S2_BASE_URL}/paper/search", params=params, headers=_headers(), timeout=15)
    resp.raise_for_status()
    return resp.json().get("data", [])


def resolve_paper(identifier: str) -> Optional[dict]:
    """Resolve a title / arXiv ID / DOI to a canonical S2 paper record.
    IDs and DOIs are looked up directly; anything else falls back to a
    title search and takes the best (top) match.
    """
    identifier = identifier.strip()

    if identifier.lower().startswith("arxiv:") or (identifier.replace(".", "").isdigit() and "." in identifier):
        pid = identifier if identifier.lower().startswith("arxiv:") else f"arXiv:{identifier}"
        resp = requests.get(f"{config.S2_BASE_URL}/paper/{pid}", params={"fields": FIELDS},
                            headers=_headers(), timeout=15)
        if resp.ok:
            return resp.json()

    if identifier.lower().startswith("10.") or "doi.org" in identifier.lower():
        doi = identifier.split("doi.org/")[-1]
        resp = requests.get(f"{config.S2_BASE_URL}/paper/DOI:{doi}", params={"fields": FIELDS},
                            headers=_headers(), timeout=15)
        if resp.ok:
            return resp.json()

    results = search_papers(identifier, limit=1)
    return results[0] if results else None


def get_recommendations(seed_paper_ids: List[str], limit: int = 20) -> List[dict]:
    """S2's learned recommender -- folds in citation-graph signal, not just
    embedding similarity. Note: this endpoint doesn't expose a per-item
    relevance score, so callers should treat all results as roughly equally
    relevant and let citation count / recency differentiate them further."""
    if not seed_paper_ids:
        return []
    resp = requests.post(
        config.S2_RECOMMENDATIONS_URL,
        params={"fields": FIELDS, "limit": limit},
        json={"positivePaperIds": seed_paper_ids},
        headers=_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("recommendedPapers", [])


def get_references(paper_id: str, limit: int = 50) -> List[dict]:
    """Papers THIS paper cites (backward -- foundational work it builds on)."""
    resp = requests.get(
        f"{config.S2_BASE_URL}/paper/{paper_id}/references",
        params={"fields": FIELDS, "limit": limit},
        headers=_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return [r["citedPaper"] for r in resp.json().get("data", []) if r.get("citedPaper")]


def get_citations(paper_id: str, limit: int = 50) -> List[dict]:
    """Papers that cite THIS paper (forward -- newer work building on it)."""
    resp = requests.get(
        f"{config.S2_BASE_URL}/paper/{paper_id}/citations",
        params={"fields": FIELDS, "limit": limit},
        headers=_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return [c["citingPaper"] for c in resp.json().get("data", []) if c.get("citingPaper")]

