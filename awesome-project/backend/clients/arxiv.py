"""
Thin wrapper around the arXiv API -- used ONLY for the monitoring feature.
Unlike Semantic Scholar (which can lag days-weeks behind new arXiv postings),
arXiv's own API has no indexing lag, so it's the right tool specifically for
"has anything new shown up since I last checked" -- not for general search,
which is Semantic Scholar's job (see clients/semantic_scholar.py).
"""
from datetime import datetime
from typing import List, Optional
import requests
import feedparser

ARXIV_API = "http://export.arxiv.org/api/query"


def check_new_papers(category: str = "cs.LG", since: Optional[datetime] = None,
                      max_results: int = 20) -> List[dict]:
    """Fetch the most recent papers in a category, newest first.
    Caller compares against a stored `since` timestamp to find genuinely new ones.
    Not wired into the main /review pipeline -- this is a separate, on-demand
    "check for updates" call (see project plan, section 3: monitoring was
    scoped down from a scheduled job to an on-demand check).
    """
    params = {
        "search_query": f"cat:{category}",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    }
    resp = requests.get(ARXIV_API, params=params, timeout=15)
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)

    papers = []
    for entry in feed.entries:
        published = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ")
        if since and published <= since:
            continue
        papers.append({
            "title": entry.title,
            "abstract": entry.summary,
            "authors": [a.name for a in entry.authors],
            "arxiv_id": entry.id.split("/abs/")[-1],
            "published": published.isoformat(),
        })
    return papers
