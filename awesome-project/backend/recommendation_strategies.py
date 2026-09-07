"""
Three independently-testable ways to go from seed paper(s) to a candidate
list. Each is a separate function so combinations can be toggled via config
and compared (overlap %, unique finds per strategy) -- see project plan,
section 5.
"""
from typing import Callable, List
import numpy as np
from backend.clients import semantic_scholar as s2


def _cosine(a: List[float], b: List[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)) + 1e-9
    return float(np.dot(a_arr, b_arr) / denom)


def _get_vector(paper: dict):
    embedding = paper.get("embedding")
    if isinstance(embedding, dict):
        return embedding.get("vector")
    return embedding


def specter_similarity(seed_papers: List[dict], candidate_pool: List[dict], top_n: int = 20) -> List[dict]:
    """Strategy 1: rank `candidate_pool` by SPECTER2 cosine similarity to the
    seed(s). Cheapest strategy -- no extra API/LLM calls beyond what's already
    fetched, since embeddings come back on the same search response."""
    seed_vectors = [v for v in (_get_vector(p) for p in seed_papers) if v]
    if not seed_vectors:
        return []

    scored = []
    for cand in candidate_pool:
        vec = _get_vector(cand)
        if not vec:
            continue
        sim = max(_cosine(vec, sv) for sv in seed_vectors)
        scored.append({**cand, "score": sim, "provenance": "seed_specter"})

    scored.sort(key=lambda p: p["score"], reverse=True)
    return scored[:top_n]


def s2_recommender(seed_papers: List[dict], top_n: int = 20) -> List[dict]:
    """Strategy 2: Semantic Scholar's own learned recommender -- folds in
    citation-graph signal, not just embedding similarity. Note: this endpoint
    doesn't expose a per-item relevance score, so every result gets a flat
    prior and citation count / recency differentiate them further downstream
    in ranking_node."""
    seed_ids = [p.get("paperId") for p in seed_papers if p.get("paperId")]
    if not seed_ids:
        return []

    recs = s2.get_recommendations(seed_ids, limit=top_n)
    for r in recs:
        r["provenance"] = "seed_recommender"
        r["score"] = 0.6  # flat prior -- see docstring
    return recs


def agentic_expansion(seed_papers: List[dict], hop_depth: int, branch_k: int,
                       relevance_judge_fn: Callable[[dict], float]) -> List[dict]:
    """Strategy 3: agent walks both references (backward -- foundational work)
    and citations (forward -- newer work) of each seed, judging relevance at
    each step. Only the top `branch_k` survivors (filtered by relevance FIRST,
    then ranked by citation count) expand into the next hop -- this is what
    keeps a highly-cited-but-irrelevant paper from dominating, and bounds the
    traversal to at most branch_k^hop_depth nodes, guaranteeing termination.

    `relevance_judge_fn(paper) -> float` should return a 0-1 relevance score
    (see pipeline/relevance_judge.py for the Gemini-backed implementation).
    """
    frontier = seed_papers
    results: List[dict] = []
    seen_ids = {p.get("paperId") for p in seed_papers if p.get("paperId")}

    for hop in range(hop_depth):
        candidates = []
        for paper in frontier:
            pid = paper.get("paperId")
            if not pid:
                continue
            neighbors = s2.get_references(pid) + s2.get_citations(pid)
            for n in neighbors:
                if n.get("paperId") and n["paperId"] not in seen_ids:
                    candidates.append(n)

        if not candidates:
            break

        # Relevance filter BEFORE citation-count ranking -- a landmark paper
        # cited by everything but off-topic for this query gets dropped here,
        # before citation count ever gets a vote.
        judged = [(c, relevance_judge_fn(c)) for c in candidates]
        judged = [(c, score) for c, score in judged if score >= 0.5]
        judged.sort(key=lambda cs: (cs[1], cs[0].get("citationCount", 0)), reverse=True)

        top = []
        for cand, score in judged[:branch_k]:
            cand["provenance"] = f"seed_agentic_hop{hop + 1}"
            cand["score"] = score
            seen_ids.add(cand.get("paperId"))
            top.append(cand)

        results.extend(top)
        frontier = top
        if not frontier:
            break

    return results
