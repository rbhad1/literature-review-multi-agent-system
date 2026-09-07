"""
Relevance-judging for agentic reference/citation traversal. Kept separate
from recommendation_strategies.py so this specific prompt can be iterated on
independently -- it's the piece most likely to need tuning once you see real
traversal results.
"""
from backend import llm_provider


def judge_relevance(paper: dict, context: str) -> float:
    """Returns a 0-1 relevance score for `paper` against `context` (the
    user's research question). In pure seed-mode with no query text, there's
    nothing to judge relevance against -- this falls back to a neutral
    pass-through and lets citation count do the ranking instead."""
    if not context:
        return 1.0

    title = paper.get("title", "")
    abstract = paper.get("abstract") or ""
    prompt = (
        f'Research question: "{context}"\n\n'
        f"Candidate paper:\nTitle: {title}\nAbstract: {abstract}\n\n"
        f"Rate how relevant this paper is to the research question, from 0.0 "
        f"(irrelevant) to 1.0 (highly relevant).\n\n"
        f'Return JSON: {{"relevance": <float>, "reason": "<one line>"}}'
    )
    result = llm_provider.complete_json("relevance_judge", prompt)
    try:
        return float(result.get("relevance", 0.0))
    except (TypeError, ValueError):
        return 0.0
