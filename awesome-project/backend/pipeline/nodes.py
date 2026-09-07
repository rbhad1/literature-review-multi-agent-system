"""
The LangGraph node functions -- this is the "fat orchestrator" side of the
project. Each function takes the shared PipelineState, does one job, and
returns the (mutated) state for the next node. See graph.py for how these
are wired into a sequence, and pipeline/state.py for the shared schema.
"""
import math
from datetime import datetime, timezone

from backend import config, llm_provider
from backend.clients import semantic_scholar as s2
from backend.pipeline.state import PipelineState
from backend.pipeline.relevance_judge import judge_relevance
from backend.recommendation_strategies import specter_similarity, s2_recommender, agentic_expansion


# ---------------------------------------------------------------------------
# Planner: infer mode from what's populated, expand the query if present.
# No mode is ever asked of the user -- see project plan, section 4.
# ---------------------------------------------------------------------------
def planner_node(state: PipelineState) -> PipelineState:
    has_query = bool(state.get("query"))
    has_seeds = bool(state.get("seed_paper_ids"))

    if has_query and has_seeds:
        state["mode"] = "hybrid"
    elif has_query:
        state["mode"] = "keyword"
    elif has_seeds:
        state["mode"] = "seed"
    else:
        raise ValueError("Either `query` or `seed_paper_ids` must be provided.")

    if has_query:
        preset = config.BREADTH_PRESETS[state.get("breadth", 3)]
        n_variants = preset["query_variants"]
        prompt = (
            f"Generate {n_variants} distinct search-engine queries for finding "
            f"CS/ML research papers relevant to this research question. Vary "
            f'terminology and specificity.\n\nResearch question: "{state["query"]}"\n\n'
            f'Return JSON: {{"queries": ["...", "..."]}}'
        )
        result = llm_provider.complete_json("planner", prompt)
        variants = result.get("queries") or [state["query"]]
        state["query_variants"] = variants[:n_variants]
    else:
        state["query_variants"] = []

    return state


# ---------------------------------------------------------------------------
# Retrieval: hit Semantic Scholar for the keyword path and/or the seed path.
# ---------------------------------------------------------------------------
def retrieval_node(state: PipelineState) -> PipelineState:
    preset = config.BREADTH_PRESETS[state.get("breadth", 3)]

    # --- keyword path ---
    query_candidates = []
    seen_ids = set()
    variants = state.get("query_variants") or []
    per_variant_limit = max(preset["pool_size"] // max(len(variants), 1), 5)

    for variant in variants:
        papers = s2.search_papers(variant, state.get("year_min"), state.get("year_max"),
                                   limit=per_variant_limit)
        for idx, paper in enumerate(papers):
            pid = paper.get("paperId")
            if pid and pid not in seen_ids:
                paper["provenance"] = "query"
                paper["score"] = 1.0 / (idx + 1)  # relevance proxy from S2's own ranked order
                seen_ids.add(pid)
                query_candidates.append(paper)

    state["query_candidates"] = query_candidates

    # --- seed path ---
    seed_ids = state.get("seed_paper_ids") or []
    resolved = [s2.resolve_paper(sid) for sid in seed_ids]
    resolved = [r for r in resolved if r]
    state["seed_papers_resolved"] = resolved

    seed_candidates = []
    if resolved:
        # Strategy 1: SPECTER2 similarity over a candidate pool. Reuse the
        # keyword-search pool if we have one; otherwise pull each seed's own
        # neighbors as the comparison pool.
        pool = list(query_candidates)
        if not pool:
            for r in resolved:
                pid = r.get("paperId")
                if pid:
                    pool.extend(s2.get_references(pid) + s2.get_citations(pid))
        seed_candidates += specter_similarity(resolved, pool, top_n=preset["branch_k"] * 2)

        # Strategy 2: S2's own recommender.
        seed_candidates += s2_recommender(resolved, top_n=preset["branch_k"] * 2)

        # Strategy 3: agentic reference/citation traversal, bounded by
        # hop_depth and branch_k from the breadth preset.
        seed_candidates += agentic_expansion(
            resolved, preset["hop_depth"], preset["branch_k"],
            relevance_judge_fn=lambda paper: judge_relevance(paper, state.get("query") or ""),
        )

    state["seed_candidates"] = seed_candidates
    return state


# ---------------------------------------------------------------------------
# Ranking: relevance (from strategy) + citation count + recency.
# ---------------------------------------------------------------------------
def _recency_score(year, now_year=None) -> float:
    if not year:
        return 0.0
    now_year = now_year or datetime.now(timezone.utc).year
    age = max(now_year - year, 0)
    return math.exp(-age / 8.0)  # rough decay -- tune later per project plan


def _rank_list(papers: list) -> list:
    scored = []
    for paper in papers:
        citation_score = math.log1p(paper.get("citationCount", 0) or 0)
        recency = _recency_score(paper.get("year"))
        base_relevance = paper.get("score", 0.0) or 0.0
        combined = 0.5 * base_relevance + 0.3 * (citation_score / 10) + 0.2 * recency
        scored.append({**paper, "score": combined})
    scored.sort(key=lambda p: p["score"], reverse=True)
    return scored


def ranking_node(state: PipelineState) -> PipelineState:
    if state.get("query_candidates"):
        state["query_ranked"] = _rank_list(state["query_candidates"])
    if state.get("seed_candidates"):
        state["seed_ranked"] = _rank_list(state["seed_candidates"])
    return state


# ---------------------------------------------------------------------------
# Extraction: per-paper summary / limitations / future work.
# ---------------------------------------------------------------------------
def _extract_one(paper: dict) -> dict:
    prompt = (
        f"Title: {paper.get('title')}\n"
        f"Abstract: {paper.get('abstract') or 'N/A'}\n\n"
        f"From the abstract, extract:\n"
        f"1. A 2-3 sentence summary\n"
        f"2. Stated or implied limitations\n"
        f"3. Suggested future work\n\n"
        f'Return JSON: {{"summary": "...", "limitations": "...", "future_work": "..."}}'
    )
    result = llm_provider.complete_json("extraction", prompt)
    return {**paper, **result}


def extraction_node(state: PipelineState) -> PipelineState:
    n = config.TOP_N_TO_EXTRACT
    if state.get("query_ranked"):
        state["query_extracted"] = [_extract_one(p) for p in state["query_ranked"][:n]]
    if state.get("seed_ranked"):
        state["seed_extracted"] = [_extract_one(p) for p in state["seed_ranked"][:n]]
    return state


# ---------------------------------------------------------------------------
# Synthesis: cross-paper themes, gaps, limitations narrative.
# ---------------------------------------------------------------------------
def synthesis_node(state: PipelineState) -> PipelineState:
    all_papers = (state.get("query_extracted") or []) + (state.get("seed_extracted") or [])
    if not all_papers:
        state["synthesis"] = {"themes": [], "gaps": [], "limitations_summary": [], "narrative": ""}
        return state

    joined = "\n\n".join(
        f"- {p.get('title')}: summary={p.get('summary')}; "
        f"limitations={p.get('limitations')}; future_work={p.get('future_work')}"
        for p in all_papers
    )
    prompt = (
        f"Given these paper summaries/limitations/future-work notes, identify:\n"
        f"1. Main themes / what has been done\n"
        f"2. Gaps in the literature (data, evaluation, methodology, experimental coverage)\n"
        f"3. A short narrative synthesis (3-5 sentences)\n\n"
        f"Papers:\n{joined}\n\n"
        f'Return JSON: {{"themes": [...], "gaps": [...], "limitations_summary": [...], "narrative": "..."}}'
    )
    state["synthesis"] = llm_provider.complete_json("synthesis", prompt)
    return state


# ---------------------------------------------------------------------------
# Critic: groundedness check on the synthesis -- doubles as the eval judge.
# ---------------------------------------------------------------------------
def critic_node(state: PipelineState) -> PipelineState:
    synthesis = state.get("synthesis") or {}
    all_papers = (state.get("query_extracted") or []) + (state.get("seed_extracted") or [])
    titles = [p.get("title") for p in all_papers]

    prompt = (
        f"Synthesis narrative:\n{synthesis.get('narrative', '')}\n\n"
        f"Claimed themes: {synthesis.get('themes', [])}\n"
        f"Claimed gaps: {synthesis.get('gaps', [])}\n\n"
        f"Source papers actually available: {titles}\n\n"
        f"Judge whether each claim is grounded in the source papers above "
        f"(not fabricated or unsupported).\n\n"
        f'Return JSON: {{"groundedness_score": <0-1 float>, "flagged_claims": ["..."], "notes": "..."}}'
    )
    state["critic"] = llm_provider.complete_json("critic", prompt)
    return state
