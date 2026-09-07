"""
The LangGraph node functions -- this is the "fat orchestrator" side of the
project. Each function takes the shared PipelineState, does one job, and
returns the (mutated) state for the next node. See graph.py for how these
are wired into a sequence, and pipeline/state.py for the shared schema.
"""
import math
import random
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
    resolved = s2.resolve_papers(seed_ids)  # one /paper/batch call for ids+DOIs
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
# TODO: play around with decay factor 
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
        base_relevance = paper.get("score", 0.0) or 0.0 # check recommendation_strats.py
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
# Ranking critic: picks ONE ranked list to carry forward, then runs an
# LLM-as-judge over its top-N BEFORE extraction so a bad ranking doesn't waste
# extraction calls. When both tracks are populated (hybrid mode) the SEED
# track wins -- merging two differently-scaled score lists was more trouble
# than it was worth; revisit later. The judge can hand back a corrected order
# and/or a drop list, applied exactly once -- there is no edge back to ranking
# (see graph.py). In pure seed-mode (no query text) there's nothing to judge
# relevance against, so the judge is skipped and the list passes through as-is.
# ---------------------------------------------------------------------------
def _critique_ranking(question: str, ranked: list) -> dict:
    # Present the papers to the judge in RANDOM order, not pipeline-rank order.
    # LLMs systematically over-rate whatever comes first in a list, so showing
    # them our current ranking would just get that ranking echoed back. The
    # pipeline's own score is left out of the listing for the same reason.
    # Output is keyed by paperId, so _apply_ranking_correction maps it back
    # regardless of presentation order -- no un-shuffle step needed.
    top = ranked[:config.TOP_N_TO_EXTRACT]
    shuffled = random.sample(top, k=len(top))
    listing = "\n".join(
        f"- [{p.get('paperId')}] {p.get('title')} "
        f"(year={p.get('year')}, citations={p.get('citationCount')}, "
        f"provenance={p.get('provenance')})\n"
        f"  abstract: {(p.get('abstract') or 'N/A')[:1200]}"
        for p in shuffled
    )
    prompt = (
        f'Research question: "{question}"\n\n'
        f"Below are candidate papers in ARBITRARY order. Act as an impartial "
        f"judge and rank them yourself from scratch by relevance to the "
        f"research question.\n\n"
        f"{listing}\n\n"
        f"Assess: (a) whether each paper is genuinely on-topic for the research "
        f"question, (b) what the correct relevance ordering is, (c) whether the "
        f"set covers the question's sub-aspects rather than piling onto one "
        f"narrow cluster.\n\n"
        f"Return the paperIds ordered most- to least-relevant, plus any that "
        f"should be dropped as off-topic. Do NOT assign numeric relevance "
        f"scores -- only the ordering matters.\n\n"
        f'Return JSON: {{"verdict": "sound" | "minor_issues" | "poor", '
        f'"flagged": ["..."], "notes": "...", '
        f'"reranked_ids": ["<paperId>", "..."], '
        f'"drop_ids": ["<paperId>", "..."]}}'
    )
    return llm_provider.complete_json("critic", prompt)


def _apply_ranking_correction(ranked: list, critique: dict) -> list:
    """Reorder `ranked` to the critic's `reranked_ids`, remove `drop_ids`.
    Any paper the critic didn't mention keeps its relative order and is
    appended after the explicitly ordered ones -- so a partial/garbled LLM
    response degrades to 'mostly unchanged', never to data loss."""
    by_id = {p.get("paperId"): p for p in ranked}
    drop = set(critique.get("drop_ids") or [])
    order = [pid for pid in (critique.get("reranked_ids") or [])
             if pid in by_id and pid not in drop]

    corrected = [by_id[pid] for pid in order]
    seen = set(order)
    for paper in ranked:
        pid = paper.get("paperId")
        if pid not in seen and pid not in drop:
            corrected.append(paper)
    return corrected


def ranking_critic_node(state: PipelineState) -> PipelineState:
    question = state.get("query") or ""

    # No merge: in hybrid mode both tracks are populated -- prefer seed.
    ranked = list(state.get("seed_ranked") or state.get("query_ranked") or [])
    if not ranked:
        state["ranked"] = []
        state["ranking_critic"] = {}
        return state

    if question:
        crit = _critique_ranking(question, ranked)
        state["ranking_critic"] = crit
        if crit.get("reranked_ids") or crit.get("drop_ids"):
            ranked = _apply_ranking_correction(ranked, crit)
    else:
        state["ranking_critic"] = {}

    state["ranked"] = ranked
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
    ranked = state.get("ranked") or []
    state["extracted"] = [_extract_one(p) for p in ranked[:n]]
    return state


# ---------------------------------------------------------------------------
# Synthesis: cross-paper themes, gaps, limitations narrative.
# ---------------------------------------------------------------------------
def synthesis_node(state: PipelineState) -> PipelineState:
    all_papers = state.get("extracted") or []
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
    all_papers = state.get("extracted") or []
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
