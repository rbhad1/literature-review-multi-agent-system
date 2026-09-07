from typing import TypedDict, List, Optional


class PipelineState(TypedDict, total=False):
    # --- inputs ---
    query: Optional[str]
    seed_paper_ids: Optional[List[str]]
    year_min: Optional[int]
    year_max: Optional[int]
    breadth: int

    # --- derived ---
    mode: str  # "keyword" | "seed" | "hybrid" -- inferred by planner_node, never asked of the user

    # --- planner output ---
    query_variants: List[str]

    # --- retrieval output ---
    query_candidates: List[dict]
    seed_papers_resolved: List[dict]
    seed_candidates: List[dict]

    # --- ranking output (per-track, pre-merge) ---
    query_ranked: List[dict]
    seed_ranked: List[dict]

    # --- ranking-critic output ---
    # ranking_critic_node merges query_ranked + seed_ranked into a single
    # `ranked` list (dedup by paperId), then runs an LLM-as-judge over its
    # top-N BEFORE extraction spend. The judge may reorder / drop papers
    # exactly once (no loop back -- see graph.py). Everything downstream of
    # here works on the one `ranked` list, hybrid mode included.
    ranked: List[dict]
    ranking_critic: dict  # the judge's JSON: {verdict, flagged, notes, reranked_ids, drop_ids}
    #   verdict is categorical ("sound"|"minor_issues"|"poor") -- the judge
    #   decides ORDERING (reranked_ids), never per-paper numeric scores.

    # --- extraction output ---
    extracted: List[dict]

    # --- synthesis / critic output ---
    synthesis: dict
    critic: dict
