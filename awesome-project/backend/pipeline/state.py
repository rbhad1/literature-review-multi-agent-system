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

    # --- ranking output ---
    query_ranked: List[dict]
    seed_ranked: List[dict]

    # --- extraction output ---
    query_extracted: List[dict]
    seed_extracted: List[dict]

    # --- synthesis / critic output ---
    synthesis: dict
    critic: dict
