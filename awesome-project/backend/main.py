"""
The thin controller layer. Everything here is about HTTP concerns --
validating requests, translating to/from the pipeline's state format, and
mapping errors to status codes. No decision about *what the pipeline does*
belongs in this file; that all lives in backend/pipeline/.
"""
from fastapi import FastAPI, HTTPException

from backend.schemas import ReviewRequest, ReviewResponse, ExtractedPaper
from backend.pipeline.graph import build_graph
from backend.utils import normalize_paper, is_doi, looks_like_doi_attempt

app = FastAPI(title="Literature Review Agent")

# Compiled once at startup, reused across requests -- building the graph is
# cheap, but there's no reason to redo it on every call.
_graph = build_graph()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/review", response_model=ReviewResponse)
def review(req: ReviewRequest):
    # The one conditional that belongs here: is this a well-formed request at
    # all? (NOT "what mode should the pipeline run in" -- that's the
    # planner's job, inside the orchestrator.)
    if not req.query and not req.seed_papers:
        raise HTTPException(400, "Provide at least a query or seed_papers.")

    # seed_papers is a mix of titles / arXiv IDs / DOIs. Only reject entries
    # that were clearly meant as a DOI but are malformed.
    for seed in req.seed_papers or []:
        if looks_like_doi_attempt(seed) and not is_doi(seed):
            raise HTTPException(400, f"DOI not valid: {seed}")

    initial_state = {
        "query": req.query,
        "seed_paper_ids": req.seed_papers,
        "year_min": req.year_min,
        "year_max": req.year_max,
        "breadth": req.breadth,
    }

    try:
        final_state = _graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {e}")

    results = [ExtractedPaper(**normalize_paper(p)) for p in final_state.get("extracted", [])]

    return ReviewResponse(
        results=results,
        synthesis=final_state.get("synthesis"),
        critic=final_state.get("critic"),
        mode=final_state.get("mode", "keyword"),
    )
