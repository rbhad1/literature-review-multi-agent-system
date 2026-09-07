from typing import List, Optional
from pydantic import BaseModel, Field


class ReviewRequest(BaseModel):
    query: Optional[str] = None
    seed_papers: Optional[List[str]] = None  # titles, arXiv IDs, or DOIs -- any mix
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    breadth: int = Field(default=3, ge=1, le=5)


class Paper(BaseModel):
    paper_id: str = ""
    title: str = ""
    authors: List[str] = []
    abstract: Optional[str] = None
    year: Optional[int] = None
    citation_count: int = 0
    url: Optional[str] = None
    provenance: str = "query"  # query | seed_specter | seed_recommender | seed_agentic_hopN
    score: float = 0.0
    # Reserved for a future thumbs up/down feedback loop -- not used yet.
    # Kept here now so adding that feature later doesn't require a schema migration.
    user_feedback: Optional[str] = None


class ExtractedPaper(Paper):
    summary: Optional[str] = None
    limitations: Optional[str] = None
    future_work: Optional[str] = None


class SynthesisResult(BaseModel):
    themes: List[str] = []
    gaps: List[str] = []
    limitations_summary: List[str] = []
    narrative: str = ""


class CriticResult(BaseModel):
    groundedness_score: float = 0.0
    flagged_claims: List[str] = []
    notes: str = ""


class ReviewResponse(BaseModel):
    # One merged, ranked list regardless of mode -- query and seed candidates
    # are unified in ranking_critic_node; `provenance` on each paper still says
    # where it came from.
    results: List[ExtractedPaper] = []
    synthesis: Optional[SynthesisResult] = None
    critic: Optional[CriticResult] = None
    mode: str = "keyword"  # keyword | seed | hybrid
