import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")  # optional, raises rate limit

S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
S2_RECOMMENDATIONS_URL = "https://api.semanticscholar.org/recommendations/v1/papers"

# --- Model routing --------------------------------------------------------
# The ONE place that maps a pipeline step to a specific Gemini model.
# High-volume steps (planner, extraction, relevance judging) use the cheap
# Flash-Lite tier; low-volume, high-value steps (synthesis, critic) use the
# larger Flash tier. Swapping providers later (Groq/Ollama/Claude) means
# changing this file and llm_provider.py -- pipeline nodes never reference a
# model name directly.
#
# Model IDs age out (the API 404s a removed model). If a run fails with
# "model ... is no longer available", list current ones with:
#   uv run python -c "from google import genai; from backend import config; \
#     [print(m.name) for m in genai.Client(api_key=config.GEMINI_API_KEY).models.list()]"
# 'gemini-2.5-pro' is unavailable / quota-exhausted on this key -- synthesis
# and critic run on gemini-3.5-flash-lite instead for now.
MODEL_ROUTING = {
    "planner": "gemini-3.5-flash-lite",
    "extraction": "gemini-3.5-flash-lite",
    "relevance_judge": "gemini-3.5-flash-lite",
    "synthesis": "gemini-3.5-flash-lite",
    "critic": "gemini-3.5-flash-lite",
}

# --- Breadth presets (niche <-> broad), keyed 1-5 -------------------------
BREADTH_PRESETS = {
    1: {"query_variants": 1, "sim_threshold": 0.80, "hop_depth": 1, "branch_k": 5, "pool_size": 20},
    2: {"query_variants": 2, "sim_threshold": 0.75, "hop_depth": 1, "branch_k": 5, "pool_size": 40},
    3: {"query_variants": 4, "sim_threshold": 0.68, "hop_depth": 2, "branch_k": 7, "pool_size": 60},
    4: {"query_variants": 6, "sim_threshold": 0.60, "hop_depth": 2, "branch_k": 10, "pool_size": 80},
    5: {"query_variants": 8, "sim_threshold": 0.55, "hop_depth": 3, "branch_k": 10, "pool_size": 120},
}

# How many top-ranked papers get full LLM extraction (cost/latency control).
TOP_N_TO_EXTRACT = 10
