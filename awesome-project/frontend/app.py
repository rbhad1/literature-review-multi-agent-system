"""
Minimal Streamlit UI for the literature-review pipeline.

    uv run streamlit run frontend/app.py

Enter keywords and/or seed DOIs, get back the synthesis paragraph and the
ranked list of papers with external links. Runs the LangGraph pipeline
in-process -- no separate API server needed.
"""
import logging
import sys
import warnings
from pathlib import Path

# Keep the UI clean: no library warnings in the console or the app.
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
for _noisy in ("google_genai", "httpx", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# Let `import backend...` work no matter where streamlit is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from backend.pipeline.graph import build_graph
from backend.utils import is_doi, DOI_FORMAT_HINT

# Pipeline node name -> plain progress label shown to the user.
STEP_LABELS = {
    "planner": "Planning search",
    "retrieval": "Collecting sources",
    "ranking": "Ranking sources",
    "ranking_critic": "Reviewing rankings",
    "extraction": "Reading papers",
    "synthesis": "Synthesizing results",
    "critic": "Checking synthesis",
}


@st.cache_resource
def _graph():
    """Compile the graph once per Streamlit session."""
    return build_graph()


def _paper_url(p: dict):
    """Best available external link for a paper, or None."""
    ext = p.get("externalIds") or {}
    if ext.get("DOI"):
        return f"https://doi.org/{ext['DOI']}"
    if ext.get("ArXiv"):
        return f"https://arxiv.org/abs/{ext['ArXiv']}"
    if p.get("paperId"):
        return f"https://www.semanticscholar.org/paper/{p['paperId']}"
    return None


st.set_page_config(page_title="Literature Review", layout="centered")
st.title("Literature Review")
st.caption("Enter keywords and/or seed DOIs. The agent searches, ranks, and synthesizes.")

with st.form("query"):
    keywords = st.text_input(
        "Keywords",
        placeholder="e.g. sample-efficient model-based reinforcement learning",
    )
    dois = st.text_input(
        "Seed DOIs (comma-separated, optional)",
        placeholder="10.48550/arXiv.2301.04104",
    )
    breadth = st.slider(
        "Breadth", 1, 5, 2,
        help="Higher = wider search, more papers, slower and more API calls.",
    )
    submitted = st.form_submit_button("Generate review")

if submitted:
    kw = keywords.strip()
    doi_list = [d.strip() for d in dois.split(",") if d.strip()]
    if not kw and not doi_list:
        st.warning("Enter at least a keyword phrase or one DOI.")
        st.stop()

    invalid = [d for d in doi_list if not is_doi(d)]
    if invalid:
        st.error("DOI is not valid: " + ", ".join(invalid) + "\n\n" + DOI_FORMAT_HINT)
        st.stop()

    payload = {"breadth": breadth}
    if kw:
        payload["query"] = kw
    if doi_list:
        payload["seed_paper_ids"] = doi_list

    state: dict = {}
    with st.status("Working...", expanded=True) as status:
        try:
            for update in _graph().stream(payload, stream_mode="updates"):
                for node, delta in update.items():
                    if node in STEP_LABELS:
                        st.write(f"✓ {STEP_LABELS[node]}")
                    if isinstance(delta, dict):
                        state.update(delta)
        except Exception as e:  # noqa: BLE001 - surface any pipeline failure to the user
            status.update(label="Failed", state="error")
            st.error(f"Pipeline failed: {e}")
            st.stop()
        status.update(label="Done", state="complete", expanded=False)

    synthesis = state.get("synthesis") or {}
    papers = state.get("extracted") or state.get("ranked") or []

    st.subheader("Synthesis")
    if synthesis.get("narrative"):
        st.write(synthesis["narrative"])
    else:
        st.info("No synthesis was produced.")

    themes, gaps = synthesis.get("themes") or [], synthesis.get("gaps") or []
    if themes or gaps:
        with st.expander("Themes & gaps"):
            if themes:
                st.markdown("**Themes**")
                st.markdown("\n".join(f"- {t}" for t in themes))
            if gaps:
                st.markdown("**Gaps**")
                st.markdown("\n".join(f"- {g}" for g in gaps))

    st.subheader(f"Papers ({len(papers)})")
    if not papers:
        st.info("No papers returned.")
    for i, p in enumerate(papers, 1):
        title = p.get("title") or "Untitled"
        url = _paper_url(p)
        heading = f"{i}. [{title}]({url})" if url else f"{i}. {title}"
        meta = " · ".join(
            str(x) for x in (
                p.get("year"),
                f"{p['citationCount']} citations" if p.get("citationCount") is not None else None,
                p.get("provenance"),
            ) if x
        )
        st.markdown(f"**{heading}**" + (f"  \n{meta}" if meta else ""))
        if p.get("summary"):
            st.markdown(p["summary"])
        st.divider()
