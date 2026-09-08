"""
End-to-end smoke test for the literature-review pipeline.

Prompts for a keyword query and/or a seed-paper DOI on the console, runs the
compiled LangGraph once (default breadth=1 -> minimal LLM calls), and prints a
readable per-stage summary, flagging anything empty or malformed. This is a
diagnostic, not an assertion suite -- it's meant to make a real run debuggable
instead of one giant dict.

    uv run python smoke_test.py

At least one of query / seed DOI must be given. Multiple DOIs: comma-separate.
"""
import json
import logging
import sys
import textwrap
import time
from pathlib import Path

from backend.pipeline.graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("smoke")
logging.getLogger("google_genai").setLevel(logging.ERROR)


def _prompt_inputs() -> dict:
    query = input("Query keywords (blank to skip): ").strip()
    dois_raw = input("Seed paper DOI(s), comma-separated (blank to skip): ").strip()
    breadth_raw = input("Breadth 1-5 [1]: ").strip()

    if not query and not dois_raw:
        raise SystemExit("Need at least a query or a seed DOI.")

    state: dict = {"breadth": int(breadth_raw) if breadth_raw else 1}
    if query:
        state["query"] = query
    if dois_raw:
        state["seed_paper_ids"] = [d.strip() for d in dois_raw.split(",") if d.strip()]
    return state


# Collapse any value to a single-line string capped at `n` chars, so long
# abstracts / lists / dicts don't blow up the console summary. Non-strings are
# JSON-dumped first (default=str handles anything not natively serializable).
def _short(x, n=200):
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    return textwrap.shorten(s, width=n, placeholder=" ...")


# Emit one aligned "label: value" row of the per-stage summary at INFO level.
def _line(label, value):
    log.info("  %-22s %s", label, value)


# Emit a WARNING row only when `cond` is true -- used to surface a suspected
# problem (empty output, missing field, malformed JSON) without aborting the run.
def _flag(cond, msg):
    if cond:
        log.warning("  !! %s", msg)


# Recursively drop SPECTER embedding vectors (~768 floats per paper, repeated
# across query_candidates / ranked / extracted / ...) so results.json stays
# small and human-readable. The `embedding.model` tag is kept for context.
def _strip_vectors(obj):
    if isinstance(obj, dict):
        return {k: _strip_vectors(v) for k, v in obj.items()
                if k not in ("vector", "_vector")}
    if isinstance(obj, list):
        return [_strip_vectors(v) for v in obj]
    return obj


def _run_graph(graph, initial: dict) -> dict:
    """Invoke via stream() so each node logs to the console as it finishes,
    with elapsed time -- makes a slow run visible instead of one long hang."""
    state: dict = dict(initial)
    t0 = time.perf_counter()
    last = t0
    for update in graph.stream(initial, stream_mode="updates"):
        for node, delta in update.items():
            now = time.perf_counter()
            keys = sorted(delta.keys()) if isinstance(delta, dict) else type(delta).__name__
            log.info("node %-16s done  (+%.1fs, total %.1fs)  set: %s",
                     node, now - last, now - t0, keys)
            last = now
            if isinstance(delta, dict):
                state.update(delta)
    log.info("graph finished in %.1fs", time.perf_counter() - t0)
    return state


def main():
    initial = _prompt_inputs()
    log.info("=" * 70)
    _line("query", initial.get("query"))
    _line("seed_paper_ids", initial.get("seed_paper_ids"))
    _line("breadth", initial.get("breadth"))
    log.info("=" * 70)

    log.info("building graph...")
    graph = build_graph()
    log.info("invoking pipeline (streaming node updates)...")
    state = _run_graph(graph, initial)

    log.info("[planner]")
    _line("mode", state.get("mode"))
    _line("query_variants", state.get("query_variants"))
    _flag(initial.get("query") and not state.get("query_variants"), "no query variants produced")

    log.info("[retrieval]")
    qc = state.get("query_candidates") or []
    sc = state.get("seed_candidates") or []
    resolved = state.get("seed_papers_resolved") or []
    _line("query_candidates", len(qc))
    _line("seed_candidates", len(sc))
    if initial.get("seed_paper_ids"):
        _line("seeds_resolved", f"{len(resolved)}/{len(initial['seed_paper_ids'])}")
        _flag(len(resolved) < len(initial["seed_paper_ids"]), "a seed DOI failed to resolve on S2")
    _flag(not qc and not sc, "retrieval returned zero candidates (S2 down / rate-limited / bad query?)")
    if qc:
        p = qc[0]
        _line("sample paper", _short(f"{p.get('title')} | id={p.get('paperId')} | year={p.get('year')} | abstract? {bool(p.get('abstract'))}"))
        _flag(not p.get("paperId"), "candidate missing paperId -- dedup/merge will misbehave")

    log.info("[ranking]")
    _line("query_ranked", len(state.get("query_ranked") or []))
    _line("seed_ranked", len(state.get("seed_ranked") or []))

    log.info("[ranking_critic]")
    rc = state.get("ranking_critic") or {}
    _line("verdict", rc.get("verdict"))
    _line("flagged", _short(rc.get("flagged")))
    _line("drop_ids", len(rc.get("drop_ids") or []))
    _line("reranked_ids", len(rc.get("reranked_ids") or []))
    _line("ranked (carried)", len(state.get("ranked") or []))
    _flag(rc and not rc.get("reranked_ids"), "critic ran but returned no ordering (JSON shape mismatch?)")

    log.info("[extraction]")
    ex = state.get("extracted") or []
    _line("extracted", len(ex))
    if ex:
        e = ex[0]
        _line("has summary", bool(e.get("summary")))
        _line("has limitations", bool(e.get("limitations")))
        _flag(not e.get("summary"), "extraction produced no summary -- prompt/JSON parse issue")

    log.info("[synthesis]")
    syn = state.get("synthesis") or {}
    _line("themes", len(syn.get("themes") or []))
    _line("gaps", len(syn.get("gaps") or []))
    _line("narrative", _short(syn.get("narrative"), 300))
    _flag(not syn.get("narrative"), "empty synthesis narrative")

    log.info("[critic]")
    cr = state.get("critic") or {}
    _line("groundedness", cr.get("groundedness_score"))
    _line("flagged_claims", _short(cr.get("flagged_claims")))

    log.info("=" * 70)
    log.info("done. final state keys: %s", sorted(state.keys()))

    out_path = Path(__file__).parent / "backend" / "results.json"
    out_path.write_text(json.dumps(_strip_vectors(state), indent=2, default=str), encoding="utf-8")
    log.info("final state written to %s (embedding vectors stripped)", out_path)


if __name__ == "__main__":
    main()
