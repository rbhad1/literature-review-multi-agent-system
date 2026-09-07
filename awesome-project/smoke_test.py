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
import textwrap

from backend.pipeline.graph import build_graph


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


def _short(x, n=200):
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    return textwrap.shorten(s, width=n, placeholder=" ...")


def _line(label, value):
    print(f"  {label:<22} {value}")


def _flag(cond, msg):
    if cond:
        print(f"  !! {msg}")


def main():
    initial = _prompt_inputs()
    print("\n" + "=" * 70)
    _line("query", initial.get("query"))
    _line("seed_paper_ids", initial.get("seed_paper_ids"))
    _line("breadth", initial.get("breadth"))
    print("=" * 70)

    graph = build_graph()
    state = graph.invoke(initial)

    print("\n[planner]")
    _line("mode", state.get("mode"))
    _line("query_variants", state.get("query_variants"))
    _flag(initial.get("query") and not state.get("query_variants"), "no query variants produced")

    print("\n[retrieval]")
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

    print("\n[ranking]")
    _line("query_ranked", len(state.get("query_ranked") or []))
    _line("seed_ranked", len(state.get("seed_ranked") or []))

    print("\n[ranking_critic]")
    rc = state.get("ranking_critic") or {}
    _line("verdict", rc.get("verdict"))
    _line("flagged", _short(rc.get("flagged")))
    _line("drop_ids", len(rc.get("drop_ids") or []))
    _line("reranked_ids", len(rc.get("reranked_ids") or []))
    _line("ranked (carried)", len(state.get("ranked") or []))
    _flag(rc and not rc.get("reranked_ids"), "critic ran but returned no ordering (JSON shape mismatch?)")

    print("\n[extraction]")
    ex = state.get("extracted") or []
    _line("extracted", len(ex))
    if ex:
        e = ex[0]
        _line("has summary", bool(e.get("summary")))
        _line("has limitations", bool(e.get("limitations")))
        _flag(not e.get("summary"), "extraction produced no summary -- prompt/JSON parse issue")

    print("\n[synthesis]")
    syn = state.get("synthesis") or {}
    _line("themes", len(syn.get("themes") or []))
    _line("gaps", len(syn.get("gaps") or []))
    _line("narrative", _short(syn.get("narrative"), 300))
    _flag(not syn.get("narrative"), "empty synthesis narrative")

    print("\n[critic]")
    cr = state.get("critic") or {}
    _line("groundedness", cr.get("groundedness_score"))
    _line("flagged_claims", _short(cr.get("flagged_claims")))

    print("\n" + "=" * 70)
    print("done. final state keys:", sorted(state.keys()))


if __name__ == "__main__":
    main()
