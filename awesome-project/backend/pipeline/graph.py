from langgraph.graph import StateGraph, END
from backend.pipeline.state import PipelineState
from backend.pipeline.nodes import (
    planner_node,
    retrieval_node,
    ranking_node,
    ranking_critic_node,
    extraction_node,
    synthesis_node,
    critic_node,
)


def build_graph():
    """Wires the node functions into the sequence described in the project
    plan: planner -> retrieval -> ranking -> ranking_critic -> extraction ->
    synthesis -> critic. Two distinct critics: `ranking_critic` (LLM-as-judge
    over the ranked list, may re-rank/drop once, no loop back) and `critic`
    (groundedness of the final synthesis). Compiled once at app startup (see
    main.py) and reused across requests."""
    graph = StateGraph(PipelineState)

    graph.add_node("planner", planner_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("ranking", ranking_node)
    graph.add_node("ranking_critic", ranking_critic_node)
    graph.add_node("extraction", extraction_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "ranking")
    graph.add_edge("ranking", "ranking_critic")
    graph.add_edge("ranking_critic", "extraction")
    graph.add_edge("extraction", "synthesis")
    graph.add_edge("synthesis", "critic")
    graph.add_edge("critic", END)

    return graph.compile()
