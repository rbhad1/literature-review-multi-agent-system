from langgraph.graph import StateGraph, END
from backend.pipeline.state import PipelineState
from backend.pipeline.nodes import (
    planner_node,
    retrieval_node,
    ranking_node,
    extraction_node,
    synthesis_node,
    critic_node,
)


def build_graph():
    """Wires the node functions into the sequence described in the project
    plan: planner -> retrieval -> ranking -> extraction -> synthesis -> critic.
    Compiled once at app startup (see main.py) and reused across requests."""
    graph = StateGraph(PipelineState)

    graph.add_node("planner", planner_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("ranking", ranking_node)
    graph.add_node("extraction", extraction_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("critic", critic_node) ## want to also critic recommendations and rankings --> should i have two critics or two edges?

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "ranking")
    graph.add_edge("ranking", "extraction")
    graph.add_edge("extraction", "synthesis")
    graph.add_edge("synthesis", "critic")
    graph.add_edge("critic", END)

    return graph.compile()
