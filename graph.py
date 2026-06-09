"""
Exclusion Management Graph — LangGraph StateGraph assembly.

Assembles all agent nodes into a directed graph with conditional routing.

Graph topology:
                     ┌──────────────────────┐
                     │  document_ingestion  │
                     └──────────┬───────────┘
                                │
                     ┌──────────▼───────────┐
                     │ exclusion_extraction │
                     └──────────┬───────────┘
                                │
                     ┌──────────▼───────────┐
                     │     validation       │
                     └──────────┬───────────┘
                                │
                    ┌───────────┴──────────────┐
             (failed)                     (passed)
                    │                          │
         ┌──────────▼──────────┐  ┌────────────▼──────────────┐
         │    notification     │  │   conflict_detection      │
         └─────────────────────┘  └────────────┬──────────────┘
                                               │
                                  ┌────────────▼──────────────┐
                                  │       application         │
                                  └────────────┬──────────────┘
                                               │
                                  ┌────────────▼──────────────┐
                                  │      notification         │
                                  └───────────────────────────┘
"""

from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.nodes import (
    document_ingestion_node,
    exclusion_extraction_node,
    validation_node,
    conflict_detection_node,
    application_node,
    notification_node,
    route_after_validation,
)


def build_exclusion_graph() -> StateGraph:
    """Build and compile the LangGraph exclusion management pipeline."""

    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("document_ingestion", document_ingestion_node)
    graph.add_node("exclusion_extraction", exclusion_extraction_node)
    graph.add_node("validation", validation_node)
    graph.add_node("conflict_detection", conflict_detection_node)
    graph.add_node("application", application_node)
    graph.add_node("notification", notification_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("document_ingestion")

    # ── Linear edges ──────────────────────────────────────────────────────────
    graph.add_edge("document_ingestion", "exclusion_extraction")
    graph.add_edge("exclusion_extraction", "validation")

    # ── Conditional edge after validation ─────────────────────────────────────
    graph.add_conditional_edges(
        "validation",
        route_after_validation,
        {
            "conflict_detection": "conflict_detection",
            "notification": "notification",
        },
    )

    graph.add_edge("conflict_detection", "application")
    graph.add_edge("application", "notification")
    graph.add_edge("notification", END)

    return graph.compile()


# ── Run helper ────────────────────────────────────────────────────────────────

def run_exclusion_pipeline(
    policy_id: str,
    member_id: str,
    s3_key: str,
) -> AgentState:
    """
    Entry point to execute the full exclusion management pipeline.

    Args:
        policy_id:  Insurance policy identifier.
        member_id:  Member/insured identifier.
        s3_key:     S3 key of the raw policy document to process.

    Returns:
        Final AgentState after all nodes have executed.
    """
    app = build_exclusion_graph()

    initial_state: AgentState = {
        "policy_id": policy_id,
        "member_id": member_id,
        "raw_document_key": s3_key,
        "document_content": None,
        "exclusions": [],
        "validation_errors": [],
        "validation_passed": False,
        "conflicts": [],
        "conflict_detected": False,
        "applied_exclusions": [],
        "rejected_exclusions": [],
        "notification_sent": False,
        "notification_payload": None,
        "audit_trail": [],
        "current_status": "pending",
        "error_message": None,
        "retry_count": 0,
    }

    final_state = app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    import json

    result = run_exclusion_pipeline(
        policy_id="POL-2025-001234",
        member_id="MBR-98765",
        s3_key="incoming/sample_policy_doc.txt",
    )

    print("\n=== PIPELINE COMPLETE ===")
    print(f"Status       : {result['current_status']}")
    print(f"Applied      : {len(result['applied_exclusions'])}")
    print(f"Rejected     : {len(result['rejected_exclusions'])}")
    print(f"Conflicts    : {len(result['conflicts'])}")
    print(f"Validation   : {'PASSED' if result['validation_passed'] else 'FAILED'}")
    print(f"Notified     : {result['notification_sent']}")
    print("\nAudit Trail:")
    for entry in result["audit_trail"]:
        print(f"  {entry}")
