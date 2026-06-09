"""
State schema for the Exclusion Management Agent.
Defines the shared state passed across all LangGraph nodes.
"""

from typing import TypedDict, Annotated, List, Optional, Dict, Any
from enum import Enum
import operator


class ExclusionStatus(str, Enum):
    PENDING = "pending"
    VALIDATED = "validated"
    FLAGGED = "flagged"
    APPLIED = "applied"
    REJECTED = "rejected"
    NOTIFIED = "notified"


class ExclusionType(str, Enum):
    GEOGRAPHIC = "geographic"
    OCCUPATIONAL = "occupational"
    MEDICAL = "medical"
    BEHAVIORAL = "behavioral"
    FINANCIAL = "financial"


class PolicyExclusion(TypedDict):
    exclusion_id: str
    policy_id: str
    member_id: str
    exclusion_type: ExclusionType
    description: str
    effective_date: str
    reason_code: str
    raw_text: str


class AgentState(TypedDict):
    """
    Central state object flowing through the LangGraph pipeline.
    All agents read from and write to this shared state.
    """
    # Input
    policy_id: str
    member_id: str
    raw_document_key: str          # S3 key for the uploaded policy document
    document_content: Optional[str]

    # Extracted data
    exclusions: Annotated[List[PolicyExclusion], operator.add]

    # Validation results
    validation_errors: Annotated[List[str], operator.add]
    validation_passed: bool

    # Conflict detection
    conflicts: Annotated[List[Dict[str, Any]], operator.add]
    conflict_detected: bool

    # Application
    applied_exclusions: Annotated[List[str], operator.add]   # exclusion_ids
    rejected_exclusions: Annotated[List[str], operator.add]

    # Notifications
    notification_sent: bool
    notification_payload: Optional[Dict[str, Any]]

    # Audit trail
    audit_trail: Annotated[List[str], operator.add]
    current_status: ExclusionStatus

    # Control
    error_message: Optional[str]
    retry_count: int
