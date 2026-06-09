"""
LangGraph Agent Nodes for Exclusion Management Pipeline.

Node execution order:
  document_ingestion_node
       ↓
  exclusion_extraction_node
       ↓
  validation_node
       ↓
  conflict_detection_node   ← conditional branch
       ↓
  application_node
       ↓
  notification_node
"""

import re
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, List

from agents.state import AgentState, ExclusionStatus, ExclusionType, PolicyExclusion
from tools.aws_tools import S3Tool, DynamoDBTool, LambdaTool
from config.settings import Settings

logger = logging.getLogger(__name__)
cfg = Settings()

# ── tool singletons ──────────────────────────────────────────────────────────
s3 = S3Tool(bucket_name=cfg.S3_BUCKET)
dynamo = DynamoDBTool(table_name=cfg.DYNAMO_TABLE)
lam = LambdaTool()


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 — Document Ingestion
# ─────────────────────────────────────────────────────────────────────────────

def document_ingestion_node(state: AgentState) -> AgentState:
    """
    Fetch the raw policy document from S3.
    Populates: document_content, audit_trail, current_status
    """
    logger.info(f"[Ingestion] Fetching: {state['raw_document_key']}")
    try:
        content = s3.read_document(state["raw_document_key"])
        return {
            **state,
            "document_content": content,
            "current_status": ExclusionStatus.PENDING,
            "audit_trail": [
                f"{_ts()} | INGESTION | Document loaded from {state['raw_document_key']}"
            ],
        }
    except Exception as e:
        return {
            **state,
            "error_message": f"Ingestion failed: {str(e)}",
            "audit_trail": [f"{_ts()} | INGESTION | ERROR: {e}"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 — Exclusion Extraction
# ─────────────────────────────────────────────────────────────────────────────

# Simple rule-based extractor (replace with Bedrock/LLM call in production)
_EXCLUSION_PATTERNS = {
    ExclusionType.GEOGRAPHIC: r"(?i)(exclud(?:es?|ing)|not covered|excluded in)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
    ExclusionType.OCCUPATIONAL: r"(?i)(occupation[a-z]*\s+exclusion|high[- ]risk\s+occup[a-z]*)",
    ExclusionType.MEDICAL: r"(?i)(pre[- ]?existing\s+condition|medical\s+exclusion|not\s+cover(?:ed|ing)\s+[a-z ]+disease)",
    ExclusionType.BEHAVIORAL: r"(?i)(self[- ]?inflict|criminal\s+act|substance\s+abuse|exclusion\s+for\s+[a-z ]+behavior)",
    ExclusionType.FINANCIAL: r"(?i)(financial\s+exclusion|fraud|misrepresent[a-z]*)",
}

_DATE_PATTERN = r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"
_REASON_CODES = {
    ExclusionType.GEOGRAPHIC: "GEO-001",
    ExclusionType.OCCUPATIONAL: "OCC-002",
    ExclusionType.MEDICAL: "MED-003",
    ExclusionType.BEHAVIORAL: "BEH-004",
    ExclusionType.FINANCIAL: "FIN-005",
}


def _extract_exclusions_from_text(
    text: str, policy_id: str, member_id: str
) -> List[PolicyExclusion]:
    exclusions: List[PolicyExclusion] = []
    date_match = re.search(_DATE_PATTERN, text)
    effective_date = date_match.group(1) if date_match else datetime.utcnow().strftime("%Y-%m-%d")

    for ex_type, pattern in _EXCLUSION_PATTERNS.items():
        for match in re.finditer(pattern, text):
            context_start = max(0, match.start() - 40)
            context_end = min(len(text), match.end() + 80)
            snippet = text[context_start:context_end].strip()

            exclusions.append(
                PolicyExclusion(
                    exclusion_id=str(uuid.uuid4()),
                    policy_id=policy_id,
                    member_id=member_id,
                    exclusion_type=ex_type,
                    description=match.group(0),
                    effective_date=effective_date,
                    reason_code=_REASON_CODES[ex_type],
                    raw_text=snippet,
                )
            )
    return exclusions


def exclusion_extraction_node(state: AgentState) -> AgentState:
    """
    Parse policy document and extract all exclusion clauses.
    Populates: exclusions, audit_trail
    """
    logger.info(f"[Extraction] Running pattern extractor for policy {state['policy_id']}")
    if not state.get("document_content"):
        return {
            **state,
            "error_message": "No document content to extract from.",
            "audit_trail": [f"{_ts()} | EXTRACTION | Skipped — no content"],
        }

    exclusions = _extract_exclusions_from_text(
        state["document_content"],
        state["policy_id"],
        state["member_id"],
    )
    logger.info(f"[Extraction] Found {len(exclusions)} exclusions")

    return {
        **state,
        "exclusions": exclusions,
        "audit_trail": [
            f"{_ts()} | EXTRACTION | {len(exclusions)} exclusions identified"
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3 — Validation
# ─────────────────────────────────────────────────────────────────────────────

_VALID_REASON_CODES = {"GEO-001", "OCC-002", "MED-003", "BEH-004", "FIN-005"}


def validation_node(state: AgentState) -> AgentState:
    """
    Validate each exclusion for completeness and business rule compliance.
    Populates: validation_errors, validation_passed, current_status, audit_trail
    """
    errors: List[str] = []

    for ex in state.get("exclusions", []):
        if not ex.get("effective_date"):
            errors.append(f"{ex['exclusion_id']}: missing effective_date")
        if ex.get("reason_code") not in _VALID_REASON_CODES:
            errors.append(f"{ex['exclusion_id']}: unknown reason_code '{ex.get('reason_code')}'")
        if not ex.get("description"):
            errors.append(f"{ex['exclusion_id']}: empty description")

    passed = len(errors) == 0
    logger.info(f"[Validation] Passed={passed}, Errors={errors}")

    return {
        **state,
        "validation_errors": errors,
        "validation_passed": passed,
        "current_status": ExclusionStatus.VALIDATED if passed else ExclusionStatus.FLAGGED,
        "audit_trail": [
            f"{_ts()} | VALIDATION | {'PASSED' if passed else 'FAILED'} — {len(errors)} error(s)"
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4 — Conflict Detection
# ─────────────────────────────────────────────────────────────────────────────

def conflict_detection_node(state: AgentState) -> AgentState:
    """
    Check each extracted exclusion against DynamoDB for conflicts with
    already-applied exclusions of the same type and date.
    Populates: conflicts, conflict_detected, audit_trail
    """
    conflicts = []

    for ex in state.get("exclusions", []):
        existing = dynamo.check_conflict(
            ex["policy_id"], ex["exclusion_type"], ex["effective_date"]
        )
        if existing:
            conflicts.append({
                "incoming": ex["exclusion_id"],
                "conflicting": existing.get("exclusion_id"),
                "type": ex["exclusion_type"],
                "effective_date": ex["effective_date"],
            })
            logger.warning(
                f"[Conflict] {ex['exclusion_id']} conflicts with {existing.get('exclusion_id')}"
            )

    detected = len(conflicts) > 0

    return {
        **state,
        "conflicts": conflicts,
        "conflict_detected": detected,
        "audit_trail": [
            f"{_ts()} | CONFLICT | {'Detected ' + str(len(conflicts)) + ' conflict(s)' if detected else 'None detected'}"
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5 — Application
# ─────────────────────────────────────────────────────────────────────────────

def application_node(state: AgentState) -> AgentState:
    """
    Persist validated exclusions to DynamoDB.
    Conflicting or invalid exclusions are rejected; the rest are applied.
    Populates: applied_exclusions, rejected_exclusions, audit_trail, current_status
    """
    applied = []
    rejected = []
    conflict_ids = {c["incoming"] for c in state.get("conflicts", [])}

    for ex in state.get("exclusions", []):
        if ex["exclusion_id"] in conflict_ids:
            rejected.append(ex["exclusion_id"])
            dynamo.update_exclusion_status(ex["policy_id"], ex["exclusion_id"], "rejected")
            logger.info(f"[Application] Rejected (conflict): {ex['exclusion_id']}")
            continue

        success = dynamo.put_exclusion({**ex, "status": "applied"})
        if success:
            applied.append(ex["exclusion_id"])
        else:
            rejected.append(ex["exclusion_id"])

    # Persist full result bundle to S3
    result_key = f"processed/{state['policy_id']}/{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.json"
    s3.write_result(
        {
            "policy_id": state["policy_id"],
            "member_id": state["member_id"],
            "applied": applied,
            "rejected": rejected,
            "conflicts": state.get("conflicts", []),
        },
        result_key,
    )

    return {
        **state,
        "applied_exclusions": applied,
        "rejected_exclusions": rejected,
        "current_status": ExclusionStatus.APPLIED,
        "audit_trail": [
            f"{_ts()} | APPLICATION | Applied={len(applied)}, Rejected={len(rejected)}"
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 6 — Notification
# ─────────────────────────────────────────────────────────────────────────────

def notification_node(state: AgentState) -> AgentState:
    """
    Fire notification Lambda with the final exclusion summary.
    Populates: notification_sent, notification_payload, current_status, audit_trail
    """
    payload = {
        "policy_id": state["policy_id"],
        "member_id": state["member_id"],
        "applied_count": len(state.get("applied_exclusions", [])),
        "rejected_count": len(state.get("rejected_exclusions", [])),
        "conflict_count": len(state.get("conflicts", [])),
        "timestamp": _ts(),
    }

    result = lam.invoke_notification(cfg.NOTIFICATION_LAMBDA, payload)
    sent = result.get("status") == "triggered"

    # Log to audit Lambda
    lam.invoke_audit_logger(
        cfg.AUDIT_LAMBDA,
        {"audit_trail": state.get("audit_trail", []), **payload},
    )

    return {
        **state,
        "notification_sent": sent,
        "notification_payload": payload,
        "current_status": ExclusionStatus.NOTIFIED,
        "audit_trail": [
            f"{_ts()} | NOTIFICATION | {'Sent' if sent else 'Failed'}"
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Edge Condition — Route after Validation
# ─────────────────────────────────────────────────────────────────────────────

def route_after_validation(state: AgentState) -> str:
    """
    Conditional edge: if validation fails, route to notification (skip apply).
    Otherwise proceed to conflict detection.
    """
    if not state.get("validation_passed", True):
        logger.warning("[Router] Validation failed — routing to notification (skip)")
        return "notification"
    return "conflict_detection"


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
