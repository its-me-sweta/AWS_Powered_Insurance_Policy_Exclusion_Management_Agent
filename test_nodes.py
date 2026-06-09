"""
Unit tests for Exclusion Management Agent nodes.
Uses moto for AWS mocking and pytest for test execution.

Run: pytest tests/ -v
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from agents.state import AgentState, ExclusionStatus, ExclusionType
from agents.nodes import (
    exclusion_extraction_node,
    validation_node,
    conflict_detection_node,
    route_after_validation,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_state() -> AgentState:
    return {
        "policy_id": "POL-TEST-001",
        "member_id": "MBR-001",
        "raw_document_key": "incoming/POL-TEST-001/MBR-001/doc.txt",
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


SAMPLE_DOCUMENT = """
INSURANCE POLICY EXCLUSIONS

This policy excludes coverage for self-inflicted injuries as defined under
behavioral exclusion clauses effective 2025-01-01.

Medical exclusion: pre-existing conditions diagnosed before policy start date
are not covered under this agreement.

Occupational exclusion applies to high-risk occupations including mining and
explosives handling. Geographic exclusion: coverage is excluded in active
conflict zones. Financial exclusion due to misrepresentation of income.

Policy effective date: 2025-01-01
"""


# ── Extraction Tests ──────────────────────────────────────────────────────────

class TestExclusionExtractionNode:
    def test_extracts_exclusions_from_document(self, base_state):
        state = {**base_state, "document_content": SAMPLE_DOCUMENT}
        result = exclusion_extraction_node(state)
        assert len(result["exclusions"]) > 0

    def test_detects_medical_exclusion(self, base_state):
        state = {**base_state, "document_content": SAMPLE_DOCUMENT}
        result = exclusion_extraction_node(state)
        types = [ex["exclusion_type"] for ex in result["exclusions"]]
        assert ExclusionType.MEDICAL in types

    def test_detects_behavioral_exclusion(self, base_state):
        state = {**base_state, "document_content": SAMPLE_DOCUMENT}
        result = exclusion_extraction_node(state)
        types = [ex["exclusion_type"] for ex in result["exclusions"]]
        assert ExclusionType.BEHAVIORAL in types

    def test_returns_empty_for_no_content(self, base_state):
        result = exclusion_extraction_node(base_state)
        assert result.get("error_message") is not None

    def test_each_exclusion_has_required_fields(self, base_state):
        state = {**base_state, "document_content": SAMPLE_DOCUMENT}
        result = exclusion_extraction_node(state)
        for ex in result["exclusions"]:
            assert ex["exclusion_id"]
            assert ex["policy_id"] == "POL-TEST-001"
            assert ex["member_id"] == "MBR-001"
            assert ex["reason_code"]


# ── Validation Tests ──────────────────────────────────────────────────────────

class TestValidationNode:
    def _make_exclusion(self, overrides=None):
        ex = {
            "exclusion_id": "EX-001",
            "policy_id": "POL-TEST-001",
            "member_id": "MBR-001",
            "exclusion_type": ExclusionType.MEDICAL,
            "description": "pre-existing condition",
            "effective_date": "2025-01-01",
            "reason_code": "MED-003",
            "raw_text": "Medical exclusion applies",
        }
        if overrides:
            ex.update(overrides)
        return ex

    def test_passes_valid_exclusion(self, base_state):
        state = {**base_state, "exclusions": [self._make_exclusion()]}
        result = validation_node(state)
        assert result["validation_passed"] is True
        assert result["validation_errors"] == []

    def test_fails_missing_effective_date(self, base_state):
        state = {**base_state, "exclusions": [self._make_exclusion({"effective_date": ""})]}
        result = validation_node(state)
        assert result["validation_passed"] is False
        assert any("effective_date" in e for e in result["validation_errors"])

    def test_fails_invalid_reason_code(self, base_state):
        state = {**base_state, "exclusions": [self._make_exclusion({"reason_code": "BAD-999"})]}
        result = validation_node(state)
        assert result["validation_passed"] is False

    def test_status_flagged_on_failure(self, base_state):
        state = {**base_state, "exclusions": [self._make_exclusion({"effective_date": ""})]}
        result = validation_node(state)
        assert result["current_status"] == ExclusionStatus.FLAGGED

    def test_status_validated_on_pass(self, base_state):
        state = {**base_state, "exclusions": [self._make_exclusion()]}
        result = validation_node(state)
        assert result["current_status"] == ExclusionStatus.VALIDATED


# ── Routing Tests ─────────────────────────────────────────────────────────────

class TestRouteAfterValidation:
    def test_routes_to_conflict_detection_when_passed(self, base_state):
        state = {**base_state, "validation_passed": True}
        assert route_after_validation(state) == "conflict_detection"

    def test_routes_to_notification_when_failed(self, base_state):
        state = {**base_state, "validation_passed": False}
        assert route_after_validation(state) == "notification"


# ── Conflict Detection Tests ──────────────────────────────────────────────────

class TestConflictDetectionNode:
    @patch("agents.nodes.dynamo")
    def test_no_conflict_when_clean(self, mock_dynamo, base_state):
        mock_dynamo.check_conflict.return_value = None
        state = {
            **base_state,
            "exclusions": [{
                "exclusion_id": "EX-001",
                "policy_id": "POL-TEST-001",
                "exclusion_type": ExclusionType.MEDICAL,
                "effective_date": "2025-01-01",
            }],
        }
        result = conflict_detection_node(state)
        assert result["conflict_detected"] is False
        assert result["conflicts"] == []

    @patch("agents.nodes.dynamo")
    def test_detects_conflict(self, mock_dynamo, base_state):
        mock_dynamo.check_conflict.return_value = {
            "exclusion_id": "EX-EXISTING",
            "status": "applied",
        }
        state = {
            **base_state,
            "exclusions": [{
                "exclusion_id": "EX-NEW",
                "policy_id": "POL-TEST-001",
                "exclusion_type": ExclusionType.MEDICAL,
                "effective_date": "2025-01-01",
            }],
        }
        result = conflict_detection_node(state)
        assert result["conflict_detected"] is True
        assert len(result["conflicts"]) == 1
        assert result["conflicts"][0]["incoming"] == "EX-NEW"
