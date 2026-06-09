"""
AWS Lambda Handler for the Exclusion Management Agent.

This is the Lambda entry point that:
  1. Parses the incoming S3 event (new policy document uploaded)
  2. Triggers the full LangGraph exclusion pipeline
  3. Returns a structured response with the processing summary

Expected S3 event structure:
  {
    "Records": [
      {
        "s3": {
          "bucket": { "name": "your-bucket" },
          "object": { "key": "incoming/POL-001/document.txt" }
        }
      }
    ]
  }

Policy ID and Member ID are extracted from the S3 key convention:
  incoming/{policy_id}/{member_id}/{filename}
"""

import json
import logging
import os
from urllib.parse import unquote_plus
from agents.graph import run_exclusion_pipeline

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    """Main Lambda handler — triggered by S3 PutObject events."""

    results = []

    for record in event.get("Records", []):
        try:
            bucket = record["s3"]["bucket"]["name"]
            key = unquote_plus(record["s3"]["object"]["key"])

            # Key convention: incoming/{policy_id}/{member_id}/{filename}
            parts = key.split("/")
            if len(parts) < 4:
                logger.error(f"Unexpected key format: {key}")
                results.append({"key": key, "status": "error", "reason": "invalid key format"})
                continue

            policy_id = parts[1]
            member_id = parts[2]

            logger.info(f"Processing: bucket={bucket}, key={key}, policy={policy_id}, member={member_id}")

            final_state = run_exclusion_pipeline(
                policy_id=policy_id,
                member_id=member_id,
                s3_key=key,
            )

            results.append({
                "key": key,
                "policy_id": policy_id,
                "member_id": member_id,
                "status": final_state["current_status"],
                "applied": len(final_state.get("applied_exclusions", [])),
                "rejected": len(final_state.get("rejected_exclusions", [])),
                "conflicts": len(final_state.get("conflicts", [])),
                "validation_passed": final_state.get("validation_passed"),
                "notification_sent": final_state.get("notification_sent"),
                "error": final_state.get("error_message"),
            })

        except Exception as e:
            logger.exception(f"Unhandled error processing record: {e}")
            results.append({"status": "error", "reason": str(e)})

    response = {
        "statusCode": 200,
        "body": json.dumps({"processed": len(results), "results": results}),
    }

    logger.info(f"Lambda complete: {json.dumps(response)}")
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Notification Lambda (separate function deployed alongside main pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def notification_lambda_handler(event: dict, context) -> dict:
    """
    Receives exclusion summary and routes notifications via SNS / SES.
    In production, replace the logger with boto3 SNS/SES publish calls.
    """
    logger.info(f"[NotificationLambda] Received: {json.dumps(event)}")

    policy_id = event.get("policy_id")
    member_id = event.get("member_id")
    applied = event.get("applied_count", 0)
    rejected = event.get("rejected_count", 0)

    # TODO: replace with boto3 SNS publish or SES send_email
    message = (
        f"Policy {policy_id} | Member {member_id} — "
        f"Exclusions applied: {applied}, rejected: {rejected}"
    )
    logger.info(f"[NotificationLambda] Notification: {message}")

    return {"status": "ok", "message": message}


# ─────────────────────────────────────────────────────────────────────────────
# Audit Lambda
# ─────────────────────────────────────────────────────────────────────────────

def audit_lambda_handler(event: dict, context) -> dict:
    """
    Receives the full audit trail and persists it to CloudWatch / DynamoDB.
    In production, write to an audit DynamoDB table or CloudWatch Logs.
    """
    logger.info(f"[AuditLambda] Audit trail received for policy {event.get('policy_id')}")
    for entry in event.get("audit_trail", []):
        logger.info(f"  AUDIT: {entry}")

    return {"status": "logged", "entries": len(event.get("audit_trail", []))}
