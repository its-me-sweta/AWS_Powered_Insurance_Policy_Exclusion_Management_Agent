"""
AWS Tool wrappers used by LangGraph agent nodes.
Handles S3 read/write, DynamoDB CRUD, and Lambda invocation.
"""

import json
import boto3
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# S3 Tools
# ──────────────────────────────────────────────

class S3Tool:
    """Handles policy document retrieval and exclusion result persistence."""

    def __init__(self, bucket_name: str, region: str = "us-east-1"):
        self.bucket = bucket_name
        self.client = boto3.client("s3", region_name=region)

    def read_document(self, s3_key: str) -> str:
        """Fetch raw policy document text from S3."""
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=s3_key)
            content = response["Body"].read().decode("utf-8")
            logger.info(f"[S3Tool] Read document: {s3_key} ({len(content)} chars)")
            return content
        except Exception as e:
            logger.error(f"[S3Tool] Failed to read {s3_key}: {e}")
            raise

    def write_result(self, result: Dict[str, Any], output_key: str) -> str:
        """Persist processed exclusion results back to S3."""
        try:
            body = json.dumps(result, indent=2, default=str)
            self.client.put_object(
                Bucket=self.bucket,
                Key=output_key,
                Body=body.encode("utf-8"),
                ContentType="application/json",
            )
            logger.info(f"[S3Tool] Wrote result to: {output_key}")
            return f"s3://{self.bucket}/{output_key}"
        except Exception as e:
            logger.error(f"[S3Tool] Failed to write {output_key}: {e}")
            raise

    def list_pending_documents(self, prefix: str = "incoming/") -> List[str]:
        """List all incoming policy documents awaiting processing."""
        paginator = self.client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys


# ──────────────────────────────────────────────
# DynamoDB Tools
# ──────────────────────────────────────────────

class DynamoDBTool:
    """CRUD operations for exclusion records."""

    def __init__(self, table_name: str, region: str = "us-east-1"):
        self.table_name = table_name
        self.dynamodb = boto3.resource("dynamodb", region_name=region)
        self.table = self.dynamodb.Table(table_name)

    def get_existing_exclusions(self, policy_id: str) -> List[Dict]:
        """Retrieve all exclusions currently on record for a policy."""
        try:
            response = self.table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key("policy_id").eq(policy_id)
            )
            return response.get("Items", [])
        except Exception as e:
            logger.error(f"[DynamoDB] Query failed for policy {policy_id}: {e}")
            return []

    def put_exclusion(self, exclusion: Dict[str, Any]) -> bool:
        """Insert or overwrite an exclusion record."""
        try:
            exclusion["created_at"] = datetime.utcnow().isoformat()
            self.table.put_item(Item=exclusion)
            logger.info(f"[DynamoDB] Saved exclusion: {exclusion.get('exclusion_id')}")
            return True
        except Exception as e:
            logger.error(f"[DynamoDB] Put failed: {e}")
            return False

    def update_exclusion_status(self, policy_id: str, exclusion_id: str, status: str) -> bool:
        """Update the processing status of a single exclusion."""
        try:
            self.table.update_item(
                Key={"policy_id": policy_id, "exclusion_id": exclusion_id},
                UpdateExpression="SET #s = :status, updated_at = :ts",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={
                    ":status": status,
                    ":ts": datetime.utcnow().isoformat(),
                },
            )
            return True
        except Exception as e:
            logger.error(f"[DynamoDB] Update failed: {e}")
            return False

    def check_conflict(self, policy_id: str, exclusion_type: str, effective_date: str) -> Optional[Dict]:
        """
        Detect if an incoming exclusion conflicts with an existing one
        of the same type and overlapping effective date.
        """
        existing = self.get_existing_exclusions(policy_id)
        for record in existing:
            if (
                record.get("exclusion_type") == exclusion_type
                and record.get("effective_date") == effective_date
                and record.get("status") == "applied"
            ):
                return record
        return None


# ──────────────────────────────────────────────
# Lambda Trigger Tool
# ──────────────────────────────────────────────

class LambdaTool:
    """Invokes downstream Lambda functions for notifications and auditing."""

    def __init__(self, region: str = "us-east-1"):
        self.client = boto3.client("lambda", region_name=region)

    def invoke_notification(self, function_name: str, payload: Dict[str, Any]) -> Dict:
        """Fire-and-forget invocation of the notification Lambda."""
        try:
            response = self.client.invoke(
                FunctionName=function_name,
                InvocationType="Event",           # async
                Payload=json.dumps(payload).encode("utf-8"),
            )
            logger.info(f"[Lambda] Triggered {function_name}: {response['StatusCode']}")
            return {"status": "triggered", "status_code": response["StatusCode"]}
        except Exception as e:
            logger.error(f"[Lambda] Invocation failed: {e}")
            return {"status": "failed", "error": str(e)}

    def invoke_audit_logger(self, function_name: str, audit_record: Dict[str, Any]) -> Dict:
        """Synchronously invoke the audit logging Lambda."""
        try:
            response = self.client.invoke(
                FunctionName=function_name,
                InvocationType="RequestResponse",  # sync
                Payload=json.dumps(audit_record).encode("utf-8"),
            )
            result = json.loads(response["Payload"].read())
            logger.info(f"[Lambda] Audit logged: {result}")
            return result
        except Exception as e:
            logger.error(f"[Lambda] Audit Lambda failed: {e}")
            return {"status": "failed", "error": str(e)}
