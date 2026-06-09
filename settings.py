"""
Configuration settings loaded from environment variables.
Set these in your Lambda environment or .env file locally.
"""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # AWS
    S3_BUCKET: str = os.getenv("S3_BUCKET", "insurance-exclusion-pipeline")
    DYNAMO_TABLE: str = os.getenv("DYNAMO_TABLE", "exclusion-records")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")

    # Lambda function names
    NOTIFICATION_LAMBDA: str = os.getenv("NOTIFICATION_LAMBDA", "exclusion-notification-fn")
    AUDIT_LAMBDA: str = os.getenv("AUDIT_LAMBDA", "exclusion-audit-fn")

    # Pipeline tuning
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
