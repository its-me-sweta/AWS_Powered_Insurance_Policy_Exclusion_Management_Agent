# Agent-Driven Exclusion Management in Insurance Using LangGraph + S3 + Lambda

_A technical deep-dive into building a fully agentic insurance policy exclusion management pipeline using LangGraph, AWS Lambda, Amazon S3, and DynamoDB — written for the AWS Skill Builder community._

---

## The Problem: Exclusion Management Is Broken

Insurance exclusion management is one of the most operationally painful workflows in the industry. Every policy document can contain multiple exclusion clauses — geographic, medical, occupational, behavioral, financial — each with its own effective dates, reason codes, and downstream implications for claims processing.

The traditional approach: a claims analyst manually reads the policy PDF, extracts exclusion clauses, cross-checks against existing records, applies or rejects them, and then notifies the relevant teams. Per policy. Across thousands of documents per week.

At Carelon Global Solutions, we faced exactly this challenge — a high-volume pipeline of policy exclusion updates, a growing backlog, and a team spending more time on data wrangling than actual underwriting insight. The solution was to build a fully agentic exclusion management pipeline using **LangGraph for orchestration**, **Amazon S3 for document storage and results persistence**, and **AWS Lambda as the event-driven trigger and notification layer**.

This post walks through the architecture, the LangGraph graph design, the AWS integration pattern, and the lessons learned deploying this in a regulated insurance environment.

---

## Architecture Overview

```
S3 PutObject Event
       │
       ▼
  Lambda Trigger
  (handler.py)
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│              LangGraph Exclusion Pipeline                │
│                                                          │
│  [document_ingestion] → [exclusion_extraction]          │
│          → [validation] ──┬──────────────────────────┐  │
│                      (fail)│                   (pass) │  │
│                            ▼                          ▼  │
│                    [notification]    [conflict_detection]│
│                                             │            │
│                                     [application]        │
│                                             │            │
│                                     [notification]       │
└──────────────────────────────────────────────────────────┘
       │
       ▼
  DynamoDB (exclusion records)
  S3 (processed results)
  Lambda → SNS/SES (notifications)
  Lambda → CloudWatch (audit trail)
```

Each box in the pipeline is a **LangGraph node** — a pure Python function that receives the shared `AgentState`, performs one focused task, and returns an updated state. Nodes never call each other directly; the graph handles routing.

---

## Why LangGraph for Insurance Workflows?

Before getting into the code, let's address the obvious question: why LangGraph over a simpler step function or a plain Lambda chain?

**1. Stateful, typed shared context.**  
Every node in the pipeline reads and writes to a single `AgentState` TypedDict. At any point you can inspect exactly what the pipeline knows — which exclusions were found, which failed validation, which caused conflicts. In regulated industries, this auditability is non-negotiable.

**2. Conditional routing without brittle if/else chains.**  
The `add_conditional_edges` API lets you express branching logic declaratively. If validation fails, route directly to notification and skip application entirely. This is one function call, not nested Lambda invocations or Step Functions states.

**3. Operator-based state merging.**  
Lists like `exclusions`, `validation_errors`, and `audit_trail` use `Annotated[List, operator.add]` — meaning parallel branches can safely append to them without race conditions.

**4. Pure Python, no vendor lock-in on orchestration.**  
The graph logic runs anywhere Python runs — Lambda, ECS, a local laptop for testing. AWS is the infrastructure layer, not the orchestration layer.

---

## The State Schema

The heart of any LangGraph pipeline is the state. Here's what flows through our exclusion pipeline:

```python
class AgentState(TypedDict):
    # Input
    policy_id: str
    member_id: str
    raw_document_key: str          # S3 key
    document_content: Optional[str]

    # Extracted
    exclusions: Annotated[List[PolicyExclusion], operator.add]

    # Validation
    validation_errors: Annotated[List[str], operator.add]
    validation_passed: bool

    # Conflicts
    conflicts: Annotated[List[Dict], operator.add]
    conflict_detected: bool

    # Application
    applied_exclusions: Annotated[List[str], operator.add]
    rejected_exclusions: Annotated[List[str], operator.add]

    # Notifications and audit
    notification_sent: bool
    audit_trail: Annotated[List[str], operator.add]
    current_status: ExclusionStatus
```

Every node returns a partial or full copy of this dict. The `Annotated[List, operator.add]` fields accumulate — each node appends without clobbering what came before. This gives you a complete audit trail of every state change, for free.

---

## Node-by-Node Walkthrough

### Node 1: Document Ingestion

```python
def document_ingestion_node(state: AgentState) -> AgentState:
    content = s3.read_document(state["raw_document_key"])
    return {
        **state,
        "document_content": content,
        "current_status": ExclusionStatus.PENDING,
        "audit_trail": [f"{ts()} | INGESTION | Document loaded"],
    }
```

Simple S3 fetch. The `S3Tool` wrapper handles pagination-aware reads and logs the byte count. If the fetch fails, the error is captured in `error_message` and the graph gracefully exits via the notification node rather than raising an unhandled exception.

---

### Node 2: Exclusion Extraction

This is where the actual intelligence lives. In our production system we call Amazon Bedrock (Claude) with a structured prompt asking it to extract exclusion clauses and return JSON. For this reference implementation, a regex-based extractor is included that you can drop-in replace:

```python
_EXCLUSION_PATTERNS = {
    ExclusionType.MEDICAL: r"(?i)(pre[- ]?existing\s+condition|...)",
    ExclusionType.BEHAVIORAL: r"(?i)(self[- ]?inflict|substance\s+abuse|...)",
    # ... other types
}
```

Each match becomes a `PolicyExclusion` TypedDict with a UUID, effective date, reason code, and the raw surrounding text context — giving auditors a direct window into why the exclusion was extracted.

**Production tip:** For Bedrock integration, replace the regex loop with:

```python
import boto3, json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

response = bedrock.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "messages": [{
            "role": "user",
            "content": f"Extract all insurance exclusion clauses from this document as JSON.\n\n{document_text}"
        }]
    })
)
```

---

### Node 3: Validation

Before writing anything to DynamoDB, every extracted exclusion goes through business rule validation:

- `effective_date` must be non-empty and parseable
- `reason_code` must be in the approved set (`GEO-001`, `MED-003`, etc.)
- `description` must be non-empty

If any exclusion fails validation, `validation_passed` is set to `False` and `current_status` becomes `FLAGGED`. The conditional edge after this node immediately routes to the notification node, skipping the entire apply phase. No partial writes, no inconsistent state.

---

### Node 4: Conflict Detection

This node queries DynamoDB for each extracted exclusion to see if a record of the same type and effective date is already in `applied` status:

```python
def check_conflict(self, policy_id, exclusion_type, effective_date):
    existing = self.get_existing_exclusions(policy_id)
    for record in existing:
        if (record["exclusion_type"] == exclusion_type
                and record["effective_date"] == effective_date
                and record["status"] == "applied"):
            return record
    return None
```

Conflicting exclusions get tagged and routed to rejection during application. Non-conflicting exclusions proceed normally. The graph doesn't halt — it handles the mixed batch gracefully within a single pipeline run.

---

### Node 5: Application

```python
def application_node(state: AgentState) -> AgentState:
    conflict_ids = {c["incoming"] for c in state["conflicts"]}

    for ex in state["exclusions"]:
        if ex["exclusion_id"] in conflict_ids:
            rejected.append(ex["exclusion_id"])
        else:
            dynamo.put_exclusion({**ex, "status": "applied"})
            applied.append(ex["exclusion_id"])

    # Persist full result bundle to S3
    s3.write_result(result_bundle, f"processed/{policy_id}/{timestamp}.json")
```

Two writes happen here: DynamoDB records for each applied exclusion, and an S3 JSON bundle with the complete processing summary (applied, rejected, conflicts). The S3 bundle is the source of truth for downstream reporting and regulatory audits.

---

### Node 6: Notification

The final node fires two Lambda invocations:

1. **Notification Lambda** (async): sends a summary to SNS or SES for the business team
2. **Audit Lambda** (sync): persists the full audit trail to CloudWatch Logs or a dedicated DynamoDB audit table

```python
lam.invoke_notification(cfg.NOTIFICATION_LAMBDA, summary_payload)
lam.invoke_audit_logger(cfg.AUDIT_LAMBDA, {"audit_trail": state["audit_trail"], ...})
```

Separating notification from auditing as distinct Lambda functions means each can scale, fail, and be monitored independently.

---

## The Lambda Trigger

The main entry point is a Lambda triggered by S3 PutObject events. The S3 key follows a convention that encodes the routing information:

```
incoming/{policy_id}/{member_id}/{filename}
```

The handler parses the key, extracts `policy_id` and `member_id`, and invokes the pipeline:

```python
def lambda_handler(event, context):
    for record in event["Records"]:
        key = unquote_plus(record["s3"]["object"]["key"])
        parts = key.split("/")
        policy_id, member_id = parts[1], parts[2]

        final_state = run_exclusion_pipeline(policy_id, member_id, key)
        # ... return summary
```

This means the entire pipeline — from S3 upload to DynamoDB write to notification — is triggered by a single file drop. No polling, no scheduler, no manual intervention.

---

## DynamoDB Table Design

```
Table: exclusion-records
PK: policy_id (String)
SK: exclusion_id (String)
GSI: member_id-index (for member-level queries)

Attributes:
  exclusion_type, description, effective_date,
  reason_code, raw_text, status, created_at, updated_at
```

The composite key lets you query all exclusions for a policy efficiently. The GSI on `member_id` supports member-level views without a full scan.

---

## IAM Permissions

The Lambda execution role needs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::insurance-exclusion-pipeline/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:UpdateItem",
        "dynamodb:Query"
      ],
      "Resource": "arn:aws:dynamodb:*:*:table/exclusion-records*"
    },
    {
      "Effect": "Allow",
      "Action": ["lambda:InvokeFunction"],
      "Resource": [
        "arn:aws:lambda:*:*:function:exclusion-notification-fn",
        "arn:aws:lambda:*:*:function:exclusion-audit-fn"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

Least-privilege: the main pipeline Lambda cannot write to notification-fn's resources and vice versa.

---

## Testing Strategy

The test suite uses `pytest` with `unittest.mock` to isolate AWS calls:

```python
@patch("agents.nodes.dynamo")
def test_detects_conflict(self, mock_dynamo, base_state):
    mock_dynamo.check_conflict.return_value = {"exclusion_id": "EX-EXISTING"}
    result = conflict_detection_node(state)
    assert result["conflict_detected"] is True
```

Key testing principle: **every node is tested in pure isolation**. No real AWS calls in unit tests. The `S3Tool`, `DynamoDBTool`, and `LambdaTool` classes are mocked at the module level. For integration tests, use `moto` to spin up local AWS mock endpoints.

Run the full suite:

```bash
pip install -r requirements.txt
pytest tests/ -v --tb=short
```

---

## Results and Impact

After deploying this pipeline in a pilot on 3,000 policy documents:

| Metric                        | Before                 | After                      |
| ----------------------------- | ---------------------- | -------------------------- |
| Manual review time per policy | ~12 min                | ~0 min                     |
| Exclusion extraction accuracy | ~87% (human)           | ~94% (agent + validation)  |
| Conflict detection latency    | 2–3 days               | < 30 seconds               |
| Audit trail completeness      | Partial (email chains) | 100% (structured DynamoDB) |
| Processing throughput         | ~200 policies/day      | ~5,000 policies/day        |

The biggest win wasn't speed — it was **audit completeness**. Every exclusion decision now has a timestamped, structured trail from document ingestion to DynamoDB write to notification. In a regulatory audit, this is the difference between a clean response and a painful evidence-gathering exercise.

---

## What's Next

This architecture is intentionally modular. Natural extensions include:

**1. Bedrock-powered extraction:** Swap the regex extractor for a Bedrock Claude call with a structured JSON output schema. This handles free-form exclusion language that regex can't reliably match.

**2. Human-in-the-loop escalation:** Add a `human_review_node` between conflict detection and application. Flagged exclusions get routed to an SQS queue; a reviewer approves or rejects via a lightweight API; the graph resumes from a checkpoint.

**3. LangGraph checkpointing:** Enable LangGraph's built-in checkpointer (backed by DynamoDB or Redis) so that long-running pipelines can pause and resume without losing state — critical for documents that require multi-day review cycles.

**4. Multi-LOB graph:** Use a router node at the start to branch into LOB-specific sub-graphs (auto, health, life) before merging back at the notification layer.

---

## Getting Started

```bash
git clone <your-repo>
cd exclusion-agent
pip install -r requirements.txt

# Set environment variables
export S3_BUCKET=insurance-exclusion-pipeline
export DYNAMO_TABLE=exclusion-records
export NOTIFICATION_LAMBDA=exclusion-notification-fn
export AUDIT_LAMBDA=exclusion-audit-fn

# Run locally
python -m agents.graph

# Run tests
pytest tests/ -v
```

For Lambda deployment, package the project with dependencies and configure the S3 trigger on the `incoming/` prefix.

---

## Key Takeaways

- **LangGraph is production-ready for regulated domains.** The typed state, conditional edges, and audit trail capabilities align naturally with insurance compliance requirements.
- **S3 as event bus + storage simplifies the architecture dramatically.** A file drop triggers the entire pipeline, and the final result lands back in S3. No extra queue infrastructure needed for basic use cases.
- **Separate node concerns strictly.** Each node does one thing. Extraction doesn't validate. Validation doesn't persist. This makes the pipeline testable, debuggable, and evolvable without touching unrelated logic.
- **Build for auditors, not just engineers.** The structured `audit_trail` field — accumulated across every node — was not an afterthought. In insurance, your pipeline's paper trail matters as much as its accuracy.

---

_Sweta Jha is a Senior Data Scientist and AI/ML Engineer at Carelon Global Solutions (Elevance Health) and an AWS Community Builder. She specializes in Agentic AI, LangGraph-based multi-agent systems, and MLOps for healthcare and insurance domains._

---

**Tags:** `#AWSCommunityBuilder` `#LangGraph` `#AgenticAI` `#AWSLambda` `#InsurTech` `#GenerativeAI` `#MLOps` `#AmazonS3` `#DynamoDB` `#HealthcareAI`

---

**GitHub:** [AWS Powered Insurance Policy Exclusion Management Agent](https://github.com/its-me-sweta/AWS_Powered_Insurance_Policy_Exclusion_Management_Agent)
