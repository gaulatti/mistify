from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class HttpCallback(BaseModel):
    url: str
    method: str = "POST"
    headers: Dict[str, str] = Field(default_factory=dict)

    @field_validator("method")
    @classmethod
    def normalize_method(cls, value: str) -> str:
        method = value.strip().upper()
        if method != "POST":
            raise ValueError("Only POST callbacks are supported")
        return method


class OperationOptions(BaseModel):
    priority: int = 0
    max_attempts: int = 3
    timeout_seconds: int = 300
    store_result: bool = True


class OperationEnvelope(BaseModel):
    operation_id: str = Field(default_factory=lambda: f"op_{uuid4().hex}")
    operation_type: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    callback: Optional[HttpCallback] = None
    options: OperationOptions = Field(default_factory=OperationOptions)
    idempotency_key: Optional[str] = None
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("operation_type")
    @classmethod
    def validate_operation_type(cls, value: str) -> str:
        operation_type = value.strip()
        if not operation_type:
            raise ValueError("operation_type is required")
        return operation_type


class QueuedOperation(BaseModel):
    envelope: OperationEnvelope
    queued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    attempts: int = 0

