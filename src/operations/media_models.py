from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MediaType(str, Enum):
    WAV = "audio/wav"
    MP3 = "audio/mpeg"
    MP4 = "video/mp4"


class MediaOperationState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class MediaStageName(str, Enum):
    PROBE = "probe"
    TRANSCRIPTION = "transcription"
    DIARIZATION = "diarization"
    SUMMARY = "summary"


class MediaStageState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class SpeakerState(str, Enum):
    IDENTIFIED = "identified"
    UNKNOWN = "unknown"


class MediaSource(BaseModel):
    uri: str = Field(min_length=1, max_length=2048)
    checksum_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    media_type: MediaType
    language_hint: Optional[str] = Field(default=None, min_length=2, max_length=35)

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, value: str) -> str:
        parsed = urlparse(value)
        if not parsed.scheme:
            raise ValueError("uri must be absolute")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("uri credentials, query, and fragment are not supported")
        return value

    @field_validator("checksum_sha256")
    @classmethod
    def normalize_checksum(cls, value: str) -> str:
        return value.lower()


class MediaOperationOptions(BaseModel):
    max_attempts: int = Field(default=3, ge=1, le=10)
    retention_seconds: int = Field(default=7 * 24 * 60 * 60, ge=300, le=30 * 24 * 60 * 60)


class MediaOperationRequest(BaseModel):
    source: MediaSource
    options: MediaOperationOptions = Field(default_factory=MediaOperationOptions)


class MediaProbeResult(BaseModel):
    checksum_sha256: str
    detected_media_type: MediaType
    duration_ms: int = Field(gt=0)
    size_bytes: int = Field(gt=0)


class TimeRange(BaseModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_order(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class TranscriptSegment(TimeRange):
    text: str = Field(min_length=1, max_length=20_000)


class TranscriptionArtifact(BaseModel):
    language: str = Field(min_length=2, max_length=35)
    language_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    segments: list[TranscriptSegment] = Field(min_length=1)


class DiarizedSegment(TranscriptSegment):
    speaker_state: SpeakerState
    speaker_label: Optional[str] = Field(default=None, min_length=1, max_length=80)
    speaker_confidence: Optional[float] = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_speaker(self):
        if self.speaker_state == SpeakerState.UNKNOWN:
            if self.speaker_label is not None or self.speaker_confidence is not None:
                raise ValueError("unknown speakers cannot have a label or confidence")
        elif self.speaker_label is None or self.speaker_confidence is None:
            raise ValueError("identified speakers require a label and confidence")
        return self


class DiarizationArtifact(BaseModel):
    segments: list[DiarizedSegment] = Field(min_length=1)


class SummaryPoint(BaseModel):
    text: str = Field(min_length=1, max_length=4_000)
    source_timecodes: list[TimeRange] = Field(min_length=1)


class SummaryArtifact(BaseModel):
    points: list[SummaryPoint] = Field(min_length=1)


class ModelIdentity(BaseModel):
    provider: str = Field(min_length=1, max_length=120)
    model: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=120)


class StageProvenance(ModelIdentity):
    input_checksum_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class MediaStageRecord(BaseModel):
    state: MediaStageState = MediaStageState.PENDING
    attempts: int = 0
    output: Optional[Dict[str, Any]] = None
    provenance: Optional[StageProvenance] = None
    error_code: Optional[str] = Field(default=None, max_length=80)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


def initial_stages() -> Dict[MediaStageName, MediaStageRecord]:
    return {stage: MediaStageRecord() for stage in MediaStageName}


class MediaOperationResult(BaseModel):
    source_checksum_sha256: str
    duration_ms: int
    language: str
    language_confidence: Optional[float] = None
    transcript: list[DiarizedSegment]
    summary: list[SummaryPoint]
    provenance: Dict[MediaStageName, StageProvenance]


class MediaOperationRecord(BaseModel):
    operation_id: str
    state: MediaOperationState = MediaOperationState.QUEUED
    source: MediaSource
    options: MediaOperationOptions
    probe: MediaProbeResult
    attempts: int = 0
    stages: Dict[MediaStageName, MediaStageRecord] = Field(default_factory=initial_stages)
    result: Optional[MediaOperationResult] = None
    error_code: Optional[str] = Field(default=None, max_length=80)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class MediaOperationSubmission(BaseModel):
    operation_id: str
    state: MediaOperationState
    status_url: str


class MediaRunOutcome(BaseModel):
    state: MediaOperationState
    result: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    retry: bool = False
