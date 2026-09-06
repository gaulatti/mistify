from __future__ import annotations

from typing import Protocol

from redis.asyncio import Redis

from src import metrics
from src.operations.media_models import (
    DiarizationArtifact,
    MediaOperationRecord,
    MediaOperationResult,
    MediaOperationState,
    MediaRunOutcome,
    MediaStageName,
    MediaStageState,
    MediaProbeResult,
    MediaSource,
    ModelIdentity,
    StageProvenance,
    SummaryArtifact,
    TranscriptionArtifact,
    utc_now,
)


MEDIA_OPERATION_PREFIX = "mistify:media:operation"


class MediaProcessingError(Exception):
    def __init__(self, code: str, *, retryable: bool) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class MediaBackend(Protocol):
    @property
    def available(self) -> bool: ...

    def identity(self, stage: MediaStageName) -> ModelIdentity: ...

    async def transcribe(
        self, source: MediaSource, probe: MediaProbeResult
    ) -> TranscriptionArtifact: ...

    async def diarize(
        self,
        source: MediaSource,
        probe: MediaProbeResult,
        transcription: TranscriptionArtifact,
    ) -> DiarizationArtifact: ...

    async def summarize(
        self,
        source: MediaSource,
        probe: MediaProbeResult,
        diarization: DiarizationArtifact,
    ) -> SummaryArtifact: ...


class UnavailableMediaBackend:
    """Fail-closed backend used until an operator selects pinned processors."""

    available = False

    def identity(self, stage: MediaStageName) -> ModelIdentity:
        raise MediaProcessingError("media_processor_unavailable", retryable=False)

    async def transcribe(
        self, source: MediaSource, probe: MediaProbeResult
    ) -> TranscriptionArtifact:
        raise MediaProcessingError("media_processor_unavailable", retryable=False)

    async def diarize(
        self,
        source: MediaSource,
        probe: MediaProbeResult,
        transcription: TranscriptionArtifact,
    ) -> DiarizationArtifact:
        raise MediaProcessingError("media_processor_unavailable", retryable=False)

    async def summarize(
        self,
        source: MediaSource,
        probe: MediaProbeResult,
        diarization: DiarizationArtifact,
    ) -> SummaryArtifact:
        raise MediaProcessingError("media_processor_unavailable", retryable=False)


class MediaOperationStore:
    def __init__(self, redis_client: Redis) -> None:
        self.redis = redis_client

    @staticmethod
    def key(operation_id: str) -> str:
        return f"{MEDIA_OPERATION_PREFIX}:{operation_id}"

    async def save(self, record: MediaOperationRecord) -> None:
        record.updated_at = utc_now()
        await self.redis.set(
            self.key(record.operation_id),
            record.model_dump_json(),
            ex=record.options.retention_seconds,
        )

    async def get(self, operation_id: str) -> MediaOperationRecord | None:
        raw = await self.redis.get(self.key(operation_id))
        if raw is None:
            return None
        return MediaOperationRecord.model_validate_json(raw)


class MediaOperationRunner:
    _STAGE_ORDER = (
        MediaStageName.TRANSCRIPTION,
        MediaStageName.DIARIZATION,
        MediaStageName.SUMMARY,
    )

    def __init__(self, store: MediaOperationStore, backend: MediaBackend) -> None:
        self.store = store
        self.backend = backend

    @property
    def available(self) -> bool:
        return self.backend.available

    async def run(self, operation_id: str) -> MediaRunOutcome:
        record = await self.store.get(operation_id)
        if record is None:
            return MediaRunOutcome(
                state=MediaOperationState.FAILED,
                error_code="media_operation_not_found",
            )
        if record.state in {
            MediaOperationState.SUCCEEDED,
            MediaOperationState.FAILED,
            MediaOperationState.CANCELED,
        }:
            return self._outcome(record)

        queue_age = max(0.0, (utc_now() - record.created_at).total_seconds())
        metrics.record_media_queue_age(queue_age)
        record.attempts += 1
        record.state = MediaOperationState.RUNNING
        record.error_code = None
        await self.store.save(record)

        for stage_name in self._STAGE_ORDER:
            record = await self.store.get(operation_id)
            if record is None:
                return MediaRunOutcome(
                    state=MediaOperationState.FAILED,
                    error_code="media_operation_not_found",
                )
            if record.state == MediaOperationState.CANCELED:
                metrics.record_media_outcome("canceled")
                return self._outcome(record)
            if record.stages[stage_name].state == MediaStageState.SUCCEEDED:
                continue

            outcome = await self._run_stage(record, stage_name)
            if outcome is not None:
                return outcome

        record = await self.store.get(operation_id)
        if record is None:
            return MediaRunOutcome(
                state=MediaOperationState.FAILED,
                error_code="media_operation_not_found",
            )
        if record.state == MediaOperationState.CANCELED:
            metrics.record_media_outcome("canceled")
            return self._outcome(record)

        try:
            record.result = self._build_result(record)
        except MediaProcessingError as exc:
            return await self._fail_record(record, exc.code, retryable=False)
        record.state = MediaOperationState.SUCCEEDED
        record.error_code = None
        await self.store.save(record)
        metrics.record_media_outcome("succeeded")
        return self._outcome(record)

    async def _run_stage(
        self, record: MediaOperationRecord, stage_name: MediaStageName
    ) -> MediaRunOutcome | None:
        stage = record.stages[stage_name]
        stage.state = MediaStageState.RUNNING
        stage.attempts += 1
        stage.started_at = utc_now()
        stage.completed_at = None
        stage.error_code = None
        await self.store.save(record)

        try:
            with metrics.record_media_phase(stage_name.value):
                output = await self._invoke_stage(record, stage_name)
                self._validate_stage_output(record, stage_name, output)
                identity = self.backend.identity(stage_name)
                provenance = StageProvenance(
                    **identity.model_dump(),
                    input_checksum_sha256=record.source.checksum_sha256,
                )
        except MediaProcessingError as exc:
            return await self._fail_stage(record, stage_name, exc.code, exc.retryable)
        except Exception:
            return await self._fail_stage(
                record,
                stage_name,
                f"{stage_name.value}_failed",
                True,
            )

        latest = await self.store.get(record.operation_id)
        if latest is None:
            return MediaRunOutcome(
                state=MediaOperationState.FAILED,
                error_code="media_operation_not_found",
            )
        if latest.state == MediaOperationState.CANCELED:
            latest_stage = latest.stages[stage_name]
            latest_stage.state = MediaStageState.CANCELED
            latest_stage.completed_at = utc_now()
            await self.store.save(latest)
            metrics.record_media_outcome("canceled")
            return self._outcome(latest)

        latest_stage = latest.stages[stage_name]
        latest_stage.state = MediaStageState.SUCCEEDED
        latest_stage.output = output.model_dump(mode="json")
        latest_stage.provenance = provenance
        latest_stage.error_code = None
        latest_stage.completed_at = utc_now()
        await self.store.save(latest)
        return None

    async def _invoke_stage(self, record: MediaOperationRecord, stage_name: MediaStageName):
        if stage_name == MediaStageName.TRANSCRIPTION:
            return await self.backend.transcribe(record.source, record.probe)

        transcription = self._transcription(record)
        if stage_name == MediaStageName.DIARIZATION:
            return await self.backend.diarize(
                record.source, record.probe, transcription
            )

        diarization = self._diarization(record)
        return await self.backend.summarize(record.source, record.probe, diarization)

    async def _fail_stage(
        self,
        record: MediaOperationRecord,
        stage_name: MediaStageName,
        error_code: str,
        retryable: bool,
    ) -> MediaRunOutcome:
        latest = await self.store.get(record.operation_id) or record
        if latest.state == MediaOperationState.CANCELED:
            metrics.record_media_outcome("canceled")
            return self._outcome(latest)
        stage = latest.stages[stage_name]
        stage.state = MediaStageState.FAILED
        stage.error_code = self._safe_error_code(error_code, stage_name)
        stage.completed_at = utc_now()
        await self.store.save(latest)
        return await self._fail_record(
            latest,
            stage.error_code,
            retryable=retryable,
        )

    async def _fail_record(
        self, record: MediaOperationRecord, error_code: str, *, retryable: bool
    ) -> MediaRunOutcome:
        record.error_code = error_code
        if retryable and record.attempts < record.options.max_attempts:
            record.state = MediaOperationState.QUEUED
            await self.store.save(record)
            metrics.record_media_outcome("retry")
            return MediaRunOutcome(
                state=record.state,
                error_code=record.error_code,
                retry=True,
            )

        record.state = MediaOperationState.FAILED
        await self.store.save(record)
        metrics.record_media_outcome("failed")
        return self._outcome(record)

    @staticmethod
    def _safe_error_code(error_code: str, stage_name: MediaStageName) -> str:
        allowed = {
            "media_processor_unavailable",
            "invalid_transcription",
            "invalid_diarization",
            "ungrounded_summary",
        }
        return error_code if error_code in allowed else f"{stage_name.value}_failed"

    @classmethod
    def _validate_stage_output(cls, record, stage_name, output) -> None:
        if stage_name == MediaStageName.TRANSCRIPTION:
            cls._validate_transcription(record.probe, output)
        elif stage_name == MediaStageName.DIARIZATION:
            cls._validate_diarization(cls._transcription(record), output)
        else:
            cls._validate_summary(cls._diarization(record), output)

    @staticmethod
    def _validate_transcription(
        probe: MediaProbeResult, artifact: TranscriptionArtifact
    ) -> None:
        previous_end = 0
        for segment in artifact.segments:
            if segment.start_ms < previous_end or segment.end_ms > probe.duration_ms:
                raise MediaProcessingError("invalid_transcription", retryable=False)
            previous_end = segment.end_ms

    @staticmethod
    def _validate_diarization(
        transcription: TranscriptionArtifact, artifact: DiarizationArtifact
    ) -> None:
        if len(transcription.segments) != len(artifact.segments):
            raise MediaProcessingError("invalid_diarization", retryable=False)
        for source, diarized in zip(transcription.segments, artifact.segments):
            if (
                source.start_ms != diarized.start_ms
                or source.end_ms != diarized.end_ms
                or source.text != diarized.text
            ):
                raise MediaProcessingError("invalid_diarization", retryable=False)

    @staticmethod
    def _validate_summary(
        diarization: DiarizationArtifact, artifact: SummaryArtifact
    ) -> None:
        source_ranges = {
            (segment.start_ms, segment.end_ms) for segment in diarization.segments
        }
        for point in artifact.points:
            if any(
                (timecode.start_ms, timecode.end_ms) not in source_ranges
                for timecode in point.source_timecodes
            ):
                raise MediaProcessingError("ungrounded_summary", retryable=False)

    @staticmethod
    def _transcription(record: MediaOperationRecord) -> TranscriptionArtifact:
        output = record.stages[MediaStageName.TRANSCRIPTION].output
        if output is None:
            raise MediaProcessingError("invalid_transcription", retryable=False)
        return TranscriptionArtifact.model_validate(output)

    @staticmethod
    def _diarization(record: MediaOperationRecord) -> DiarizationArtifact:
        output = record.stages[MediaStageName.DIARIZATION].output
        if output is None:
            raise MediaProcessingError("invalid_diarization", retryable=False)
        return DiarizationArtifact.model_validate(output)

    @staticmethod
    def _build_result(record: MediaOperationRecord) -> MediaOperationResult:
        transcription = MediaOperationRunner._transcription(record)
        diarization = MediaOperationRunner._diarization(record)
        summary_output = record.stages[MediaStageName.SUMMARY].output
        if summary_output is None:
            raise MediaProcessingError("ungrounded_summary", retryable=False)
        summary = SummaryArtifact.model_validate(summary_output)
        provenance = {
            stage_name: stage.provenance
            for stage_name, stage in record.stages.items()
            if stage.provenance is not None
        }
        if len(provenance) != len(MediaStageName):
            raise MediaProcessingError("media_provenance_incomplete", retryable=False)
        return MediaOperationResult(
            source_checksum_sha256=record.source.checksum_sha256,
            duration_ms=record.probe.duration_ms,
            language=transcription.language,
            language_confidence=transcription.language_confidence,
            transcript=diarization.segments,
            summary=summary.points,
            provenance=provenance,
        )

    @staticmethod
    def _outcome(record: MediaOperationRecord) -> MediaRunOutcome:
        return MediaRunOutcome(
            state=record.state,
            result=(record.result.model_dump(mode="json") if record.result else None),
            error_code=record.error_code,
        )
