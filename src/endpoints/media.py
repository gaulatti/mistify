from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from src import metrics
from src.operations.media_models import (
    MediaOperationRecord,
    MediaOperationRequest,
    MediaOperationState,
    MediaOperationSubmission,
    MediaStageName,
    MediaStageState,
    ModelIdentity,
    StageProvenance,
    utc_now,
)
from src.operations.media_probe import MediaProbeError
from src.operations.models import OperationEnvelope, OperationOptions


router = APIRouter(prefix="/media/operations", tags=["media operations"])


def require_media_auth(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    token = request.state.app_state.config["MEDIA_OPERATION_BEARER_TOKEN"]
    if token is None:
        raise HTTPException(status_code=404, detail="Not Found")
    if not secrets.compare_digest(authorization or "", f"Bearer {token}"):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post(
    "",
    response_model=MediaOperationSubmission,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_media_operation(
    payload: MediaOperationRequest,
    request: Request,
    _authorized: Annotated[None, Depends(require_media_auth)],
) -> MediaOperationSubmission:
    app_state = request.state.app_state
    runner = app_state.media_operation_runner
    if not runner.available:
        raise HTTPException(status_code=503, detail="Media processor unavailable")

    try:
        with metrics.record_media_phase(MediaStageName.PROBE.value):
            probe = await app_state.media_probe.probe(payload.source)
    except MediaProbeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc

    envelope = OperationEnvelope(
        operation_type="process_media",
        payload={"source_checksum_sha256": payload.source.checksum_sha256},
        options=OperationOptions(max_attempts=payload.options.max_attempts),
    )
    record = MediaOperationRecord(
        operation_id=envelope.operation_id,
        source=payload.source,
        options=payload.options,
        probe=probe,
    )
    probe_stage = record.stages[MediaStageName.PROBE]
    probe_stage.state = MediaStageState.SUCCEEDED
    probe_stage.attempts = 1
    probe_stage.output = probe.model_dump(mode="json")
    probe_stage.provenance = StageProvenance(
        **ModelIdentity(
            provider="mistify",
            model="local-media-probe",
            version="1",
        ).model_dump(),
        input_checksum_sha256=payload.source.checksum_sha256,
    )
    probe_stage.started_at = utc_now()
    probe_stage.completed_at = utc_now()
    await app_state.media_operation_store.save(record)

    try:
        enqueued = await app_state.operation_queue.enqueue(envelope)
    except Exception as exc:
        record.state = MediaOperationState.FAILED
        record.error_code = "media_queue_unavailable"
        await app_state.media_operation_store.save(record)
        metrics.record_queue_event("process_media", "error")
        raise HTTPException(status_code=503, detail="Media queue unavailable") from exc

    if not enqueued:
        record.state = MediaOperationState.FAILED
        record.error_code = "media_queue_rejected"
        await app_state.media_operation_store.save(record)
        raise HTTPException(status_code=503, detail="Media queue unavailable")

    metrics.record_queue_event("process_media", "enqueued")
    return MediaOperationSubmission(
        operation_id=record.operation_id,
        state=record.state,
        status_url=str(
            request.url_for("get_media_operation", operation_id=record.operation_id)
        ),
    )


@router.get("/{operation_id}", response_model=MediaOperationRecord)
async def get_media_operation(
    operation_id: str,
    request: Request,
    _authorized: Annotated[None, Depends(require_media_auth)],
) -> MediaOperationRecord:
    record = await request.state.app_state.media_operation_store.get(operation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Media operation not found")
    return record


@router.post("/{operation_id}/cancel", response_model=MediaOperationRecord)
async def cancel_media_operation(
    operation_id: str,
    request: Request,
    _authorized: Annotated[None, Depends(require_media_auth)],
) -> MediaOperationRecord:
    store = request.state.app_state.media_operation_store
    record = await store.get(operation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Media operation not found")
    if record.state not in {
        MediaOperationState.SUCCEEDED,
        MediaOperationState.FAILED,
        MediaOperationState.CANCELED,
    }:
        record.state = MediaOperationState.CANCELED
        for stage in record.stages.values():
            if stage.state == MediaStageState.RUNNING:
                stage.state = MediaStageState.CANCELED
                stage.completed_at = utc_now()
        await store.save(record)
        metrics.record_media_outcome("canceled")
    return record
