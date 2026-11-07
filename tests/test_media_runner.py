"""Covers `src/operations/media.py`: the durable media-operation store and runner.

The runner executes transcription, diarization and summary as resumable stages.
Its contract: stages already succeeded are never redone; a stage output is
validated against the probe and the previous stage before it is accepted;
transient failures requeue until attempts run out; a cancel observed at any point
wins over progress; and error codes surfaced to callers come from a fixed set.
The processors are fakes; the store is an in-memory stand-in for Valkey.
"""

import pytest

from src.operations.media import (
    MEDIA_OPERATION_PREFIX,
    MediaOperationRunner,
    MediaOperationStore,
    MediaProcessingError,
    UnavailableMediaBackend,
)
from src.operations.media_models import (
    DiarizationArtifact,
    DiarizedSegment,
    MediaOperationOptions,
    MediaOperationRecord,
    MediaOperationState,
    MediaProbeResult,
    MediaSource,
    MediaStageName,
    MediaStageState,
    MediaType,
    ModelIdentity,
    SpeakerState,
    StageProvenance,
    SummaryArtifact,
    SummaryPoint,
    TimeRange,
    TranscriptSegment,
    TranscriptionArtifact,
)

CHECKSUM = "a" * 64
OP = "op_runner"

SEGMENTS = [(0, 1000, "first line"), (1000, 2000, "second line")]


def source():
    return MediaSource(uri="file:///media/a.wav", checksum_sha256=CHECKSUM, media_type=MediaType.WAV)


def probe(duration_ms=10_000):
    return MediaProbeResult(
        checksum_sha256=CHECKSUM, detected_media_type=MediaType.WAV, duration_ms=duration_ms, size_bytes=100
    )


def transcription(segments=SEGMENTS):
    return TranscriptionArtifact(
        language="en",
        language_confidence=0.9,
        segments=[TranscriptSegment(start_ms=s, end_ms=e, text=t) for s, e, t in segments],
    )


def diarization(segments=SEGMENTS):
    return DiarizationArtifact(segments=[
        DiarizedSegment(
            start_ms=s, end_ms=e, text=t, speaker_state=SpeakerState.IDENTIFIED,
            speaker_label="Ana", speaker_confidence=0.8,
        )
        for s, e, t in segments
    ])


def summary(ranges=((1000, 2000),)):
    return SummaryArtifact(points=[
        SummaryPoint(text="A fact.", source_timecodes=[TimeRange(start_ms=s, end_ms=e) for s, e in ranges])
    ])


def record(**fields):
    rec = MediaOperationRecord(
        operation_id=OP, source=source(), probe=probe(), options=MediaOperationOptions(), **fields
    )
    # Admission has already probed the media, so that stage is complete.
    stage = rec.stages[MediaStageName.PROBE]
    stage.state = MediaStageState.SUCCEEDED
    stage.provenance = StageProvenance(
        provider="mistify", model="local-media-probe", version="1", input_checksum_sha256=CHECKSUM
    )
    return rec


class FakeRedis:
    def __init__(self):
        self.values, self.ttls = {}, {}

    async def set(self, key, value, *, ex):
        self.values[key], self.ttls[key] = value, ex

    async def get(self, key):
        return self.values.get(key)


class ScriptedStore(MediaOperationStore):
    """Store whose Nth `get` can be replaced, to model concurrent changes."""

    def __init__(self):
        super().__init__(FakeRedis())
        self.gets = 0
        self.script = {}

    async def get(self, operation_id):
        self.gets += 1
        found = await super().get(operation_id)
        change = self.script.get(self.gets)
        return change(found) if change else found


def canceled(found):
    found.state = MediaOperationState.CANCELED
    return found


class FakeBackend:
    available = True

    def __init__(self):
        self.calls = []
        self.errors = {}
        self.outputs = {
            MediaStageName.TRANSCRIPTION: transcription(),
            MediaStageName.DIARIZATION: diarization(),
            MediaStageName.SUMMARY: summary(),
        }
        self.identity_error = None
        self.during = {}

    def identity(self, stage):
        if self.identity_error:
            raise self.identity_error
        return ModelIdentity(provider="fixture", model=f"{stage.value}-model", version="1")

    async def _run(self, stage):
        self.calls.append(stage)
        if stage in self.during:
            await self.during[stage]()
        if stage in self.errors:
            raise self.errors[stage]
        return self.outputs[stage]

    async def transcribe(self, source, probe):
        return await self._run(MediaStageName.TRANSCRIPTION)

    async def diarize(self, source, probe, transcription):
        return await self._run(MediaStageName.DIARIZATION)

    async def summarize(self, source, probe, diarization):
        return await self._run(MediaStageName.SUMMARY)


async def setup(rec=None, backend=None):
    store = ScriptedStore()
    await store.save(rec or record())
    backend = backend or FakeBackend()
    return store, backend, MediaOperationRunner(store, backend)


async def stored(store):
    return await MediaOperationStore.get(store, OP)


# --- the fail-closed backend ------------------------------------------------

@pytest.mark.asyncio
async def test_unavailable_backend_refuses_every_stage_without_retry():
    backend = UnavailableMediaBackend()
    calls = [
        lambda: backend.identity(MediaStageName.SUMMARY),
        lambda: backend.transcribe(source(), probe()),
        lambda: backend.diarize(source(), probe(), transcription()),
        lambda: backend.summarize(source(), probe(), diarization()),
    ]

    assert backend.available is False
    for call in calls:
        with pytest.raises(MediaProcessingError) as caught:
            result = call()
            if hasattr(result, "__await__"):
                await result
        assert (caught.value.code, caught.value.retryable) == ("media_processor_unavailable", False)


@pytest.mark.asyncio
async def test_runner_availability_follows_its_backend():
    store, backend, runner = await setup()
    assert runner.available is True
    backend.available = False
    assert runner.available is False


# --- the store --------------------------------------------------------------

@pytest.mark.asyncio
async def test_store_round_trips_a_record_with_a_retention_ttl():
    store = MediaOperationStore(FakeRedis())
    rec = record()
    before = rec.updated_at

    await store.save(rec)

    key = f"{MEDIA_OPERATION_PREFIX}:{OP}"
    assert MediaOperationStore.key(OP) == key
    assert store.redis.ttls[key] == rec.options.retention_seconds
    loaded = await store.get(OP)
    assert loaded.operation_id == OP and loaded.updated_at >= before
    assert await store.get("op_missing") is None


# --- run: terminal and missing records --------------------------------------

@pytest.mark.asyncio
async def test_a_missing_operation_fails_with_not_found():
    _store, backend, runner = await setup()

    outcome = await runner.run("op_unknown")

    assert (outcome.state, outcome.error_code) == (MediaOperationState.FAILED, "media_operation_not_found")
    assert backend.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", [MediaOperationState.SUCCEEDED, MediaOperationState.FAILED, MediaOperationState.CANCELED]
)
async def test_terminal_operations_are_replayed_without_running_processors(state):
    store, backend, runner = await setup(record(state=state, error_code="earlier"))

    outcome = await runner.run(OP)

    assert (outcome.state, outcome.error_code, outcome.retry) == (state, "earlier", False)
    assert backend.calls == []


# --- run: success -----------------------------------------------------------

@pytest.mark.asyncio
async def test_a_clean_run_succeeds_with_grounded_result_and_full_provenance():
    store, backend, runner = await setup()

    outcome = await runner.run(OP)

    assert outcome.state == MediaOperationState.SUCCEEDED and outcome.error_code is None
    assert backend.calls == [MediaStageName.TRANSCRIPTION, MediaStageName.DIARIZATION, MediaStageName.SUMMARY]
    result = outcome.result
    assert result["language"] == "en" and result["duration_ms"] == 10_000
    assert [s["text"] for s in result["transcript"]] == ["first line", "second line"]
    assert result["summary"][0]["source_timecodes"] == [{"start_ms": 1000, "end_ms": 2000}]
    assert set(result["provenance"]) == {"probe", "transcription", "diarization", "summary"}
    assert result["provenance"]["summary"]["input_checksum_sha256"] == CHECKSUM
    saved = await stored(store)
    assert saved.attempts == 1 and saved.state == MediaOperationState.SUCCEEDED
    assert all(saved.stages[s].state == MediaStageState.SUCCEEDED for s in MediaOperationRunner._STAGE_ORDER)


@pytest.mark.asyncio
async def test_stages_that_already_succeeded_are_not_run_again():
    store, backend, runner = await setup()
    backend.errors[MediaStageName.SUMMARY] = RuntimeError("flaky")
    first = await runner.run(OP)
    assert first.retry is True

    del backend.errors[MediaStageName.SUMMARY]
    backend.calls.clear()
    second = await runner.run(OP)

    assert second.state == MediaOperationState.SUCCEEDED
    assert backend.calls == [MediaStageName.SUMMARY]
    assert (await stored(store)).attempts == 2


# --- run: failures and retry ------------------------------------------------

@pytest.mark.asyncio
async def test_a_transient_failure_requeues_with_a_safe_stage_code():
    store, backend, runner = await setup()
    backend.errors[MediaStageName.TRANSCRIPTION] = RuntimeError("secret internal detail")

    outcome = await runner.run(OP)

    assert (outcome.state, outcome.error_code, outcome.retry) == (
        MediaOperationState.QUEUED, "transcription_failed", True,
    )
    saved = await stored(store)
    stage = saved.stages[MediaStageName.TRANSCRIPTION]
    assert stage.state == MediaStageState.FAILED and stage.error_code == "transcription_failed"
    assert "secret" not in saved.model_dump_json()


@pytest.mark.asyncio
async def test_retries_stop_when_attempts_are_exhausted():
    rec = record()
    rec.options = MediaOperationOptions(max_attempts=1)
    store, backend, runner = await setup(rec)
    backend.errors[MediaStageName.TRANSCRIPTION] = RuntimeError("flaky")

    outcome = await runner.run(OP)

    assert (outcome.state, outcome.retry) == (MediaOperationState.FAILED, False)
    assert outcome.error_code == "transcription_failed"


@pytest.mark.asyncio
async def test_a_non_retryable_processor_error_fails_terminally_with_an_allowed_code():
    store, backend, runner = await setup()
    backend.errors[MediaStageName.DIARIZATION] = MediaProcessingError("media_processor_unavailable", retryable=False)

    outcome = await runner.run(OP)

    assert (outcome.state, outcome.error_code, outcome.retry) == (
        MediaOperationState.FAILED, "media_processor_unavailable", False,
    )


@pytest.mark.asyncio
async def test_an_unlisted_processor_error_code_is_replaced_by_a_stage_code():
    store, backend, runner = await setup()
    backend.errors[MediaStageName.SUMMARY] = MediaProcessingError("raw provider message", retryable=False)

    outcome = await runner.run(OP)

    assert outcome.error_code == "summary_failed" and outcome.state == MediaOperationState.FAILED


@pytest.mark.asyncio
async def test_an_identity_failure_fails_the_stage():
    store, backend, runner = await setup()
    backend.identity_error = MediaProcessingError("media_processor_unavailable", retryable=False)

    outcome = await runner.run(OP)

    assert outcome.error_code == "media_processor_unavailable"
    assert (await stored(store)).stages[MediaStageName.TRANSCRIPTION].output is None


# --- run: output validation -------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "segments",
    [
        [(0, 20_000, "past the end")],              # beyond the probed duration
        [(0, 1500, "a"), (1000, 2000, "overlap")],  # overlapping segments
    ],
)
async def test_transcription_must_fit_inside_the_probed_media(segments):
    store, backend, runner = await setup()
    backend.outputs[MediaStageName.TRANSCRIPTION] = transcription(segments)

    outcome = await runner.run(OP)

    assert (outcome.state, outcome.error_code, outcome.retry) == (
        MediaOperationState.FAILED, "invalid_transcription", False,
    )
    assert backend.calls == [MediaStageName.TRANSCRIPTION]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "segments",
    [
        SEGMENTS[:1],                                          # missing a segment
        [(0, 1000, "first line"), (1000, 2000, "other")],      # text changed
        [(0, 1000, "first line"), (1000, 2100, "second line")],  # boundary changed
    ],
)
async def test_diarization_must_mirror_the_transcription(segments):
    store, backend, runner = await setup()
    backend.outputs[MediaStageName.DIARIZATION] = diarization(segments)

    outcome = await runner.run(OP)

    assert outcome.error_code == "invalid_diarization" and outcome.state == MediaOperationState.FAILED


@pytest.mark.asyncio
async def test_summary_points_must_cite_real_diarized_segments():
    store, backend, runner = await setup()
    backend.outputs[MediaStageName.SUMMARY] = summary(ranges=((1000, 2000), (5, 6)))

    outcome = await runner.run(OP)

    assert outcome.error_code == "ungrounded_summary" and outcome.retry is False


# --- run: cancellation and vanishing records --------------------------------

@pytest.mark.asyncio
async def test_a_cancel_seen_before_a_stage_stops_the_run():
    store, backend, runner = await setup()
    store.script[2] = canceled  # the first read inside the stage loop

    outcome = await runner.run(OP)

    assert outcome.state == MediaOperationState.CANCELED
    assert backend.calls == []


@pytest.mark.asyncio
async def test_a_cancel_arriving_during_a_stage_discards_its_progress():
    store, backend, runner = await setup()

    async def cancel_now():
        saved = await stored(store)
        saved.state = MediaOperationState.CANCELED
        await store.save(saved)

    backend.during[MediaStageName.TRANSCRIPTION] = cancel_now

    outcome = await runner.run(OP)

    assert outcome.state == MediaOperationState.CANCELED
    saved = await stored(store)
    assert saved.stages[MediaStageName.TRANSCRIPTION].state == MediaStageState.CANCELED
    assert saved.stages[MediaStageName.TRANSCRIPTION].output is None
    assert backend.calls == [MediaStageName.TRANSCRIPTION]


@pytest.mark.asyncio
async def test_a_cancel_arriving_while_a_stage_fails_wins_over_the_failure():
    store, backend, runner = await setup()

    async def cancel_then_fail():
        saved = await stored(store)
        saved.state = MediaOperationState.CANCELED
        await store.save(saved)

    backend.during[MediaStageName.TRANSCRIPTION] = cancel_then_fail
    backend.errors[MediaStageName.TRANSCRIPTION] = RuntimeError("late failure")

    outcome = await runner.run(OP)

    assert outcome.state == MediaOperationState.CANCELED and outcome.error_code is None


@pytest.mark.asyncio
async def test_a_cancel_seen_after_the_last_stage_prevents_the_result():
    store, backend, runner = await setup()
    store.script[8] = canceled  # the read after the three stages

    outcome = await runner.run(OP)

    assert outcome.state == MediaOperationState.CANCELED and outcome.result is None


@pytest.mark.asyncio
@pytest.mark.parametrize("read", [2, 3, 8])
async def test_a_record_that_disappears_mid_run_fails_with_not_found(read):
    store, backend, runner = await setup()
    store.script[read] = lambda found: None

    outcome = await runner.run(OP)

    assert (outcome.state, outcome.error_code) == (MediaOperationState.FAILED, "media_operation_not_found")


# --- result assembly --------------------------------------------------------

def finished_record():
    rec = record()
    for stage_name, artifact in (
        (MediaStageName.TRANSCRIPTION, transcription()),
        (MediaStageName.DIARIZATION, diarization()),
        (MediaStageName.SUMMARY, summary()),
    ):
        stage = rec.stages[stage_name]
        stage.state = MediaStageState.SUCCEEDED
        stage.output = artifact.model_dump(mode="json")
        stage.provenance = StageProvenance(provider="p", model="m", version="1", input_checksum_sha256=CHECKSUM)
    return rec


@pytest.mark.asyncio
async def test_a_record_with_every_stage_done_is_finalised_without_processors():
    store, backend, runner = await setup(finished_record())

    outcome = await runner.run(OP)

    assert outcome.state == MediaOperationState.SUCCEEDED and backend.calls == []


@pytest.mark.asyncio
async def test_missing_provenance_blocks_the_result():
    rec = finished_record()
    rec.stages[MediaStageName.SUMMARY].provenance = None
    store, backend, runner = await setup(rec)

    outcome = await runner.run(OP)

    assert (outcome.state, outcome.error_code, outcome.retry) == (
        MediaOperationState.FAILED, "media_provenance_incomplete", False,
    )


@pytest.mark.asyncio
async def test_a_succeeded_summary_stage_without_output_blocks_the_result():
    rec = finished_record()
    rec.stages[MediaStageName.SUMMARY].output = None
    store, backend, runner = await setup(rec)

    outcome = await runner.run(OP)

    assert outcome.error_code == "ungrounded_summary" and outcome.state == MediaOperationState.FAILED


@pytest.mark.parametrize(
    "reader, code",
    [
        (MediaOperationRunner._transcription, "invalid_transcription"),
        (MediaOperationRunner._diarization, "invalid_diarization"),
    ],
)
def test_reading_an_earlier_stage_output_that_is_absent_is_an_error(reader, code):
    with pytest.raises(MediaProcessingError) as caught:
        reader(record())

    assert (caught.value.code, caught.value.retryable) == (code, False)


@pytest.mark.asyncio
async def test_a_stage_that_depends_on_a_missing_output_fails_instead_of_running():
    rec = record()
    store, backend, runner = await setup(rec)
    rec.stages[MediaStageName.TRANSCRIPTION].state = MediaStageState.SUCCEEDED  # but no output stored
    await store.save(rec)

    outcome = await runner.run(OP)

    assert outcome.error_code == "invalid_transcription"
    assert outcome.state == MediaOperationState.FAILED
    assert backend.calls == []
