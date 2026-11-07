"""Remaining reachable branches in translation, clustering and the worker.

Each test pins a behaviour that earlier suites left implicit: GPU cache hygiene
around translation attempts, detection being skipped for blank text, a candidate
the embedder returned nothing for, domain filtering with unrelated categories,
and the worker's batching and media-outcome rules.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.endpoints import translation as translation_module
from src.endpoints.clustering import _filter_candidate_posts
from src.endpoints.translation import translate_text
from src.helpers import async_wrappers as wrappers
from src.helpers.async_wrappers import _translate_sync
from src.helpers.clustering import should_cluster_by_domain
from src.models import PostData, TranslationRequest
from src.operations.media_models import MediaOperationState, MediaRunOutcome
from src.operations.models import GrpcCallback, OperationEnvelope, QueuedOperation
from src.operations.worker import ANALYSIS_BATCH_MAX_ITEMS, OperationWorker


# --- GPU cache hygiene around translation -----------------------------------

@pytest.fixture
def cache_flushes(monkeypatch):
    flushes = []
    monkeypatch.setattr(wrappers.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(wrappers.torch.cuda, "empty_cache", lambda: flushes.append(True))
    return flushes


def test_cache_is_flushed_before_and_after_a_successful_attempt(cache_flushes):
    result = _translate_sync(lambda text, **_k: [{"translation_text": "ok"}], "hola mundo", "es")

    assert result == [{"translation_text": "ok"}]
    assert len(cache_flushes) == 2


def test_cache_is_flushed_after_a_failed_attempt_and_the_next_one_still_runs(cache_flushes):
    calls = []

    def flaky(text, **_k):
        calls.append(text)
        if len(calls) == 1:
            raise RuntimeError("out of memory")
        return [{"translation_text": "ok"}]

    long_text = " ".join(f"word{i}" for i in range(900))

    assert _translate_sync(flaky, long_text, "es") == [{"translation_text": "ok"}]
    assert len(calls) >= 2 and len(cache_flushes) >= 3


def test_the_last_resort_flushes_the_cache_on_success(cache_flushes):
    outcomes = iter([RuntimeError("a")] * 20)
    calls = []

    def translator(text, **kwargs):
        calls.append(kwargs)
        if "max_length" in kwargs and kwargs == {"max_length": 100}:
            return [{"translation_text": "last resort"}]
        raise next(outcomes)

    assert _translate_sync(translator, "short text here", "es") == [{"translation_text": "last resort"}]
    assert calls[-1] == {"max_length": 100}
    assert cache_flushes


def test_when_every_strategy_fails_the_result_is_none_and_the_cache_is_flushed(cache_flushes):
    def always_fails(*_a, **_k):
        raise RuntimeError("model gone")

    assert _translate_sync(always_fails, "short text here", "es") is None
    assert cache_flushes


# --- /translate: detection preconditions ------------------------------------

class RankedFastText:
    def __init__(self, ranked):
        self.ranked, self.calls = ranked, 0

    def predict(self, _text, k=1):
        self.calls += 1
        ranked = self.ranked[:k]
        return [f"__label__{lang}" for lang, _ in ranked], [prob for _, prob in ranked]


def translate_state(model):
    return SimpleNamespace(
        fasttext_model=model, translator=object(), translator_model_name=None,
        translation_lock=asyncio.Lock(), translation_pool=ThreadPoolExecutor(max_workers=1),
        config={"TIMEOUT": 30},
    )


@pytest.mark.asyncio
async def test_blank_text_is_not_sent_to_the_detector(monkeypatch):
    model = RankedFastText([("es", 0.99)])
    monkeypatch.setattr(translation_module, "_translate_sync", lambda *_a, **_k: [{"translation_text": "x"}])
    request = SimpleNamespace(state=SimpleNamespace(app_state=translate_state(model)))

    response = await translate_text(TranslationRequest(text="  \n\t ", source_language=None), request)

    assert model.calls == 0
    assert response.source_language == "en"


@pytest.mark.asyncio
async def test_low_confidence_unsupported_candidates_fall_back_to_the_default(monkeypatch):
    model = RankedFastText([("zzz", 0.2), ("qqq", 0.1)])
    monkeypatch.setattr(translation_module, "_translate_sync", lambda *_a, **_k: [{"translation_text": "x"}])
    request = SimpleNamespace(state=SimpleNamespace(app_state=translate_state(model)))

    response = await translate_text(TranslationRequest(text="unclear text", source_language=None), request)

    assert model.calls == 1
    assert response.source_language == "en"


# --- candidate pre-filter ---------------------------------------------------

def post(post_id, content, embeddings=None):
    return PostData(
        id=post_id, content=content, source="rss", createdAt="2026-08-29T12:00:00Z",
        hash=f"hash-{post_id}", uuid=f"0000000-0000-0000-0000-{post_id:012d}", embeddings=embeddings,
    )


def test_a_candidate_the_embedder_returned_nothing_for_is_dropped():
    class ShortEmbedder:
        def encode(self, texts, **_k):
            return SimpleNamespace(tolist=lambda: [[1.0, 0.0]])  # one row for two texts

    main = post(1, "main story", embeddings=[1.0, 0.0])

    kept = _filter_candidate_posts(main, [post(2, "first"), post(3, "second")], ShortEmbedder(), min_similarity=0.5, max_age_hours=None)

    assert [c.id for c in kept] == [2]


# --- domain filtering -------------------------------------------------------

def test_unrelated_categories_do_not_block_clustering():
    config = {"domain_filtering": True}

    assert should_cluster_by_domain(["sports"], ["weather"], config) is True
    assert should_cluster_by_domain(["weather"], ["sports"], config) is True
    assert should_cluster_by_domain(["sports"], ["Cine y Series"], config) is False
    assert should_cluster_by_domain(["sports"], ["Copa Sudamericana"], config) is True
    assert should_cluster_by_domain(["politics"], ["conflict"], config) is True


# --- worker batching and media outcomes -------------------------------------

def analysis(items, **kwargs):
    return QueuedOperation(
        envelope=OperationEnvelope(operation_type="analyze_posts", payload={"items": items}, **kwargs),
        receipts=[f"r{len(items)}"],
    )


@pytest.mark.asyncio
async def test_a_full_batch_or_malformed_items_are_not_extended():
    queue = SimpleNamespace(dequeue_nowait=AsyncMock())
    worker = OperationWorker(queue, SimpleNamespace())
    malformed = QueuedOperation(envelope=OperationEnvelope(operation_type="analyze_posts", payload={"items": "nope"}))
    full = analysis([{"post": {"id": i}} for i in range(1000)])

    assert await worker._coalesce_analysis(malformed) is malformed
    assert await worker._coalesce_analysis(full) is full
    queue.dequeue_nowait.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_incompatible_neighbour_is_returned_to_the_front_of_the_queue():
    first = analysis([{"post": {"id": 1}}], grpc_callback=GrpcCallback(target="a:1"))
    other = analysis([{"post": {"id": 2}}], grpc_callback=GrpcCallback(target="b:2"))
    queue = SimpleNamespace(dequeue_nowait=AsyncMock(side_effect=[other]), requeue_next=AsyncMock())
    worker = OperationWorker(queue, SimpleNamespace())

    combined = await worker._coalesce_analysis(first)

    queue.requeue_next.assert_awaited_once_with(other)
    assert [i["post"]["id"] for i in combined.envelope.payload["items"]] == [1]
    assert combined.receipts == ["r1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state, status",
    [(MediaOperationState.FAILED, "failed"), (MediaOperationState.CANCELED, "canceled")],
)
async def test_terminal_media_outcomes_are_delivered_with_their_status(state, status):
    runner = SimpleNamespace(run=AsyncMock(return_value=MediaRunOutcome(state=state, error_code="x")))
    queue = SimpleNamespace(acknowledge=AsyncMock(), requeue_later=AsyncMock())
    worker = OperationWorker(queue, SimpleNamespace(media_operation_runner=runner))
    worker._deliver_callback_safely = AsyncMock(return_value=True)
    queued = QueuedOperation(
        envelope=OperationEnvelope(operation_type="process_media", payload={}), receipts=["m"]
    )

    await worker._process_media(queued)

    delivered = worker._deliver_callback_safely.await_args
    assert delivered.args[1] == status
    queue.requeue_later.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_in_executor_helper_runs_the_function_on_the_given_executor():
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        result = await wrappers.run_in_executor(asyncio.get_running_loop(), executor, lambda a, b: a + b, 2, 3)
    finally:
        executor.shutdown()

    assert result == 5


@pytest.mark.asyncio
async def test_coalescing_stops_as_soon_as_the_batch_is_exactly_full():
    first = analysis([{"post": {"id": i}} for i in range(ANALYSIS_BATCH_MAX_ITEMS - 1)])
    last = analysis([{"post": {"id": "last"}}])
    queue = SimpleNamespace(dequeue_nowait=AsyncMock(side_effect=[last]), requeue_next=AsyncMock())
    worker = OperationWorker(queue, SimpleNamespace())

    combined = await worker._coalesce_analysis(first)

    assert len(combined.envelope.payload["items"]) == ANALYSIS_BATCH_MAX_ITEMS
    assert queue.dequeue_nowait.await_count == 1
    queue.requeue_next.assert_not_awaited()


def test_candidates_with_empty_identifiers_are_still_considered_once():
    class Embedder:
        def encode(self, texts, **_k):
            return SimpleNamespace(tolist=lambda: [[1.0, 0.0]] * len(texts))

    main = post(1, "main story", embeddings=[1.0, 0.0])
    anonymous = PostData(
        id=0, content="anonymous", source="rss", createdAt="2026-08-29T12:00:00Z", hash="", uuid="u",
    )
    twin = PostData(
        id=0, content="anonymous", source="rss", createdAt="2026-08-29T12:00:00Z", hash="", uuid="u2",
    )

    kept = _filter_candidate_posts(main, [anonymous, twin], Embedder(), min_similarity=0.5, max_age_hours=None)

    assert [c.content for c in kept] == ["anonymous"]  # the duplicate is caught by content instead
