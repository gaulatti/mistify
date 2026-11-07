"""Covers the remaining branches of scout query generation and runtime metrics.

Scout queries degrade step by step (language detection, translation, entity
extraction can each fail and the query still comes out). Runtime metrics must
reflect GPU and model state without ever failing a scrape.
"""

import asyncio
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from src import metrics
from src.helpers import scout_queries
from src.helpers.scout_queries import _keyword_query, generate_scout_queries, rank_scout_candidates
from src.models import LanguageDetectionResponse, TranslationResponse


# --- keyword queries --------------------------------------------------------

class FakeNlp:
    def __init__(self, entities=(), error=None):
        self.entities, self.error = entities, error

    def __call__(self, text):
        if self.error:
            raise self.error
        return SimpleNamespace(ents=[SimpleNamespace(text=e) for e in self.entities])


def test_entities_lead_the_query_and_their_words_are_not_repeated():
    query = _keyword_query("Flooding hits Buenos Aires as Milei visits", FakeNlp(["Buenos Aires", "  ", "Milei"]))

    assert query.split()[:3] == ["Buenos", "Aires", "Milei"]
    assert query.split().count("Milei") == 1
    assert "Flooding" in query


def test_a_failing_entity_extractor_is_logged_and_tokens_are_used(caplog):
    with caplog.at_level(logging.WARNING, logger="mistify"):
        query = _keyword_query("Storm batters coast", FakeNlp(error=RuntimeError("spacy down")))

    assert query == "Storm batters coast"
    assert "Scout entity extraction failed" in caplog.text


def test_repeated_words_and_stopwords_are_dropped_and_the_length_is_capped():
    assert _keyword_query("Storm storm STORM the coast") == "Storm coast"
    assert len(_keyword_query(" ".join(f"word{i}x" for i in range(30))).split()) == 10
    assert _keyword_query("of to it") == "of to it"  # nothing usable: the original text is kept


# --- query generation -------------------------------------------------------

def state(**overrides):
    base = dict(fasttext_model=None, translator=None, nlp=None)
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_language_is_detected_when_not_supplied(monkeypatch):
    seen = []

    async def detect(req, _request):
        seen.append(req.text)
        return LanguageDetectionResponse(languages=["fr"], probabilities=[0.9])

    monkeypatch.setattr(scout_queries, "detect_language", detect)

    _queries, _title, language = await generate_scout_queries(state(fasttext_model=object()), "Titre du jour")

    assert seen == ["Titre du jour"] and language == "fr"


@pytest.mark.asyncio
async def test_supplied_language_skips_detection(monkeypatch):
    async def detect(*_a):
        raise AssertionError("must not detect")

    monkeypatch.setattr(scout_queries, "detect_language", detect)

    _q, _t, language = await generate_scout_queries(state(fasttext_model=object()), "Titulo", " es ")

    assert language == "es"


@pytest.mark.asyncio
async def test_detection_failures_and_empty_results_leave_the_language_blank(monkeypatch, caplog):
    async def empty(*_a):
        return LanguageDetectionResponse(languages=[], probabilities=[])

    async def broken(*_a):
        raise RuntimeError("detector down")

    monkeypatch.setattr(scout_queries, "detect_language", empty)
    assert (await generate_scout_queries(state(fasttext_model=object()), "Title"))[2] == ""

    monkeypatch.setattr(scout_queries, "detect_language", broken)
    with caplog.at_level(logging.WARNING, logger="mistify"):
        assert (await generate_scout_queries(state(fasttext_model=object()), "Title"))[2] == ""
    assert "Scout language detection failed" in caplog.text


@pytest.mark.asyncio
async def test_a_translated_title_adds_a_broader_second_query(monkeypatch):
    async def translate(req, _request):
        assert (req.source_language, req.target_language) == ("es", "eng")
        return TranslationResponse(
            original_text=req.text,
            translated_text="Flooding devastates northern provinces leaving thousands homeless overnight",
            source_language="es", target_language="eng",
        )

    monkeypatch.setattr(scout_queries, "translate_text", translate)

    queries, translated, language = await generate_scout_queries(
        state(translator=object()), "Inundaciones devastan provincias", "es", max_queries=3
    )

    assert translated.startswith("Flooding devastates") and language == "es"
    assert len(queries) == 2
    assert len(queries[0].split()) > len(queries[1].split()) == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("max_queries, expected", [(0, 1), (1, 1), (2, 2), (9, 2)])
async def test_the_query_count_is_clamped(monkeypatch, max_queries, expected):
    async def translate(req, _request):
        return TranslationResponse(
            original_text=req.text, translated_text="Flooding devastates northern provinces leaving many homeless",
            source_language="es", target_language="eng",
        )

    monkeypatch.setattr(scout_queries, "translate_text", translate)

    queries, _t, _l = await generate_scout_queries(state(translator=object()), "Titular", "es", max_queries)

    assert len(queries) == expected


@pytest.mark.asyncio
async def test_translation_failure_or_empty_output_keeps_the_original_title(monkeypatch, caplog):
    async def broken(*_a):
        raise RuntimeError("translator down")

    async def empty(req, _request):
        return TranslationResponse(original_text=req.text, translated_text="", source_language="es", target_language="eng")

    monkeypatch.setattr(scout_queries, "translate_text", broken)
    with caplog.at_level(logging.WARNING, logger="mistify"):
        queries, translated, _l = await generate_scout_queries(state(translator=object()), "Titular fuerte", "es")
    assert translated == "Titular fuerte" and len(queries) == 1
    assert "Scout translation failed" in caplog.text

    monkeypatch.setattr(scout_queries, "translate_text", empty)
    assert (await generate_scout_queries(state(translator=object()), "Titular fuerte", "es"))[1] == "Titular fuerte"


@pytest.mark.asyncio
async def test_english_titles_are_never_translated(monkeypatch):
    async def translate(*_a):
        raise AssertionError("must not translate")

    monkeypatch.setattr(scout_queries, "translate_text", translate)

    for language in ("en", "ENG"):
        await generate_scout_queries(state(translator=object()), "English title here", language)


# --- ranking guards ---------------------------------------------------------

def rank_state(embedder=object()):
    return SimpleNamespace(embedder=embedder, embedding_lock=asyncio.Lock(), embedding_pool=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seed, candidates, embedder",
    [("  ", [("a", "title")], object()), ("seed", [("a", "  ")], object()), ("seed", [], object()), ("seed", [("a", "t")], None)],
)
async def test_ranking_returns_nothing_when_there_is_nothing_to_compare(seed, candidates, embedder):
    assert await rank_scout_candidates(rank_state(embedder), seed, candidates) == []


@pytest.mark.asyncio
async def test_ranking_applies_default_threshold_and_caps_the_result(monkeypatch):
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.1], [0.3, 0.9]], dtype=np.float32)
    monkeypatch.setattr(scout_queries, "_embed_sync", lambda *_a: vectors)
    candidates = [("a", "one"), ("b", "two"), ("c", "three")]

    everything = await rank_scout_candidates(rank_state(), "seed", candidates, min_score=-5, max_candidates=0)
    thresholded = await rank_scout_candidates(rank_state(), "seed", candidates, min_score=0, max_candidates=10)
    capped = await rank_scout_candidates(rank_state(), "seed", candidates, min_score=0, max_candidates=1)

    assert [c for c, _ in everything] == ["a", "b", "c"]  # min_score clamps to -1, 0 candidates -> 15
    assert [c for c, _ in thresholded] == ["a", "b"]      # min_score 0 means the 0.55 default
    assert [c for c, _ in capped] == ["a"]


# --- runtime metrics --------------------------------------------------------

def gauge(g):
    return g._value.get()


def fake_cuda(monkeypatch, devices=2, smi=None, memory_error=None):
    monkeypatch.setattr(metrics.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(metrics.torch.cuda, "device_count", lambda: devices)

    def allocated(i):
        if memory_error:
            raise memory_error
        return 100 * (i + 1)

    monkeypatch.setattr(metrics.torch.cuda, "memory_allocated", allocated)
    monkeypatch.setattr(metrics.torch.cuda, "memory_reserved", lambda i: 1000 * (i + 1))

    def run(cmd, **_k):
        index = int(cmd[-1])
        outcome = (smi or {}).get(index)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(returncode=outcome[0], stdout=outcome[1])

    monkeypatch.setattr(metrics.subprocess, "run", run)


def test_gpu_memory_and_the_busiest_gpu_utilisation_are_reported(monkeypatch):
    fake_cuda(monkeypatch, smi={0: (0, "35\n"), 1: (0, "80\n")})

    metrics.update_runtime_metrics(None)

    assert gauge(metrics.GPU_MEMORY_ALLOCATED_BYTES) == 300
    assert gauge(metrics.GPU_MEMORY_RESERVED_BYTES) == 3000
    assert gauge(metrics.GPU_UTILIZATION_PERCENT) == 80.0
    assert gauge(metrics.TORCH_DEVICE_INFO.labels(device="cuda")) == 1.0


def test_unavailable_nvidia_smi_does_not_block_memory_metrics(monkeypatch):
    fake_cuda(monkeypatch, devices=1, smi={0: FileNotFoundError("no smi")})
    metrics.GPU_UTILIZATION_PERCENT.set(11.0)

    metrics.update_runtime_metrics(None)

    assert gauge(metrics.GPU_MEMORY_ALLOCATED_BYTES) == 100
    assert gauge(metrics.GPU_UTILIZATION_PERCENT) == 11.0  # unchanged: no reading


def test_a_failing_nvidia_smi_status_is_ignored(monkeypatch):
    fake_cuda(monkeypatch, devices=1, smi={0: (1, "")})
    metrics.GPU_UTILIZATION_PERCENT.set(12.0)

    metrics.update_runtime_metrics(None)

    assert gauge(metrics.GPU_UTILIZATION_PERCENT) == 12.0


def test_a_gpu_memory_failure_never_fails_the_scrape(monkeypatch):
    fake_cuda(monkeypatch, memory_error=RuntimeError("driver error"))
    metrics.GPU_MEMORY_ALLOCATED_BYTES.set(7.0)

    metrics.update_runtime_metrics(SimpleNamespace(embedder=object()))

    assert gauge(metrics.GPU_MEMORY_ALLOCATED_BYTES) == 7.0
    assert gauge(metrics.MODEL_AVAILABLE.labels(model="embedder")) == 1.0


def test_model_and_media_processor_availability_follow_app_state():
    runner = SimpleNamespace(available=True)
    app_state = SimpleNamespace(fasttext_model=object(), classifier=None, media_operation_runner=runner)

    metrics.update_runtime_metrics(app_state)

    assert gauge(metrics.MODEL_AVAILABLE.labels(model="fasttext")) == 1.0
    assert gauge(metrics.MODEL_AVAILABLE.labels(model="classifier")) == 0.0
    assert gauge(metrics.MODEL_AVAILABLE.labels(model="media_processor")) == 1.0
    runner.available = False
    metrics.update_runtime_metrics(app_state)
    assert gauge(metrics.MODEL_AVAILABLE.labels(model="media_processor")) == 0.0
    metrics.update_runtime_metrics(SimpleNamespace())
    assert gauge(metrics.MODEL_AVAILABLE.labels(model="media_processor")) == 0.0


# --- route labels and job outcomes ------------------------------------------

@pytest.mark.parametrize(
    "scope, label",
    [
        ({"route": SimpleNamespace(path="/items/{id}"), "path": "/items/7"}, "/items/{id}"),
        ({"route": SimpleNamespace(path=None), "path": "/metrics"}, "/metrics"),
        ({"route": SimpleNamespace(path=""), "path": "/health"}, "/health"),
        ({"path": "/metrics"}, "/metrics"),
        ({"path": "/anything/with/an/id/123"}, "unmatched"),
        ({}, "unmatched"),
    ],
)
def test_route_labels_are_templates_or_one_of_a_few_fixed_values(scope, label):
    assert metrics.route_label_from_request_scope(scope) == label


def job_count(operation, outcome):
    return metrics.OPERATION_JOBS_TOTAL.labels(
        operation=metrics.operation_label(operation), outcome=outcome
    )._value.get()


@pytest.mark.parametrize("outcome", ["retry", "canceled"])
def test_job_outcomes_set_by_the_caller_are_recorded(outcome):
    before = job_count("embed_text", outcome)

    with metrics.record_job("embed_text") as state:
        state["outcome"] = outcome

    assert job_count("embed_text", outcome) == before + 1


def test_an_unknown_job_outcome_is_recorded_as_an_error():
    before = job_count("embed_text", "error")

    with metrics.record_job("embed_text") as state:
        state["outcome"] = "private-free-form-outcome"

    assert job_count("embed_text", "error") == before + 1


def test_a_job_that_raises_is_recorded_as_an_error_and_reraised():
    before = job_count("embed_text", "error")

    with pytest.raises(ValueError):
        with metrics.record_job("embed_text"):
            raise ValueError("boom")

    assert job_count("embed_text", "error") == before + 1
