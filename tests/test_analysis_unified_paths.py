"""Covers the `/analyze` orchestration in `src/endpoints/analysis.py`.

`unified_analysis` has two shapes: a per-item path (one post) and a batched path
(two or more posts) that shares embedding and classification passes. Each stage
(language detection, translation, embedding, editorial scoring, taxonomy tagging)
must degrade independently: one failing stage may not discard what other stages
produced. Collaborators are fakes; the endpoint helpers they call are replaced
at the module boundary.
"""

import asyncio
import logging
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.endpoints import analysis
from src.endpoints.analysis import EDITORIAL_PRIORITY_LABELS, unified_analysis
from src.models import (
    ClassificationResponse,
    LanguageDetectionResponse,
    TranslationResponse,
    UnifiedAnalysisItemRequest,
    UnifiedAnalysisRequest,
)

TAXONOMY = ["sports", "politics"]


def item(post_id, content, lang=None):
    return UnifiedAnalysisItemRequest(
        id=post_id,
        source="rss",
        uri=f"https://example.com/{post_id}",
        content=content,
        createdAt="2026-07-17T00:00:00Z",
        hash=f"hash-{post_id}",
        lang=lang,
    )


def request(*items, **options):
    return UnifiedAnalysisRequest(items=list(items), **options)


class FakeClassifier:
    """Editorial label sets answer from `editorial`, taxonomy sets from `taxonomy`."""

    def __init__(self, editorial=None, taxonomy=None, error=None):
        self.editorial = editorial or {}
        self.taxonomy = taxonomy or {}
        self.error = error
        self.calls = []

    def _answer(self, text, labels):
        if labels == EDITORIAL_PRIORITY_LABELS:
            top = self.editorial.get(text, "routine publishable update")
            return {"labels": [top] + [l for l in labels if l != top], "scores": [0.9, 0.05, 0.02, 0.02, 0.01]}
        top = self.taxonomy.get(text, labels[0])
        return {"labels": [top] + [l for l in labels if l != top], "scores": [0.8, 0.1][: len(labels)]}

    def __call__(self, text, candidate_labels, **kwargs):
        self.calls.append((text, list(candidate_labels), kwargs))
        if self.error is not None:
            raise self.error
        if isinstance(text, list):
            return [self._answer(t, candidate_labels) for t in text]
        return self._answer(text, candidate_labels)


class FakeEmbedder:
    def __init__(self, error=None, delay=0.0):
        self.error = error
        self.delay = delay
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return np.array([[float(len(t)), 1.0, 2.0] for t in texts], dtype=np.float32)


def make_state(**overrides):
    config = {
        "TIMEOUT": 5,
        "EMBEDDING_TIMEOUT": 5,
        "MIN_SCORE": 0.30,
        "MIN_MARGIN": 0.10,
        "DEFAULT_CLASSIFICATION_LABELS": TAXONOMY,
        "ANALYZE_SLOW_MS": 10_000.0,
    }
    config.update(overrides.pop("config", {}))
    state = SimpleNamespace(
        fasttext_model=None,
        translator=None,
        embedder=None,
        classifier=None,
        classification_lock=asyncio.Lock(),
        editorial_lock=asyncio.Lock(),
        embedding_lock=asyncio.Lock(),
        classification_pool=None,
        embedding_pool=None,
        config=config,
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def http(state):
    return SimpleNamespace(state=SimpleNamespace(app_state=state))


@pytest.fixture
def stages(monkeypatch):
    """Replace the three endpoint helpers and record how they were called."""
    log = SimpleNamespace(detect=[], translate=[], classify=[], detect_error=None,
                          translate_error=None, classify_error=None)

    async def detect(req, _http):
        log.detect.append(req.text)
        if log.detect_error and log.detect_error in req.text:
            raise RuntimeError("detector down")
        language = "es" if "hola" in req.text else "en"
        return LanguageDetectionResponse(languages=[language], probabilities=[0.99])

    async def translate(req, _http):
        log.translate.append((req.text, req.source_language))
        if log.translate_error and log.translate_error in req.text:
            raise RuntimeError("translator down")
        return TranslationResponse(
            original_text=req.text, translated_text=f"EN: {req.text}",
            source_language=req.source_language, target_language="eng",
        )

    async def classify(req, _http):
        log.classify.append((req.text, req.labels))
        if log.classify_error:
            raise RuntimeError("taxonomy down")
        return ClassificationResponse(label="sports", score=0.8, full_result={"labels": ["sports"], "scores": [0.8]})

    monkeypatch.setattr(analysis, "detect_language", detect)
    monkeypatch.setattr(analysis, "translate_text", translate)
    monkeypatch.setattr(analysis, "classify_content", classify)
    return log


# --- batched path -----------------------------------------------------------

@pytest.mark.asyncio
async def test_batched_upstream_language_is_used_without_calling_the_detector(stages):
    state = make_state(fasttext_model=object())
    req = request(item("a", "plain text", lang={"language": "fr"}), item("b", "other", lang="de"),
                  translate_to_english=False)

    response = await unified_analysis(req, http(state))

    assert stages.detect == []
    assert [r.language_detection.languages for r in response.results] == [["fr"], ["de"]]
    assert all(r.language_detection.probabilities == [1.0] for r in response.results)


@pytest.mark.asyncio
async def test_batched_detection_failure_leaves_that_item_without_a_language(stages):
    stages.detect_error = "broken"
    state = make_state(fasttext_model=object())
    req = request(item("a", "hola amigos"), item("b", "broken text"), translate_to_english=False)

    response = await unified_analysis(req, http(state))

    first, second = response.results
    assert first.language_detection.languages == ["es"]
    assert second.language_detection is None


@pytest.mark.asyncio
async def test_batched_detection_is_skipped_when_disabled_or_unavailable(stages):
    req = request(item("a", "hola"), item("b", "hola"), detect_language=False, translate_to_english=False)
    await unified_analysis(req, http(make_state(fasttext_model=object())))
    await unified_analysis(request(item("a", "hola"), item("b", "hola")), http(make_state()))

    assert stages.detect == []


@pytest.mark.asyncio
async def test_batched_translation_replaces_content_for_non_english_items_only(stages):
    state = make_state(fasttext_model=object(), translator=object())
    req = request(item("a", "hola mundo"), item("b", "english text"))

    response = await unified_analysis(req, http(state))

    spanish, english = response.results
    assert stages.translate == [("hola mundo", "es")]
    assert spanish.content == "EN: hola mundo" and spanish.translation.source_language == "es"
    assert english.content == "english text"
    assert english.translation.translated_text == "english text"
    assert english.translation.target_language == "eng"


@pytest.mark.asyncio
async def test_batched_translation_failure_keeps_the_original_content(stages):
    stages.translate_error = "hola"
    state = make_state(fasttext_model=object(), translator=object())

    response = await unified_analysis(request(item("a", "hola mundo"), item("b", "hola otra")), http(state))

    assert all(r.translation is None for r in response.results)
    assert [r.content for r in response.results] == ["hola mundo", "hola otra"]


@pytest.mark.asyncio
async def test_batched_without_a_translator_records_a_passthrough_translation(stages):
    state = make_state(fasttext_model=object())

    response = await unified_analysis(request(item("a", "hola mundo"), item("b", "hola otra")), http(state))

    assert stages.translate == []
    assert response.results[0].translation.translated_text == "hola mundo"
    assert response.results[0].translation.source_language == "es"


@pytest.mark.asyncio
async def test_batched_embeddings_are_computed_on_the_final_content_in_one_pass(stages):
    embedder = FakeEmbedder()
    state = make_state(fasttext_model=object(), translator=object(), embedder=embedder)

    response = await unified_analysis(request(item("a", "hola"), item("b", "english")), http(state))

    assert embedder.calls == [["EN: hola", "english"]]
    assert response.results[0].embedding == [8.0, 1.0, 2.0]
    assert response.results[1].embedding == [7.0, 1.0, 2.0]


@pytest.mark.asyncio
async def test_batched_embedding_failure_does_not_discard_other_stages(stages, caplog, monkeypatch):
    monkeypatch.setattr(analysis, "_embed_sync", lambda *a: (_ for _ in ()).throw(RuntimeError("oom")))
    state = make_state(embedder=FakeEmbedder(), classifier=FakeClassifier())

    with caplog.at_level(logging.ERROR, logger="mistify"):
        response = await unified_analysis(request(item("a", "x one"), item("b", "y two"), translate_to_english=False), http(state))

    assert all(r.embedding is None for r in response.results)
    assert all(r.newsworthiness is not None for r in response.results)
    assert "Batched embedding generation failed" in caplog.text


@pytest.mark.asyncio
async def test_batched_editorial_scores_blend_classifier_and_urgency_signals(stages):
    classifier = FakeClassifier(editorial={"quiet news": "not newsworthy", "gunman opens fire": "routine publishable update"})
    state = make_state(classifier=classifier)

    response = await unified_analysis(
        request(item("a", "quiet news"), item("b", "gunman opens fire"), translate_to_english=False, classify_content=False),
        http(state),
    )

    quiet, urgent = response.results
    assert quiet.newsworthiness < urgent.newsworthiness
    assert urgent.urgency >= 4.0  # active_violence signal outweighs the routine label
    assert quiet.urgency < urgent.urgency
    assert classifier.calls[0][1] == EDITORIAL_PRIORITY_LABELS
    assert classifier.calls[0][2]["batch_size"] == 8


@pytest.mark.asyncio
async def test_batched_editorial_failure_zeroes_scores_for_every_item(stages):
    state = make_state(classifier=FakeClassifier(error=RuntimeError("model crashed")))

    response = await unified_analysis(request(item("a", "one"), item("b", "two"), translate_to_english=False), http(state))

    assert [(r.newsworthiness, r.urgency) for r in response.results] == [(0.0, 0.0), (0.0, 0.0)]
    assert all(r.content_classification is None for r in response.results)


@pytest.mark.asyncio
async def test_batched_taxonomy_runs_only_for_items_worth_tagging(stages):
    classifier = FakeClassifier(
        editorial={"junk post": "not newsworthy", "minor item": "niche or low-priority item"},
        taxonomy={"big story": "politics"},
    )
    state = make_state(classifier=classifier)

    response = await unified_analysis(
        request(item("a", "junk post"), item("b", "big story"), item("c", "minor item"), translate_to_english=False),
        http(state),
    )

    junk, big, minor = response.results
    assert junk.content_classification is None and minor.content_classification is None
    assert big.content_classification.label == "politics"
    taxonomy_call = classifier.calls[-1]
    assert taxonomy_call[0] == ["big story"] and taxonomy_call[1] == TAXONOMY


@pytest.mark.asyncio
async def test_batched_taxonomy_marks_a_weak_prediction_uncertain(stages):
    classifier = FakeClassifier()
    state = make_state(classifier=classifier, config={"MIN_SCORE": 0.95})

    response = await unified_analysis(request(item("a", "one"), item("b", "two"), translate_to_english=False), http(state))

    assert [r.content_classification.label for r in response.results] == ["uncertain", "uncertain"]


@pytest.mark.asyncio
async def test_batched_single_label_taxonomy_gets_an_other_label(stages):
    classifier = FakeClassifier()
    state = make_state(classifier=classifier)

    await unified_analysis(
        request(item("a", "one"), item("b", "two"), translate_to_english=False, classification_labels=["sports"]),
        http(state),
    )

    assert classifier.calls[-1][1] == ["sports", "other"]


@pytest.mark.asyncio
async def test_batched_taxonomy_is_skipped_without_labels_or_when_disabled(stages):
    for options, config in (({}, {"DEFAULT_CLASSIFICATION_LABELS": []}), ({"classify_content": False}, {})):
        classifier = FakeClassifier()
        state = make_state(classifier=classifier, config=config)
        await unified_analysis(request(item("a", "one"), item("b", "two"), translate_to_english=False, **options), http(state))
        assert [c[1] for c in classifier.calls] == [EDITORIAL_PRIORITY_LABELS]


@pytest.mark.asyncio
async def test_batched_taxonomy_failure_keeps_editorial_scores(stages, monkeypatch):
    state = make_state(classifier=FakeClassifier())

    def broken(_result, _state):
        raise RuntimeError("bad taxonomy payload")

    monkeypatch.setattr(analysis, "_classification_response", broken)

    response = await unified_analysis(request(item("a", "one"), item("b", "two"), translate_to_english=False), http(state))

    assert all(r.content_classification is None for r in response.results)
    assert all(r.newsworthiness is not None for r in response.results)


# --- per-item path ----------------------------------------------------------

@pytest.mark.asyncio
async def test_single_item_runs_every_stage_and_reports_timings(stages):
    classifier = FakeClassifier(editorial={"EN: hola mundo": "major breaking story"})
    embedder = FakeEmbedder()
    state = make_state(fasttext_model=object(), translator=object(), embedder=embedder, classifier=classifier)

    response = await unified_analysis(request(item("a", "hola mundo"), include_timings=True), http(state))

    result = response.results[0]
    assert result.language_detection.languages == ["es"]
    assert result.content == "EN: hola mundo" and result.translation.source_language == "es"
    assert result.embedding == [14.0, 1.0, 2.0]
    assert result.newsworthiness > 8 and result.urgency > 5
    assert result.content_classification.label == "sports"
    assert stages.classify == [("EN: hola mundo", None)]
    timings = response.timings
    assert timings.item_count == 1 and timings.total_ms >= 0
    assert result.timings.total_ms >= 0
    assert result.timings.language_detection_ms is not None
    assert result.timings.translation_ms is not None
    assert result.timings.content_classification_ms is not None
    assert result.timings.newsworthiness_ms == result.timings.urgency_ms


@pytest.mark.asyncio
async def test_single_item_without_models_still_answers_and_warns(stages, caplog):
    with caplog.at_level(logging.WARNING, logger="mistify"):
        response = await unified_analysis(request(item("a", "hello"), translate_to_english=False), http(make_state()))

    result = response.results[0]
    assert result.embedding is None and result.newsworthiness is None
    assert response.timings is None
    assert "embedder is not loaded" in caplog.text
    assert "classifier is not loaded" in caplog.text


@pytest.mark.asyncio
async def test_single_item_stage_failures_are_independent(stages, monkeypatch):
    stages.detect_error = "hola"
    stages.classify_error = True
    monkeypatch.setattr(analysis, "_embed_sync", lambda *a: (_ for _ in ()).throw(ValueError("bad shape")))
    classifier = FakeClassifier()
    embedder = FakeEmbedder()
    state = make_state(fasttext_model=object(), embedder=embedder, classifier=classifier)

    response = await unified_analysis(request(item("a", "hola mundo"), include_timings=True), http(state))

    result = response.results[0]
    assert result.language_detection is None
    assert result.embedding is None
    assert result.newsworthiness is not None
    assert result.content_classification is None
    assert result.timings.language_detection_ms is not None  # recorded even though detection failed


@pytest.mark.asyncio
async def test_single_item_editorial_failure_scores_zero_and_skips_taxonomy(stages):
    state = make_state(classifier=FakeClassifier(error=RuntimeError("boom")))

    response = await unified_analysis(request(item("a", "hello"), translate_to_english=False), http(state))

    result = response.results[0]
    assert (result.newsworthiness, result.urgency) == (0.0, 0.0)
    assert result.content_classification is None


@pytest.mark.asyncio
async def test_single_item_translation_failure_keeps_content(stages):
    stages.translate_error = "hola"
    state = make_state(fasttext_model=object(), translator=object())

    response = await unified_analysis(request(item("a", "hola mundo"), include_timings=True), http(state))

    result = response.results[0]
    assert result.content == "hola mundo" and result.translation is None
    assert result.timings.translation_ms is not None


@pytest.mark.asyncio
async def test_single_item_english_is_recorded_without_translating(stages):
    state = make_state(fasttext_model=object(), translator=object())

    response = await unified_analysis(request(item("a", "english text")), http(state))

    assert stages.translate == []
    assert response.results[0].translation.translated_text == "english text"


@pytest.mark.asyncio
async def test_single_item_embedding_timeout_is_tolerated(stages, caplog):
    state = make_state(embedder=FakeEmbedder(delay=0.3), config={"EMBEDDING_TIMEOUT": 0.01})

    with caplog.at_level(logging.ERROR, logger="mistify"):
        response = await unified_analysis(request(item("a", "hello"), translate_to_english=False), http(state))

    assert response.results[0].embedding is None
    assert "timed out" in caplog.text


@pytest.mark.asyncio
async def test_low_value_single_item_skips_taxonomy(stages):
    classifier = FakeClassifier(editorial={"nothing": "not newsworthy"})
    state = make_state(classifier=classifier)

    response = await unified_analysis(request(item("a", "nothing"), translate_to_english=False), http(state))

    assert stages.classify == [] and response.results[0].content_classification is None


@pytest.mark.asyncio
async def test_slow_batches_are_logged_with_stage_totals(stages, caplog):
    state = make_state(config={"ANALYZE_SLOW_MS": 0.0})

    with caplog.at_level(logging.WARNING, logger="mistify"):
        await unified_analysis(request(item("a", "hello"), translate_to_english=False), http(state))

    assert "Slow /analyze batch: items=1" in caplog.text


@pytest.mark.asyncio
async def test_an_encoder_error_surfaces_as_a_zero_vector_not_a_missing_embedding(stages):
    """`_embed_sync` swallows encoder errors, so callers receive zero embeddings."""
    state = make_state(embedder=FakeEmbedder(error=RuntimeError("oom")))

    response = await unified_analysis(request(item("a", "hello"), translate_to_english=False), http(state))

    assert response.results[0].embedding == [0.0] * 384


@pytest.mark.asyncio
async def test_single_item_upstream_language_is_exposed_without_detection(stages):
    state = make_state(fasttext_model=object())

    response = await unified_analysis(request(item("a", "hola", lang="pt"), translate_to_english=False), http(state))

    assert stages.detect == []
    assert response.results[0].language_detection.languages == ["pt"]


@pytest.mark.asyncio
async def test_single_item_detector_with_no_candidates_leaves_translation_off(stages, monkeypatch):
    async def empty(_req, _http):
        return LanguageDetectionResponse(languages=[], probabilities=[])

    monkeypatch.setattr(analysis, "detect_language", empty)
    state = make_state(fasttext_model=object(), translator=object())

    response = await unified_analysis(request(item("a", "???")), http(state))

    assert response.results[0].language_detection.languages == []
    assert response.results[0].translation is None


@pytest.mark.asyncio
async def test_single_item_empty_embedding_result_leaves_the_embedding_unset(stages, monkeypatch):
    monkeypatch.setattr(analysis, "_embed_sync", lambda *a: np.empty((0, 3), dtype=np.float32))
    state = make_state(embedder=FakeEmbedder())

    response = await unified_analysis(request(item("a", "hello"), translate_to_english=False), http(state))

    assert response.results[0].embedding is None
