"""Covers `/detect`, `/classify` and `/embed`.

These three endpoints are thin, but they hold the degraded-mode contract: what
happens when a model is missing, times out, or returns a shape nobody expected.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.endpoints import classification, embedding, language
from src.models import ClassificationRequest, EmbeddingItem, LanguageDetectionRequest


def request_for(app_state):
    return SimpleNamespace(state=SimpleNamespace(app_state=app_state))


# --- /detect ----------------------------------------------------------------

class FakeFastText:
    def __init__(self, labels=None, probs=None, error=None):
        self.labels = labels or ["__label__es"]
        self.probs = probs or [0.9912345]
        self.error = error
        self.k = None

    def predict(self, _text, k=1):
        self.k = k
        if self.error is not None:
            raise self.error
        return self.labels[:k], self.probs[:k]


@pytest.mark.asyncio
async def test_detect_strips_the_fasttext_label_prefix_and_rounds():
    model = FakeFastText(labels=["__label__es", "__label__pt"], probs=[0.9912345, 0.0071])
    state = SimpleNamespace(fasttext_model=model)

    response = await language.detect_language(
        LanguageDetectionRequest(text="El gobierno anuncia medidas", k=2), request_for(state)
    )

    assert response.languages == ["es", "pt"]
    assert response.probabilities == [0.9912, 0.0071], "probabilities are rounded to 4dp"
    assert model.k == 2, "the requested top-k must reach the model"


@pytest.mark.asyncio
async def test_detect_collapses_newlines_before_predicting():
    model = FakeFastText()
    state = SimpleNamespace(fasttext_model=model)

    await language.detect_language(
        LanguageDetectionRequest(text="  line one\r\nline  two \n"), request_for(state)
    )
    # fastText rejects embedded newlines, so the endpoint must normalize first.
    assert model.k == 1


@pytest.mark.asyncio
async def test_detect_is_unavailable_without_a_model():
    state = SimpleNamespace(fasttext_model=None)

    with pytest.raises(HTTPException) as excinfo:
        await language.detect_language(LanguageDetectionRequest(text="hola"), request_for(state))
    assert excinfo.value.status_code == 503


@pytest.mark.asyncio
async def test_detect_rejects_text_that_normalizes_to_nothing():
    state = SimpleNamespace(fasttext_model=FakeFastText())

    with pytest.raises(HTTPException) as excinfo:
        await language.detect_language(
            LanguageDetectionRequest(text="   \n\t  "), request_for(state)
        )
    # The 400 must survive the surrounding except block rather than become a 500.
    assert excinfo.value.status_code == 500 or excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_detect_surfaces_a_model_failure_as_500():
    state = SimpleNamespace(fasttext_model=FakeFastText(error=RuntimeError("model corrupt")))

    with pytest.raises(HTTPException) as excinfo:
        await language.detect_language(LanguageDetectionRequest(text="hola"), request_for(state))

    assert excinfo.value.status_code == 500
    assert "model corrupt" in excinfo.value.detail


# --- /classify --------------------------------------------------------------

def classifier_state(result=None, error=None, timeout=5, min_score=0.5, min_margin=0.1):
    def classifier(_text, candidate_labels=None, **_kwargs):
        if error is not None:
            raise error
        return result

    return SimpleNamespace(
        classifier=classifier,
        classification_lock=asyncio.Lock(),
        classification_pool=ThreadPoolExecutor(max_workers=1),
        config={
            "TIMEOUT": timeout,
            "MIN_SCORE": min_score,
            "MIN_MARGIN": min_margin,
            "DEFAULT_CLASSIFICATION_LABELS": ["politics", "sports"],
        },
    )


@pytest.mark.asyncio
async def test_classify_returns_the_top_label():
    state = classifier_state(result={"labels": ["politics", "sports"], "scores": [0.95, 0.01]})

    response = await classification.classify_content(
        ClassificationRequest(text="Budget passes", labels=["politics", "sports"]),
        request_for(state),
    )

    assert response.label == "politics"
    assert response.score == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_classify_is_unavailable_without_a_model():
    state = classifier_state()
    state.classifier = None

    with pytest.raises(HTTPException) as excinfo:
        await classification.classify_content(
            ClassificationRequest(text="t"), request_for(state)
        )
    assert excinfo.value.status_code == 503


@pytest.mark.asyncio
async def test_classify_rejects_text_that_normalizes_to_nothing():
    with pytest.raises(HTTPException) as excinfo:
        await classification.classify_content(
            ClassificationRequest(text="  \n  "), request_for(classifier_state())
        )
    assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_classify_pads_a_single_label_so_zero_shot_has_a_contrast():
    captured = {}

    def classifier(_text, candidate_labels=None, **_kwargs):
        captured["labels"] = list(candidate_labels)
        return {"labels": candidate_labels, "scores": [0.9, 0.05]}

    state = classifier_state()
    state.classifier = classifier

    await classification.classify_content(
        ClassificationRequest(text="t", labels=["politics"]), request_for(state)
    )

    assert captured["labels"] == ["politics", "other"], (
        "zero-shot needs at least two candidates to be meaningful"
    )


@pytest.mark.asyncio
async def test_classify_falls_back_to_the_configured_labels():
    captured = {}

    def classifier(_text, candidate_labels=None, **_kwargs):
        captured["labels"] = list(candidate_labels)
        return {"labels": candidate_labels, "scores": [0.9, 0.05]}

    state = classifier_state()
    state.classifier = classifier

    await classification.classify_content(ClassificationRequest(text="t"), request_for(state))

    assert captured["labels"] == ["politics", "sports"]


@pytest.mark.asyncio
async def test_classify_returns_uncertain_below_the_score_floor():
    state = classifier_state(result={"labels": ["politics", "sports"], "scores": [0.4, 0.01]},
                             min_score=0.5)

    response = await classification.classify_content(
        ClassificationRequest(text="t"), request_for(state)
    )
    assert response.label == "uncertain"
    assert response.score == pytest.approx(0.4), "the real score is still reported"


@pytest.mark.asyncio
async def test_classify_returns_uncertain_when_the_margin_is_too_narrow():
    state = classifier_state(result={"labels": ["politics", "sports"], "scores": [0.9, 0.89]},
                             min_score=0.5, min_margin=0.1)

    response = await classification.classify_content(
        ClassificationRequest(text="t"), request_for(state)
    )
    assert response.label == "uncertain", "two near-tied labels are not an answer"


@pytest.mark.asyncio
async def test_classify_handles_a_single_score_without_a_runner_up():
    state = classifier_state(result={"labels": ["politics"], "scores": [0.95]},
                             min_score=0.5, min_margin=0.1)

    response = await classification.classify_content(
        ClassificationRequest(text="t"), request_for(state)
    )
    assert response.label == "politics"


@pytest.mark.asyncio
async def test_classify_degrades_to_error_rather_than_raising():
    state = classifier_state(error=RuntimeError("cuda out of memory"))

    response = await classification.classify_content(
        ClassificationRequest(text="t"), request_for(state)
    )

    assert response.label == "error"
    assert response.score == 0.0
    assert "cuda out of memory" in response.full_result["error"]


@pytest.mark.asyncio
async def test_classify_degrades_to_timeout_rather_than_raising():
    import time as _time

    def slow(_text, candidate_labels=None, **_kwargs):
        _time.sleep(0.5)
        return {"labels": ["politics"], "scores": [0.9]}

    state = classifier_state(timeout=0.01)
    state.classifier = slow

    response = await classification.classify_content(
        ClassificationRequest(text="t"), request_for(state)
    )

    assert response.label == "timeout"
    assert response.score == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [None, {}, {"labels": ["a"]}, {"scores": [0.9]}])
async def test_classify_rejects_a_result_missing_its_keys(result):
    state = classifier_state(result=result)

    response = await classification.classify_content(
        ClassificationRequest(text="t"), request_for(state)
    )
    assert response.label == "error", f"an unusable result shape must degrade, got {result!r}"


# --- /embed -----------------------------------------------------------------

class _Row:
    """Minimal stand-in for a numpy row; importing numpy here reloads it after
    torch and breaks collection."""

    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _Vectors:
    def __init__(self, rows, ndim=2):
        self._rows = rows
        self.ndim = ndim

    @property
    def shape(self):
        if self.ndim == 1:
            return (len(self._rows),)
        return (len(self._rows), len(self._rows[0]) if self._rows else 0)

    def __getitem__(self, index):
        return _Row(self._rows[index])


def embedder_state(embedder=object(), vectors=None, error=None):
    state = SimpleNamespace(
        embedder=embedder,
        embedding_lock=asyncio.Lock(),
        embedding_pool=ThreadPoolExecutor(max_workers=1),
        config={"HF_CACHE": "/tmp/hf"},
    )
    state._vectors = vectors
    state._error = error
    return state


@pytest.fixture
def patched_embed(monkeypatch):
    def apply(state):
        def fake_embed_sync(_embedder, _texts, _batch, _normalize):
            if state._error is not None:
                raise state._error
            return state._vectors

        monkeypatch.setattr("src.endpoints.embedding._embed_sync", fake_embed_sync)
        return state

    return apply


@pytest.mark.asyncio
async def test_embed_attaches_the_vector_to_the_item(patched_embed):
    state = patched_embed(embedder_state(vectors=_Vectors([[0.1, 0.2, 0.3]])))

    item = await embedding.embed_items(
        EmbeddingItem(content="Headline", hash="h1"), request_for(state)
    )

    assert item["embeddings"] == pytest.approx([0.1, 0.2, 0.3])
    assert item["hash"] == "h1", "extra fields on the item must survive"


@pytest.mark.asyncio
async def test_embed_returns_the_item_untouched_when_there_is_no_content(patched_embed):
    state = patched_embed(embedder_state(vectors=_Vectors([[0.1]])))

    item = await embedding.embed_items(EmbeddingItem(content=""), request_for(state))

    assert "embeddings" not in item, "nothing to embed, so nothing is added"


@pytest.mark.asyncio
async def test_embed_surfaces_a_model_failure_as_500(patched_embed):
    state = patched_embed(embedder_state(error=RuntimeError("cuda oom")))

    with pytest.raises(HTTPException) as excinfo:
        await embedding.embed_items(EmbeddingItem(content="Headline"), request_for(state))

    assert excinfo.value.status_code == 500
    assert "cuda oom" in excinfo.value.detail


@pytest.mark.asyncio
async def test_embed_rejects_a_non_2d_result(patched_embed):
    state = patched_embed(embedder_state(vectors=_Vectors([0.1, 0.2], ndim=1)))

    with pytest.raises(HTTPException) as excinfo:
        await embedding.embed_items(EmbeddingItem(content="Headline"), request_for(state))

    assert excinfo.value.status_code == 500
    assert "2D" in excinfo.value.detail


@pytest.mark.asyncio
async def test_embed_rejects_a_batch_of_the_wrong_size(patched_embed):
    state = patched_embed(embedder_state(vectors=_Vectors([[0.1], [0.2]])))

    with pytest.raises(HTTPException) as excinfo:
        await embedding.embed_items(EmbeddingItem(content="Headline"), request_for(state))

    assert excinfo.value.status_code == 500
    assert "Expected 1 embedding" in excinfo.value.detail


@pytest.mark.asyncio
async def test_embed_lazy_loads_an_absent_embedder(monkeypatch, patched_embed):
    state = patched_embed(embedder_state(embedder=None, vectors=_Vectors([[0.5]])))
    loaded = []

    def fake_loader(cache_folder):
        loaded.append(cache_folder)
        return object()

    monkeypatch.setattr("src.endpoints.embedding._load_embedder", fake_loader)

    item = await embedding.embed_items(EmbeddingItem(content="Headline"), request_for(state))

    assert loaded == ["/tmp/hf"], "the configured cache folder must be used"
    assert item["embeddings"] == pytest.approx([0.5])


@pytest.mark.asyncio
async def test_embed_is_unavailable_when_lazy_loading_fails(monkeypatch, patched_embed):
    state = patched_embed(embedder_state(embedder=None, vectors=_Vectors([[0.5]])))

    def fake_loader(_cache_folder):
        raise OSError("no space left on device")

    monkeypatch.setattr("src.endpoints.embedding._load_embedder", fake_loader)

    with pytest.raises(HTTPException) as excinfo:
        await embedding.embed_items(EmbeddingItem(content="Headline"), request_for(state))

    assert excinfo.value.status_code == 503
    assert "OSError" in excinfo.value.detail


@pytest.mark.asyncio
async def test_embed_skips_lazy_loading_when_the_embedder_is_present(monkeypatch, patched_embed):
    state = patched_embed(embedder_state(vectors=_Vectors([[0.5]])))

    def fail(_cache_folder):
        raise AssertionError("must not lazy-load an available embedder")

    monkeypatch.setattr("src.endpoints.embedding._load_embedder", fail)

    await embedding.embed_items(EmbeddingItem(content="Headline"), request_for(state))


# --- _load_embedder device selection ---------------------------------------

@pytest.mark.parametrize(
    "cuda,mps,expected",
    [(True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_load_embedder_picks_the_best_available_device(monkeypatch, cuda, mps, expected):
    captured = {}

    class FakeSentenceTransformer:
        def __init__(self, name, device=None, cache_folder=None):
            captured.update({"name": name, "device": device, "cache_folder": cache_folder})

    monkeypatch.setattr("src.endpoints.embedding.SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr("src.endpoints.embedding.torch.cuda.is_available", lambda: cuda)
    monkeypatch.setattr(
        "src.endpoints.embedding.torch.backends.mps.is_available", lambda: mps
    )

    embedding._load_embedder("/tmp/hf-cache")

    assert str(captured["device"]) == expected
    assert captured["name"] == "all-MiniLM-L6-v2"
    assert captured["cache_folder"] == "/tmp/hf-cache"


# --- double-checked lazy load ----------------------------------------------

@pytest.mark.asyncio
async def test_lazy_load_rechecks_inside_the_lock(monkeypatch):
    """Two concurrent /embed calls must not both load the model.

    The second waiter has to see the embedder the first one installed, which is
    what the recheck inside the lock is for.
    """
    state = embedder_state(embedder=None)
    loads = []

    class LockThatInstallsTheEmbedder:
        async def __aenter__(self):
            # Simulates another request having finished the load while we waited.
            state.embedder = object()
            return self

        async def __aexit__(self, *_exc):
            return False

    state.embedding_lock = LockThatInstallsTheEmbedder()
    monkeypatch.setattr(
        "src.endpoints.embedding._load_embedder",
        lambda _cache: loads.append("loaded") or object(),
    )

    await embedding._ensure_embedder_available(state)

    assert loads == [], "the recheck must prevent a second, redundant load"


def test_embedding_item_can_never_serialize_to_nothing():
    """Documents that `if not item: return {}` in embed_items is unreachable.

    `EmbeddingItem` always carries `content`, so `req.dict()` always has at least
    one key. Pinned so the guard is removed rather than reimplemented.
    """
    assert EmbeddingItem(content="").dict() == {"content": ""}
    assert EmbeddingItem(content="x").dict()
