"""Covers `src/helpers/models.py`, the startup model loader.

Every loader here is wrapped in try/except so a missing model degrades the
service rather than stopping it. That degradation is the contract worth pinning:
which component ends up None, and which translator name is reported, decides
what `/health` tells an operator.
"""

from types import SimpleNamespace

import pytest

from src.helpers import models as models_module

CONFIG = {
    "FASTTEXT_MODEL_PATH": "/tmp/lid.176.bin",
    "FASTTEXT_MODEL_URL": "https://example.test/lid.176.bin",
    "HF_CACHE": "/tmp/hf-cache",
}


class FakePipelineFactory:
    """Routes `pipeline()` calls by model name, like the real dispatcher."""

    def __init__(self, failures=(), results=None):
        self.failures = set(failures)
        self.results = results or {}
        self.calls = []

    def __call__(self, task, model=None, **kwargs):
        self.calls.append({"task": task, "model": model, **kwargs})
        for marker in self.failures:
            if marker in model:
                raise RuntimeError(f"{marker} unavailable")
        for marker, value in self.results.items():
            if marker in model:
                return value
        return SimpleNamespace(name=model)


@pytest.fixture
def loaders(monkeypatch):
    """Patches every external loader; tests override the pieces they care about."""
    state = SimpleNamespace(
        pipeline=FakePipelineFactory(),
        fasttext_model=SimpleNamespace(predict=lambda *_a, **_k: (["__label__en"], [0.9])),
        fasttext_error=None,
        path_exists=True,
        download_ok=True,
        embedder=object(),
        embedder_error=None,
        nlp=object(),
        nlp_error=None,
    )

    def fake_load_model(_path):
        if state.fasttext_error is not None:
            raise state.fasttext_error
        return state.fasttext_model

    def fake_download(_path, _url):
        return state.download_ok

    def fake_sentence_transformer(_name, device=None, cache_folder=None):
        if state.embedder_error is not None:
            raise state.embedder_error
        state.embedder_args = {"device": device, "cache_folder": cache_folder}
        return state.embedder

    def fake_spacy_load(_name, disable=None):
        if state.nlp_error is not None:
            raise state.nlp_error
        state.spacy_disable = disable
        return state.nlp

    monkeypatch.setattr(models_module.fasttext, "load_model", fake_load_model)
    monkeypatch.setattr(models_module, "download_fasttext_model", fake_download)
    monkeypatch.setattr(models_module, "SentenceTransformer", fake_sentence_transformer)
    monkeypatch.setattr(models_module.spacy, "load", fake_spacy_load)
    monkeypatch.setattr(models_module.os.path, "exists", lambda _p: state.path_exists)
    monkeypatch.setattr(models_module, "pipeline", lambda *a, **k: state.pipeline(*a, **k))
    monkeypatch.setattr(models_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(models_module.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(models_module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setenv("TRANSLATION_ENABLED", "true")
    return state


def initialize(state=None):
    return models_module.initialize_models(CONFIG)


# --- download_fasttext_model ------------------------------------------------

def test_download_reports_success(monkeypatch):
    called = {}
    monkeypatch.setattr(
        models_module.urllib.request, "urlretrieve",
        lambda url, path: called.update({"url": url, "path": path}),
    )

    assert models_module.download_fasttext_model("/tmp/m.bin", "https://e.test/m.bin") is True
    assert called == {"url": "https://e.test/m.bin", "path": "/tmp/m.bin"}


def test_download_reports_failure_without_raising(monkeypatch):
    def boom(_url, _path):
        raise OSError("no space left on device")

    monkeypatch.setattr(models_module.urllib.request, "urlretrieve", boom)

    # Startup must continue even if the download fails.
    assert models_module.download_fasttext_model("/tmp/m.bin", "https://e.test/m.bin") is False


# --- device selection -------------------------------------------------------

def test_cuda_is_preferred_and_the_cache_is_cleared(loaders, monkeypatch):
    cleared = []
    monkeypatch.setattr(models_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(models_module.torch.cuda, "empty_cache", lambda: cleared.append(1))

    initialize()

    assert cleared == [1], "CUDA selection must clear the cache before loading"
    classifier_call = next(c for c in loaders.pipeline.calls if "distilbart" in c["model"])
    assert classifier_call["device"] == 0
    assert classifier_call["model_kwargs"] == {"low_cpu_mem_usage": True}


def test_mps_is_used_when_cuda_is_absent(loaders, monkeypatch):
    monkeypatch.setattr(models_module.torch.backends.mps, "is_available", lambda: True)

    initialize()

    classifier_call = next(c for c in loaders.pipeline.calls if "distilbart" in c["model"])
    assert classifier_call["device"] == 0
    assert classifier_call["model_kwargs"] == {}, "only CUDA gets the low-memory kwargs"


def test_cpu_is_the_fallback_device(loaders):
    initialize()

    classifier_call = next(c for c in loaders.pipeline.calls if "distilbart" in c["model"])
    assert classifier_call["device"] == -1


# --- FastText ---------------------------------------------------------------

def test_fasttext_loads_from_an_existing_file(loaders):
    fasttext_model, *_ = initialize()
    assert fasttext_model is loaders.fasttext_model


def test_fasttext_is_downloaded_when_absent(loaders, monkeypatch):
    loaders.path_exists = False
    downloads = []
    monkeypatch.setattr(
        models_module, "download_fasttext_model",
        lambda path, url: downloads.append((path, url)) or True,
    )

    fasttext_model, *_ = initialize()

    assert downloads == [(CONFIG["FASTTEXT_MODEL_PATH"], CONFIG["FASTTEXT_MODEL_URL"])]
    assert fasttext_model is loaders.fasttext_model


def test_fasttext_is_none_when_the_download_fails(loaders, monkeypatch):
    loaders.path_exists = False
    monkeypatch.setattr(models_module, "download_fasttext_model", lambda _p, _u: False)

    fasttext_model, *_ = initialize()

    assert fasttext_model is None, "a failed download must not stop startup"


def test_fasttext_is_none_when_loading_raises(loaders):
    loaders.fasttext_error = RuntimeError("model file corrupt")

    fasttext_model, *_ = initialize()
    assert fasttext_model is None


def test_fasttext_is_smoke_tested_after_loading(loaders):
    predictions = []
    loaders.fasttext_model = SimpleNamespace(
        predict=lambda text, k=1: predictions.append((text, k)) or (["__label__en"], [0.9])
    )

    initialize()

    assert predictions == [("Hello world", 1)], (
        "a model that loads but cannot predict must be caught at startup"
    )


def test_a_model_failing_its_smoke_test_is_still_installed(loaders):
    """Documents a bug rather than asserting desired behaviour.

    `fasttext_model` is assigned before `predict()` is called, and the except
    block does not reset it, so a model that loads but cannot predict is logged
    as a failure and then returned anyway. The smoke test therefore detects the
    problem without preventing it: /health reports fasttext_loaded=true and every
    /detect call fails at runtime instead.

    Filed separately. Pinned here so a fix has a target.
    """
    def failing_predict(*_a, **_k):
        raise RuntimeError("corrupt vectors")

    broken = SimpleNamespace(predict=failing_predict)
    loaders.fasttext_model = broken

    fasttext_model, *_ = initialize()

    assert fasttext_model is broken, (
        "the broken model is no longer returned; the bug is fixed, update this test"
    )


# --- classifier -------------------------------------------------------------

def test_classifier_is_none_when_loading_fails(loaders):
    loaders.pipeline = FakePipelineFactory(failures=["distilbart"])

    _, classifier, *_ = initialize()
    assert classifier is None


def test_classifier_uses_the_configured_hypothesis_template(loaders):
    initialize()

    call = next(c for c in loaders.pipeline.calls if "distilbart" in c["model"])
    assert call["hypothesis_template"] == "This post is {}."
    assert call["cache_dir"] == str(CONFIG["HF_CACHE"])


# --- translator -------------------------------------------------------------

def test_translation_can_be_disabled_entirely(loaders, monkeypatch):
    monkeypatch.setenv("TRANSLATION_ENABLED", "false")

    _, _, translator, _, _, name = initialize()

    assert translator is None
    assert name == "none"
    assert not any("seamless" in c["model"] for c in loaders.pipeline.calls)


def test_seamless_is_loaded_and_named(loaders):
    _, _, translator, _, _, name = initialize()

    assert name == "seamless-m4t-v2"
    assert translator is not None


def test_seamless_model_is_moved_onto_the_target_device(loaders):
    moved = []
    translator = SimpleNamespace(
        model=SimpleNamespace(
            to=lambda device: moved.append(device),
            parameters=lambda: iter([SimpleNamespace(device="cpu")]),
        )
    )
    loaders.pipeline = FakePipelineFactory(results={"seamless": translator})

    _, _, _, _, _, name = initialize()

    assert moved, "pipelines sometimes leave parts on CPU; the move must be explicit"
    assert name == "seamless-m4t-v2"


def test_a_failed_device_move_does_not_discard_the_translator(loaders):
    def failing_to(_device):
        raise RuntimeError("device mismatch")

    translator = SimpleNamespace(model=SimpleNamespace(to=failing_to))
    loaders.pipeline = FakePipelineFactory(results={"seamless": translator})

    _, _, loaded, _, _, name = initialize()

    assert loaded is translator
    assert name == "seamless-m4t-v2", "a move failure is a warning, not a load failure"


def test_helsinki_is_the_fallback_when_seamless_fails(loaders):
    loaders.pipeline = FakePipelineFactory(failures=["seamless"])

    _, _, translator, _, _, name = initialize()

    assert name == "helsinki-nlp"
    assert translator is not None
    assert any("opus-mt-mul-en" in c["model"] for c in loaders.pipeline.calls)


def test_translator_is_none_when_both_models_fail(loaders):
    loaders.pipeline = FakePipelineFactory(failures=["seamless", "opus-mt"])

    _, _, translator, _, _, name = initialize()

    assert translator is None
    assert name == "none", "/health must report that translation is unavailable"


# --- embedder and spaCy -----------------------------------------------------

def test_embedder_receives_the_device_and_cache_folder(loaders):
    _, _, _, embedder, _, _ = initialize()

    assert embedder is loaders.embedder
    assert loaders.embedder_args["cache_folder"] == str(CONFIG["HF_CACHE"])
    assert str(loaders.embedder_args["device"]) == "cpu"


def test_embedder_is_none_when_loading_fails(loaders):
    loaders.embedder_error = RuntimeError("no such model")

    _, _, _, embedder, _, _ = initialize()
    assert embedder is None


def test_spacy_disables_the_components_clustering_does_not_use(loaders):
    _, _, _, _, nlp, _ = initialize()

    assert nlp is loaders.nlp
    assert set(loaders.spacy_disable) == {"tagger", "parser", "lemmatizer"}


def test_spacy_is_none_when_the_model_is_missing(loaders):
    loaders.nlp_error = OSError("en_core_web_sm not found")

    _, _, _, _, nlp, _ = initialize()
    assert nlp is None


# --- degraded startup -------------------------------------------------------

def test_every_model_can_fail_without_stopping_startup(loaders):
    loaders.fasttext_error = RuntimeError("x")
    loaders.embedder_error = RuntimeError("x")
    loaders.nlp_error = RuntimeError("x")
    loaders.pipeline = FakePipelineFactory(failures=["distilbart", "seamless", "opus-mt"])

    fasttext_model, classifier, translator, embedder, nlp, name = initialize()

    assert (fasttext_model, classifier, translator, embedder, nlp) == (None,) * 5
    assert name == "none"


def test_the_translator_name_is_always_set_before_the_final_guard(loaders, monkeypatch):
    """Documents that the `translator_model_name is None` guard is unreachable.

    Every path assigns it: "none" when translation is disabled, "seamless-m4t-v2"
    on success, "helsinki-nlp" on fallback, and "none" when both fail. The guard
    is defensive, so it is documented rather than proposed for removal - but it
    means the reported name can never be None, which /health depends on.
    """
    cases = {
        "disabled": (FakePipelineFactory(), {"TRANSLATION_ENABLED": "false"}, "none"),
        "seamless": (FakePipelineFactory(), {}, "seamless-m4t-v2"),
        "fallback": (FakePipelineFactory(failures=["seamless"]), {}, "helsinki-nlp"),
        "both failed": (FakePipelineFactory(failures=["seamless", "opus-mt"]), {}, "none"),
    }
    for label, (factory, env, expected) in cases.items():
        loaders.pipeline = factory
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        else:
            if not env:
                monkeypatch.setenv("TRANSLATION_ENABLED", "true")

        *_, name = initialize()
        assert name == expected, f"{label}: reported {name!r}"
        assert name is not None
