"""A post whose language is already known must never be run through a detector.

Owner rule: if a post has no language we detect it, but we do not determine the
language of posts whose language is already known. Every public entry point that
can reach the FastText detector is pinned here with a detector that records each
call, so a call cannot be swallowed by an endpoint's own error handling.
"""

import asyncio
import pathlib
import re
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from src.endpoints import analysis as analysis_module
from src.endpoints import translation as translation_module
from src.helpers import scout_queries as scout_module
from src.helpers.scout_queries import generate_scout_queries
from src.endpoints.translation import translate_text
from src.models import TranslationRequest, TranslationResponse
from src.operations.models import OperationEnvelope
from src.operations.worker import OperationWorker

SRC = pathlib.Path(__file__).parent.parent / "src"


class RecordingFastText:
    """FastText stand-in that records every call; it answers Spanish."""

    def __init__(self):
        self.calls = []

    def predict(self, text, k=1):
        self.calls.append(text)
        return ["__label__es"], [0.99]


def request_for(app_state):
    return SimpleNamespace(state=SimpleNamespace(app_state=app_state))


def analysis_state(fasttext):
    return SimpleNamespace(
        fasttext_model=fasttext,
        translator=object(),
        embedder=None,
        classifier=None,
        config={
            "TIMEOUT": 30,
            "MIN_SCORE": 0.0,
            "MIN_MARGIN": 0.0,
            "DEFAULT_CLASSIFICATION_LABELS": ["news"],
            "PROCESSING_TRANSLATE_TO_ENGLISH": True,
        },
    )


@pytest.fixture
def translated_sources(monkeypatch):
    """Record the source language the analysis pipeline hands to translation."""
    sources = []

    async def fake_translate(req, _http_request):
        sources.append(req.source_language)
        return TranslationResponse(
            original_text=req.text,
            translated_text="translated",
            source_language=req.source_language,
            target_language="eng",
        )

    monkeypatch.setattr(analysis_module, "translate_text", fake_translate)
    return sources


def item(item_id, lang):
    payload = {
        "id": item_id,
        "source": "rss",
        "uri": f"https://example.test/{item_id}",
        "hash": f"hash-{item_id}",
        "content": "contenido de prueba",
        "createdAt": "2026-08-29T12:00:00Z",
    }
    if lang is not ...:
        payload["lang"] = lang
    return payload


def analysis_request(lang, count):
    return analysis_module.UnifiedAnalysisRequest(
        items=[item(f"p{i}", lang) for i in range(count)],
        translate_to_english=True,
    )


KNOWN = [
    ("fi", "fi"),
    ("  pt ", "pt"),
    ({"lang": "de"}, "de"),
    ({"language": "fr"}, "fr"),
    ({"code": "it"}, "it"),
    ({"languages": ["ja", "ko"]}, "ja"),
]
BLANK = ["", "   ", None, ..., {}, {"lang": " "}, {"languages": []}]


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2], ids=["single", "batched"])
@pytest.mark.parametrize("lang,expected", KNOWN)
async def test_analysis_never_detects_a_known_language(lang, expected, count, translated_sources):
    fasttext = RecordingFastText()
    response = await analysis_module.unified_analysis(
        analysis_request(lang, count), request_for(analysis_state(fasttext))
    )

    assert fasttext.calls == []
    assert translated_sources == [expected] * count
    assert [r.language_detection.languages for r in response.results] == [[expected]] * count


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2], ids=["single", "batched"])
@pytest.mark.parametrize("lang", BLANK, ids=repr)
async def test_analysis_detects_when_the_language_is_blank(lang, count, translated_sources):
    fasttext = RecordingFastText()
    response = await analysis_module.unified_analysis(
        analysis_request(lang, count), request_for(analysis_state(fasttext))
    )

    assert len(fasttext.calls) == count
    assert translated_sources == ["es"] * count
    assert [r.language_detection.languages for r in response.results] == [["es"]] * count


@pytest.mark.asyncio
async def test_analysis_does_not_detect_when_detection_is_not_requested(translated_sources):
    fasttext = RecordingFastText()
    request = analysis_request(None, 1)
    request.detect_language = False

    await analysis_module.unified_analysis(request, request_for(analysis_state(fasttext)))

    assert fasttext.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2], ids=["single", "batched"])
async def test_analyze_posts_operation_keeps_the_known_language(count, translated_sources):
    """Queue/worker entry point: payload `lang` survives to the response unchanged."""
    fasttext = RecordingFastText()
    worker = OperationWorker(SimpleNamespace(), analysis_state(fasttext))
    envelope = OperationEnvelope(
        operation_type="analyze_posts",
        payload={"items": [{"post": item(f"p{i}", "fi")} for i in range(count)]},
    )

    result = await worker._run_operation(envelope)

    assert fasttext.calls == []
    assert translated_sources == ["fi"] * count
    assert [i["post"]["lang"] for i in result["items"]] == ["fi"] * count


@pytest.mark.asyncio
async def test_analyze_posts_operation_detects_a_blank_language(translated_sources):
    fasttext = RecordingFastText()
    worker = OperationWorker(SimpleNamespace(), analysis_state(fasttext))
    envelope = OperationEnvelope(
        operation_type="analyze_posts",
        payload={"items": [{"post": item("p0", "")}]},
    )

    result = await worker._run_operation(envelope)

    assert len(fasttext.calls) == 1
    assert result["items"][0]["post"]["lang"] == "es"


def translating_state(fasttext):
    return SimpleNamespace(
        fasttext_model=fasttext,
        translator=object(),
        translation_lock=asyncio.Lock(),
        translation_pool=ThreadPoolExecutor(max_workers=1),
        config={"TIMEOUT": 30},
    )


@pytest.fixture
def translator_calls(monkeypatch):
    calls = []

    def fake_translate_sync(_translator, text, source_lang, target_lang, _model_name):
        calls.append(source_lang)
        return [{"translation_text": f"TRANSLATED::{source_lang}"}]

    monkeypatch.setattr(translation_module, "_translate_sync", fake_translate_sync)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["es", "fi", "sq", "xx", "lb"])
async def test_translation_never_detects_a_known_source_language(language, translator_calls):
    """Supported, unsupported and unknown codes alike: the given one is final."""
    fasttext = RecordingFastText()
    response = await translate_text(
        TranslationRequest(text="Kryeministri paraqiti buxhetin", source_language=language),
        request_for(translating_state(fasttext)),
    )

    assert fasttext.calls == []
    expected = translation_module._normalize_translation_language(language)
    assert response.source_language in {language, expected}
    assert translator_calls in ([], [expected])


@pytest.mark.asyncio
@pytest.mark.parametrize("language", [None, "", "   "])
async def test_translation_detects_when_the_source_language_is_blank(language, translator_calls):
    fasttext = RecordingFastText()
    response = await translate_text(
        TranslationRequest(text="texto en español", source_language=language),
        request_for(translating_state(fasttext)),
    )

    assert len(fasttext.calls) == 1
    assert response.source_language == "es"
    assert translator_calls == ["es"]


@pytest.mark.asyncio
async def test_translate_operation_never_detects_a_known_source_language(translator_calls):
    fasttext = RecordingFastText()
    worker = OperationWorker(SimpleNamespace(), translating_state(fasttext))
    envelope = OperationEnvelope(
        operation_type="translate_text",
        payload={"text": "Kryeministri paraqiti buxhetin", "source_language": "sq"},
    )

    await worker._run_operation(envelope)

    assert fasttext.calls == []


@pytest.fixture
def scout_detector_calls(monkeypatch):
    calls = []

    async def fake_detect(req, _http_request):
        calls.append(req.text)
        return SimpleNamespace(languages=["es"])

    async def fake_translate(req, _http_request):
        return TranslationResponse(
            original_text=req.text,
            translated_text="translated",
            source_language=req.source_language,
            target_language="eng",
        )

    monkeypatch.setattr(scout_module, "detect_language", fake_detect)
    monkeypatch.setattr(scout_module, "translate_text", fake_translate)
    return calls


@pytest.mark.asyncio
async def test_scout_queries_never_detect_a_known_source_language(scout_detector_calls):
    state = SimpleNamespace(fasttext_model=object(), translator=object(), nlp=None)

    _, _, language = await generate_scout_queries(state, "Titulo de prueba", "fi")

    assert scout_detector_calls == []
    assert language == "fi"


@pytest.mark.asyncio
@pytest.mark.parametrize("language", [None, "", "   "])
async def test_scout_queries_detect_when_the_source_language_is_blank(language, scout_detector_calls):
    state = SimpleNamespace(fasttext_model=object(), translator=object(), nlp=None)

    _, _, detected = await generate_scout_queries(state, "Titulo de prueba", language)

    assert scout_detector_calls == ["Titulo de prueba"]
    assert detected == "es"


def _modules_matching(pattern):
    regex = re.compile(pattern)
    return sorted(
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if "grpc/mistify" not in str(path) and regex.search(path.read_text())
    )


def test_detector_call_sites_are_the_audited_ones():
    """A new detector call site must be reviewed against the known-language rule."""
    assert _modules_matching(r"fasttext_model\.predict\(") == [
        "endpoints/language.py",
        "endpoints/translation.py",
    ]
    assert _modules_matching(r"(?<!def )\bdetect_language\(") == [
        "endpoints/analysis.py",
        "helpers/scout_queries.py",
        "operations/worker.py",
    ]
