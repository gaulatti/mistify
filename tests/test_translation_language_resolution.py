import ast
import asyncio
import pathlib
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families

from src.endpoints import translation as translation_module
from src.metrics import TRANSLATION_SKIP_REASONS
from src.endpoints.translation import (
    SUPPORTED_TRANSLATION_LANGUAGES,
    UNTRANSLATABLE_LANGUAGES,
    _normalize_translation_language,
    translate_text,
)
from src.models import TranslationRequest

_ASYNC_WRAPPERS = pathlib.Path(__file__).parent.parent / "src" / "helpers" / "async_wrappers.py"


def _seamless_lang_mapping():
    """Read the mapping table literal; it is a local inside `_translate_sync`."""
    tree = ast.parse(_ASYNC_WRAPPERS.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "seamless_lang_mapping" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("seamless_lang_mapping not found")


class _FakeFastText:
    """Returns fastText-shaped output: labels ordered by descending probability."""

    def __init__(self, ranked):
        self._ranked = ranked

    def predict(self, _text, k=1):
        ranked = self._ranked[:k]
        return (
            [f"__label__{lang}" for lang, _ in ranked],
            [prob for _, prob in ranked],
        )


def _request(http_state):
    return SimpleNamespace(state=SimpleNamespace(app_state=http_state))


def _app_state(fasttext_model):
    # A truthy translator is enough: these cases must all return before using it.
    return SimpleNamespace(fasttext_model=fasttext_model, translator=object())


def _translating_app_state(fasttext_model):
    """A state complete enough to reach the translator call, which is recorded."""
    return SimpleNamespace(
        fasttext_model=fasttext_model,
        translator=object(),
        translation_lock=asyncio.Lock(),
        translation_pool=ThreadPoolExecutor(max_workers=1),
        config={"TIMEOUT": 30},
    )


def _record_translator_calls(monkeypatch):
    """Replace the model call so the resolved source language becomes observable."""
    calls = []

    def fake_translate_sync(_translator, text, source_lang, target_lang, _model_name):
        calls.append({"text": text, "source": source_lang, "target": target_lang})
        return [{"translation_text": f"TRANSLATED::{source_lang}"}]

    monkeypatch.setattr(translation_module, "_translate_sync", fake_translate_sync)
    return calls


def test_finnish_is_supported_and_maps_to_seamless_fin():
    """Seamless M4T v2 exposes a `fin` token; `fi` was missing from both tables."""
    assert "fi" in SUPPORTED_TRANSLATION_LANGUAGES
    assert "fin" in SUPPORTED_TRANSLATION_LANGUAGES
    mapping = _seamless_lang_mapping()
    assert mapping["fi"] == "fin"
    assert mapping["fin"] == "fin"


def test_every_supported_two_letter_code_reaches_the_translator():
    """Guards the bug class: a language may not be accepted and then unmappable."""
    mapping = _seamless_lang_mapping()
    unmappable = sorted(
        code
        for code in SUPPORTED_TRANSLATION_LANGUAGES
        if len(code) == 2 and code not in mapping
    )
    assert unmappable == []


@pytest.mark.parametrize(
    "language,expected",
    [("cnr", "sr"), ("lb", "de"), ("tk", "tr"), ("ast", "es")],
)
def test_unsupported_languages_normalize_to_intelligible_relatives(language, expected):
    assert _normalize_translation_language(language) == expected
    assert expected in SUPPORTED_TRANSLATION_LANGUAGES


@pytest.mark.asyncio
async def test_confident_unsupported_detection_skips_translation():
    """Finnish at 0.998 must not lose to Thai at 0.0005.

    Regression guard: the old code picked the first *supported* candidate regardless
    of probability, so confidently-detected Albanian was translated as something else.
    """
    fasttext = _FakeFastText([("sq", 0.991), ("th", 0.0004), ("it", 0.0002)])
    response = await translate_text(
        TranslationRequest(text="Kryeministri paraqiti buxhetin e vitit", source_language=None),
        _request(_app_state(fasttext)),
    )

    assert response.source_language == "sq"
    assert response.translated_text == response.original_text


@pytest.mark.asyncio
async def test_declared_untranslatable_language_short_circuits():
    assert "dv" in UNTRANSLATABLE_LANGUAGES
    response = await translate_text(
        TranslationRequest(text="އަސްލު ބަސް", source_language="dv"),
        _request(_app_state(_FakeFastText([("dv", 0.99)]))),
    )

    assert response.translated_text == response.original_text


@pytest.mark.asyncio
async def test_low_confidence_top_label_still_yields_to_supported_candidate(monkeypatch):
    """Below the confidence floor the supported runner-up is still the better guess."""
    calls = _record_translator_calls(monkeypatch)
    fasttext = _FakeFastText([("mi", 0.21), ("es", 0.19), ("en", 0.08)])
    response = await translate_text(
        TranslationRequest(text="texto corto ambiguo", source_language=None),
        _request(_translating_app_state(fasttext)),
    )

    # "mi" is untranslatable but was not confident, so Spanish is used instead and
    # translation proceeds rather than short-circuiting.
    assert response.source_language == "es"
    assert [call["source"] for call in calls] == ["es"]


@pytest.mark.asyncio
async def test_finnish_is_handed_to_the_translator_as_finnish(monkeypatch):
    """The exact prod regression: `fi` reached Seamless as Thai, Italian or Hungarian."""
    calls = _record_translator_calls(monkeypatch)
    response = await translate_text(
        TranslationRequest(text="Hallituksen esitys uudesta talousarviosta", source_language="fi"),
        _request(_translating_app_state(_FakeFastText([("fi", 0.998), ("th", 0.0005)]))),
    )

    assert [call["source"] for call in calls] == ["fi"]
    assert response.translated_text == "TRANSLATED::fi"


@pytest.mark.asyncio
async def test_skips_are_counted_with_bounded_reason_labels():
    """Exercises the real collector output, not just the increment call."""
    await translate_text(
        TranslationRequest(text="އަސްލު", source_language="dv"),
        _request(_app_state(_FakeFastText([("dv", 0.99)]))),
    )
    await translate_text(
        TranslationRequest(text="Kryeministri paraqiti buxhetin", source_language=None),
        _request(_app_state(_FakeFastText([("sq", 0.99), ("th", 0.0004)]))),
    )

    samples = [
        sample
        for family in text_string_to_metric_families(generate_latest().decode())
        if family.name == "mistify_translation_skipped"
        for sample in family.samples
    ]
    assert samples, "mistify_translation_skipped_total absent from exposition"

    observed = {sample.labels["reason"] for sample in samples}
    assert observed <= set(TRANSLATION_SKIP_REASONS), observed
    assert {"declared_untranslatable", "detected_unsupported"} <= observed
    # The label set must stay a controlled enum: no language codes, no free text.
    assert all(set(sample.labels) == {"reason"} for sample in samples)


@pytest.mark.asyncio
async def test_missing_translator_still_raises():
    with pytest.raises(HTTPException) as excinfo:
        await translate_text(
            TranslationRequest(text="hola", source_language="es"),
            _request(SimpleNamespace(fasttext_model=None, translator=None)),
        )

    assert excinfo.value.status_code == 503
