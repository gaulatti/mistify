"""Covers `/translate`'s resolution branches and model-response handling.

These are real failure modes: a timed-out model call, a null result, and the
several response shapes the different translator backends return.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.endpoints import translation as translation_module
from src.endpoints.translation import _normalize_translation_language, translate_text
from src.models import TranslationRequest


class _FakeFastText:
    def __init__(self, ranked=None, error=None):
        self._ranked = ranked or []
        self._error = error

    def predict(self, _text, k=1):
        if self._error is not None:
            raise self._error
        ranked = self._ranked[:k]
        return (
            [f"__label__{lang}" for lang, _ in ranked],
            [prob for _, prob in ranked],
        )


def _request(app_state):
    return SimpleNamespace(state=SimpleNamespace(app_state=app_state))


def _state(fasttext_model=None, timeout=30, model_name=None):
    return SimpleNamespace(
        fasttext_model=fasttext_model,
        translator=object(),
        translator_model_name=model_name,
        translation_lock=asyncio.Lock(),
        translation_pool=ThreadPoolExecutor(max_workers=1),
        config={"TIMEOUT": timeout},
    )


def _returns(value, monkeypatch):
    monkeypatch.setattr(
        translation_module,
        "_translate_sync",
        lambda *_args, **_kwargs: value,
    )


# --- normalization branches -------------------------------------------------

def test_alias_is_applied_before_fallbacks():
    assert _normalize_translation_language("eng") == "en"
    assert _normalize_translation_language("__label__spa") == "es"


def test_region_suffix_falls_back_on_its_base_language():
    """`cnr-ME` must resolve through the base code, not be treated as unknown."""
    assert _normalize_translation_language("cnr-ME") == "sr"
    assert _normalize_translation_language("ast-ES") == "es"


def test_country_variant_without_fallback_keeps_its_base():
    assert _normalize_translation_language("pt-BR") == "pt"


# --- resolution branches ----------------------------------------------------

@pytest.mark.asyncio
async def test_supplied_language_is_normalized_before_use(monkeypatch):
    _returns([{"translation_text": "done"}], monkeypatch)
    response = await translate_text(
        TranslationRequest(text="Hola mundo", source_language="SPA"),
        _request(_state()),
    )
    assert response.source_language == "es"


@pytest.mark.asyncio
async def test_unsupported_supplied_language_is_replaced_by_supported_candidate(monkeypatch):
    """Below the confidence floor, a supported runner-up replaces the supplied code."""
    _returns([{"translation_text": "done"}], monkeypatch)
    fasttext = _FakeFastText([("zza", 0.21), ("es", 0.19)])
    response = await translate_text(
        TranslationRequest(text="texto ambiguo", source_language="zzb"),
        _request(_state(fasttext)),
    )
    assert response.source_language == "es"


@pytest.mark.asyncio
async def test_detection_failure_does_not_fail_the_request(monkeypatch):
    """A broken detector must not take the translation down with it."""
    _returns([{"translation_text": "done"}], monkeypatch)
    fasttext = _FakeFastText(error=RuntimeError("fasttext exploded"))
    response = await translate_text(
        TranslationRequest(text="texto", source_language=None),
        _request(_state(fasttext)),
    )
    # Falls through to the unknown-language default rather than raising.
    assert response.source_language == "en"


@pytest.mark.asyncio
async def test_unknown_language_without_a_detector_defaults_to_english(monkeypatch):
    _returns([{"translation_text": "done"}], monkeypatch)
    response = await translate_text(
        TranslationRequest(text="some text", source_language=None),
        _request(_state(fasttext_model=None)),
    )
    assert response.source_language == "en"


@pytest.mark.asyncio
async def test_unsupported_language_without_a_detector_skips_translation():
    """No detector to rescue it, so decline rather than mislabel it English."""
    response = await translate_text(
        TranslationRequest(text="eнекий текст", source_language="zzb"),
        _request(_state(fasttext_model=None)),
    )
    assert response.translated_text == response.original_text
    assert response.source_language == "zzb"


# --- model execution and response handling ----------------------------------

@pytest.mark.asyncio
async def test_timeout_surfaces_as_408(monkeypatch):
    def slow(*_args, **_kwargs):
        time.sleep(0.5)
        return [{"translation_text": "late"}]

    monkeypatch.setattr(translation_module, "_translate_sync", slow)
    with pytest.raises(HTTPException) as excinfo:
        await translate_text(
            TranslationRequest(text="hola", source_language="es"),
            _request(_state(timeout=0.01)),
        )
    assert excinfo.value.status_code == 408


@pytest.mark.asyncio
async def test_model_exception_surfaces_as_500(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("cuda out of memory")

    monkeypatch.setattr(translation_module, "_translate_sync", boom)
    with pytest.raises(HTTPException) as excinfo:
        await translate_text(
            TranslationRequest(text="hola", source_language="es"),
            _request(_state()),
        )
    assert excinfo.value.status_code == 500


@pytest.mark.asyncio
async def test_null_result_surfaces_as_500(monkeypatch):
    _returns(None, monkeypatch)
    with pytest.raises(HTTPException) as excinfo:
        await translate_text(
            TranslationRequest(text="hola", source_language="es"),
            _request(_state()),
        )
    assert excinfo.value.status_code == 500


@pytest.mark.asyncio
async def test_list_result_falls_back_through_alternative_keys(monkeypatch):
    _returns([{"generated_text": "from generated_text"}], monkeypatch)
    response = await translate_text(
        TranslationRequest(text="hola", source_language="es"),
        _request(_state()),
    )
    assert response.translated_text == "from generated_text"


@pytest.mark.asyncio
async def test_dict_result_is_accepted(monkeypatch):
    _returns({"text": "from dict text"}, monkeypatch)
    response = await translate_text(
        TranslationRequest(text="hola", source_language="es"),
        _request(_state()),
    )
    assert response.translated_text == "from dict text"


@pytest.mark.asyncio
async def test_unexpected_result_type_is_stringified(monkeypatch):
    _returns(12345, monkeypatch)
    response = await translate_text(
        TranslationRequest(text="hola", source_language="es"),
        _request(_state()),
    )
    assert response.translated_text == "12345"


@pytest.mark.asyncio
async def test_blank_translation_preserves_the_original_text(monkeypatch):
    _returns([{"translation_text": "   "}], monkeypatch)
    response = await translate_text(
        TranslationRequest(text="hola mundo", source_language="es"),
        _request(_state()),
    )
    assert response.translated_text == "hola mundo"


@pytest.mark.asyncio
async def test_unparseable_result_surfaces_as_500(monkeypatch):
    class Hostile:
        def get(self, *_args, **_kwargs):
            raise TypeError("not a mapping")

    _returns([Hostile()], monkeypatch)
    with pytest.raises(HTTPException) as excinfo:
        await translate_text(
            TranslationRequest(text="hola", source_language="es"),
            _request(_state()),
        )
    assert excinfo.value.status_code == 500
