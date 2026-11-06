"""Covers the translation orchestration in `src/helpers/async_wrappers.py`.

`_translate_sync` escalates through progressively shorter inputs until the model
succeeds. These tests pin that escalation order and each backend's call shape, so
the behaviour is locked down independently of how the function is structured.
"""

from types import SimpleNamespace

import pytest

from src.helpers.async_wrappers import (
    LAST_RESORT_WORDS,
    SEAMLESS_LANG_MAPPING,
    SEAMLESS_MAX_INPUT_TOKENS,
    STANDARD_LANG_MAPPING,
    STANDARD_MAX_INPUT_TOKENS,
    ULTRA_CONSERVATIVE_WORDS,
    _translate_sync,
    clean_language_code,
    map_language,
    max_input_tokens_for,
    smart_truncate,
    truncation_candidates,
)


# --- language code normalization --------------------------------------------

@pytest.mark.parametrize(
    "supplied,expected",
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("es", "es"),
        ("ES", "es"),
        ("  es  ", "es"),
        ("__label__es", "es"),          # raw fastText label
        ("es.something", "es"),         # fastText can append a suffix
        ("pt_BR", "pt"),                # underscore normalized, region dropped
        ("zh-cn", "zh"),                # alias
        ("jpn", "ja"),                  # three-letter alias
        ("english", "en"),              # spelled-out alias
        ("zzz", "zzz"),                 # unknown passes through unchanged
    ],
)
def test_clean_language_code(supplied, expected):
    assert clean_language_code(supplied) == expected


def test_map_language_falls_back_when_unmappable():
    assert map_language("es", SEAMLESS_LANG_MAPPING, "eng") == "spa"
    assert map_language(None, SEAMLESS_LANG_MAPPING, "eng") == "eng"
    assert map_language("zzz", SEAMLESS_LANG_MAPPING, "eng") == "eng"
    assert map_language("es", STANDARD_LANG_MAPPING, "es") == "es"


def test_finnish_maps_to_the_seamless_token():
    """Regression: `fi` was missing, so Finnish was translated as another language."""
    assert map_language("fi", SEAMLESS_LANG_MAPPING, "eng") == "fin"


# --- input budget -----------------------------------------------------------

def test_input_budget_differs_per_backend():
    assert max_input_tokens_for(True) == SEAMLESS_MAX_INPUT_TOKENS == 400
    assert max_input_tokens_for(False) == STANDARD_MAX_INPUT_TOKENS == 350


# --- smart_truncate ---------------------------------------------------------

def test_smart_truncate_keeps_whole_sentences_within_budget():
    text = "One two three. Four five six. Seven eight nine."
    assert smart_truncate(text, 6) == "One two three. Four five six."


def test_smart_truncate_adds_the_missing_terminator():
    assert smart_truncate("One two three", 10) == "One two three."


def test_smart_truncate_preserves_an_existing_terminator():
    assert smart_truncate("One two three.", 10) == "One two three."


def test_smart_truncate_returns_none_when_no_sentence_fits():
    """The caller must fall through to word-boundary truncation."""
    assert smart_truncate("One two three four five.", 2) is None


# --- truncation_candidates --------------------------------------------------

def test_short_text_starts_with_the_full_text():
    candidates = truncation_candidates("alpha beta gamma", 350)
    assert [label for label, _ in candidates] == [
        "(full text)",
        "(hard truncated)",
        "(ultra-conservative)",
    ]
    assert candidates[0][1] == "alpha beta gamma"


def test_long_text_starts_with_smart_truncation_not_the_full_text():
    text = ". ".join(f"sentence {i} words here" for i in range(50))
    candidates = truncation_candidates(text, 20)
    labels = [label for label, _ in candidates]

    assert "(full text)" not in labels, "oversized input must never be sent whole"
    assert labels == ["(smart truncated)", "(hard truncated)", "(ultra-conservative)"]


def test_long_text_without_a_usable_sentence_skips_smart_truncation():
    text = " ".join(f"word{i}" for i in range(100))  # no sentence boundaries
    labels = [label for label, _ in truncation_candidates(text, 10)]
    assert labels == ["(hard truncated)", "(ultra-conservative)"]


def test_candidates_shrink_monotonically():
    text = " ".join(f"word{i}" for i in range(500))
    candidates = truncation_candidates(text, 350)

    assert len(candidates[-2][1].split()) == 350
    assert len(candidates[-1][1].split()) == ULTRA_CONSERVATIVE_WORDS


# --- backend call shapes ----------------------------------------------------

class RecordingTranslator:
    """A pipeline-shaped callable; `failures` leading calls raise."""

    def __init__(self, failures=0, result=None):
        self.failures = failures
        self.result = result if result is not None else [{"translation_text": "ok"}]
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        if len(self.calls) <= self.failures:
            raise RuntimeError("CUDA out of memory")
        return self.result


def test_seamless_pipeline_receives_mapped_language_pair():
    translator = RecordingTranslator()
    result = _translate_sync(translator, "Hola mundo", "es", "eng", "seamless-m4t-v2")

    assert result == [{"translation_text": "ok"}]
    [call] = translator.calls
    assert call["src_lang"] == "spa"
    assert call["tgt_lang"] == "eng"
    assert call["no_repeat_ngram_size"] == 3, "repetition guard must reach the model"


def test_helsinki_backend_is_called_without_a_language_pair():
    """The Helsinki pipeline rejects src_lang/tgt_lang keywords."""
    translator = RecordingTranslator()
    _translate_sync(translator, "Hola mundo", "es", "eng", "helsinki-nlp")

    [call] = translator.calls
    assert "src_lang" not in call
    assert "tgt_lang" not in call
    assert call["max_length"] >= 50


def test_standard_backend_with_a_source_language_targets_english():
    translator = RecordingTranslator()
    _translate_sync(translator, "Hola mundo", "es", "eng", "some-other-model")

    [call] = translator.calls
    assert call["src_lang"] == "es"
    assert call["tgt_lang"] == "en"


def test_standard_backend_without_a_source_language_sends_text_only():
    translator = RecordingTranslator()
    _translate_sync(translator, "Hola mundo", None, "eng", "some-other-model")

    [call] = translator.calls
    assert "src_lang" not in call


def test_seamless_direct_model_path_decodes_the_full_sequence():
    """The production path uses model.generate() and batch_decode()."""
    decoded = []

    class Processor:
        def __call__(self, text, src_lang, return_tensors):
            decoded.append(("processor", text, src_lang))
            return {"input_ids": [[1, 2, 3]]}

        def batch_decode(self, tokens, skip_special_tokens):
            decoded.append(("decode", tuple(tokens[0]), skip_special_tokens))
            return ["full decoded sentence"]

    class Model:
        def parameters(self):
            return iter([SimpleNamespace(device="cpu")])

        def generate(self, **kwargs):
            decoded.append(("generate", kwargs["tgt_lang"]))
            return [[4, 5, 6]]

    translator = SimpleNamespace(model=Model(), tokenizer=Processor())
    result = _translate_sync(translator, "Hola mundo", "es", "eng", "seamless-m4t-v2")

    assert result == [{"translation_text": "full decoded sentence"}]
    assert ("processor", "Hola mundo", "spa") in decoded
    assert ("generate", "eng") in decoded
    assert ("decode", (4, 5, 6), True) in decoded


# --- escalation -------------------------------------------------------------

def test_a_failed_attempt_escalates_to_the_next_candidate():
    translator = RecordingTranslator(failures=1)
    result = _translate_sync(translator, "alpha beta gamma", "es", "eng", "seamless")

    assert result == [{"translation_text": "ok"}]
    assert len(translator.calls) == 2, "retried after the first failure"


def test_seamless_retries_the_pipeline_within_a_single_attempt():
    """Each seamless attempt costs two model calls, not one.

    `_attempt_translation` tries the direct model/processor path, then retries
    through the pipeline with an explicit language pair before giving up on that
    candidate. Worth knowing when counting GPU work.
    """
    translator = RecordingTranslator(failures=1)
    _translate_sync(translator, "alpha beta", "es", "eng", "seamless")

    assert len(translator.calls) == 2
    assert translator.calls[0]["text"] == translator.calls[1]["text"]


def test_every_candidate_failing_falls_through_to_the_last_resort():
    long_text = " ".join(f"word{i}" for i in range(500))
    # No sentence boundaries, so smart truncation is skipped: two candidates
    # remain, each costing two seamless calls.
    translator = RecordingTranslator(failures=4)
    result = _translate_sync(translator, long_text, "es", "eng", "seamless")

    assert result == [{"translation_text": "ok"}]
    assert len(translator.calls) == 5
    assert [len(call["text"].split()) for call in translator.calls] == [
        400, 400, ULTRA_CONSERVATIVE_WORDS, ULTRA_CONSERVATIVE_WORDS, LAST_RESORT_WORDS
    ]
    last = translator.calls[-1]
    assert last["src_lang"] == "spa"
    assert last["tgt_lang"] == "eng"
    assert "no_repeat_ngram_size" not in last, "last resort uses model defaults"


def test_the_last_resort_for_a_standard_backend_caps_max_length():
    translator = RecordingTranslator(failures=3)
    _translate_sync(translator, "alpha beta gamma", "es", "eng", "other-model")

    assert translator.calls[-1]["max_length"] == 100


def test_returning_none_when_even_the_last_resort_fails():
    translator = RecordingTranslator(failures=99)
    assert _translate_sync(translator, "alpha beta", "es", "eng", "seamless") is None


def test_a_missing_model_name_is_treated_as_a_standard_backend():
    translator = RecordingTranslator()
    _translate_sync(translator, "Hola mundo", "es", "eng", None)

    [call] = translator.calls
    assert "max_length" in call, "standard backends pass max_length"
