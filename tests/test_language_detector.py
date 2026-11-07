"""Detector selection, GlotLID label normalisation, and label-space alignment.

The property worth pinning is that every label a detector can emit resolves to
exactly one explicit outcome: supported, a supported relative, or declined. A
confident detection of a language the translator cannot handle must never
silently choose a different language.
"""

import os
from types import SimpleNamespace

import pytest

from src.endpoints.translation import (
    SUPPORTED_TRANSLATION_LANGUAGES,
    TRANSLATION_LANGUAGE_ALIASES,
    UNSUPPORTED_TRANSLATION_FALLBACKS,
    UNTRANSLATABLE_LANGUAGES,
    _normalize_translation_language,
)
from src.helpers import language_detector as ld

SCRIPTS = ("Latn", "Cyrl", "Arab", "Hant", "Deva")

# Labels that GlotLID emits for the audited misclassifications (issue #33).
AUDITED = ["kin_Latn", "uzn_Cyrl", "uzn_Latn", "hrv_Latn", "srp_Cyrl", "srp_Latn", "bos_Latn", "som_Latn", "fra_Latn", "scn_Latn"]


def label_space():
    """Every language code the service already names, in every GlotLID label shape."""
    codes = (
        set(SUPPORTED_TRANSLATION_LANGUAGES)
        | set(TRANSLATION_LANGUAGE_ALIASES)
        | set(UNSUPPORTED_TRANSLATION_FALLBACKS)
        | set(UNTRANSLATABLE_LANGUAGES)
        | {"xyz", "qqq", "zzz"}  # codes the translator knows nothing about
    )
    codes = {code.split("-", 1)[0] for code in codes}
    labels = {f"__label__{code}_{script}" for code in codes for script in SCRIPTS}
    labels |= {f"__label__{code}" for code in codes}
    labels |= {f"__label__{label}" for label in AUDITED}
    return sorted(labels)


@pytest.mark.parametrize("label", label_space())
def test_every_label_resolves_to_one_explicit_outcome(label):
    resolution = ld.resolve_label(label)

    assert resolution.kind in {"supported", "relative", "declined"}
    if resolution.kind in {"supported", "relative"}:
        assert resolution.language in SUPPORTED_TRANSLATION_LANGUAGES
    else:
        # Declined means the translator is never asked to guess another language.
        assert resolution.language not in SUPPORTED_TRANSLATION_LANGUAGES


def test_audited_labels_resolve_as_the_issue_requires():
    kinds = {label: ld.resolve_label(label).kind for label in AUDITED}

    assert kinds["kin_Latn"] == "declined"      # Kinyarwanda: kept, never mistranslated
    assert kinds["scn_Latn"] == "declined"      # Sicilian: explicit, not a silent fallback
    assert kinds["fra_Latn"] == "supported"
    assert kinds["uzn_Cyrl"] == "supported" and ld.resolve_label("uzn_Cyrl").language == "uzn"
    assert ld.resolve_label("cnr_Latn") == ld.Resolution("relative", "sr")


def test_resolution_agrees_with_the_translation_endpoint_normaliser():
    for label in label_space():
        resolution = ld.resolve_label(label)
        assert resolution.language == (_normalize_translation_language(ld.base_language(label)) or ld.base_language(label).lower())


def test_base_language_strips_prefix_and_script():
    assert ld.base_language("__label__uzn_Cyrl") == "uzn"
    assert ld.base_language("__label__en") == "en"


def test_detector_settings_select_model_and_reject_unknown_names():
    assert ld.detector_settings("lid176", {}) == ("lid.176.bin", "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
    assert ld.detector_settings("lid176", {"FASTTEXT_MODEL_PATH": "/m/x.bin"})[0] == "/m/x.bin"
    assert ld.detector_settings("glotlid", {}) == ("glotlid_v3.bin", ld.GLOTLID_MODEL_URL)
    assert ld.detector_settings("glotlid", {"GLOTLID_MODEL_PATH": "/m/g.bin"})[0] == "/m/g.bin"
    with pytest.raises(ValueError, match="Unknown LANGUAGE_DETECTOR 'nope'"):
        ld.detector_settings("nope", {})


def test_script_variants_of_one_language_merge_and_keep_top_k_order():
    calls = []

    def predict(text, k):
        calls.append(k)
        return (
            ["__label__srp_Cyrl", "__label__hrv_Latn", "__label__srp_Latn", "__label__bos_Latn"],
            [0.30, 0.35, 0.20, 0.05],
        )

    detector = ld.ScriptMergingDetector(SimpleNamespace(predict=predict, get_labels=lambda: ["__label__srp_Cyrl"]))
    labels, probs = detector.predict("tekst", k=2)

    assert labels == ("__label__srp", "__label__hrv")
    assert probs == pytest.approx((0.50, 0.35))
    assert calls == [6]  # overfetched so merging cannot starve top-k
    assert detector.get_labels() == ["__label__srp_Cyrl"]


def test_merged_labels_pass_through_the_translation_normaliser():
    detector = ld.ScriptMergingDetector(SimpleNamespace(predict=lambda t, k: (["__label__uzn_Cyrl"], [0.9])))
    labels, _ = detector.predict("matn", k=1)

    assert _normalize_translation_language(labels[0]) == "uzn"
    assert "uzn" in SUPPORTED_TRANSLATION_LANGUAGES


def test_k_below_one_still_returns_the_top_candidate():
    detector = ld.ScriptMergingDetector(SimpleNamespace(predict=lambda t, k: (["__label__eng_Latn"], [0.9])))
    assert detector.predict("hello", k=0)[0] == ("__label__eng",)


@pytest.mark.slow
def test_full_glotlid_label_space_is_total():
    """Runs only against a downloaded model; CI never downloads it (docs: language detector)."""
    path = os.environ.get("GLOTLID_MODEL_PATH")
    if not path or not os.path.exists(path):
        pytest.skip("GLOTLID_MODEL_PATH is not set to a downloaded model")
    import fasttext

    for label in fasttext.load_model(path).get_labels():
        resolution = ld.resolve_label(label)
        assert resolution.kind in {"supported", "relative", "declined"}
