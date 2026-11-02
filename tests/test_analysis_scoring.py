"""Covers the editorial scoring and language/classification helpers in
`src/endpoints/analysis.py`.

These decide what Monitor stores as newsworthiness, urgency, language and
taxonomy labels. They are pure, so they are testable without a model.
"""

from types import SimpleNamespace

import pytest

from src.endpoints.analysis import (
    EDITORIAL_SCORE_MAP,
    URGENCY_CATEGORY_WEIGHTS,
    _classification_response,
    _compute_urgency_signals,
    _extract_detected_language,
    _final_urgency,
    _record_analyze_step_timing,
    _record_shared_editorial_timing,
    _scores_from_editorial_result,
    _should_run_content_classification,
)
from src.models import UnifiedAnalysisItemTimings


# --- _compute_urgency_signals ----------------------------------------------

def test_no_text_scores_zero_urgency():
    assert _compute_urgency_signals("") == 0.0
    assert _compute_urgency_signals(None) == 0.0


def test_ordinary_copy_scores_zero_urgency():
    assert _compute_urgency_signals("Council approves the annual budget") == 0.0


@pytest.mark.parametrize(
    "category,text",
    [
        ("evacuation", "Residents evacuated from the area"),
        ("hazardous_agent", "A radioactive leak was found"),
        ("active_violence", "Reports of a shooting downtown"),
        ("fire", "Crews battle a blaze"),
    ],
)
def test_each_signal_category_contributes_its_weight(category, text):
    assert _compute_urgency_signals(text) == URGENCY_CATEGORY_WEIGHTS[category]


def test_a_category_contributes_once_however_many_terms_match():
    one = _compute_urgency_signals("A shooting was reported")
    many = _compute_urgency_signals("A shooting: the gunman opened gunfire, armed and barricaded")

    assert one == many == URGENCY_CATEGORY_WEIGHTS["active_violence"], (
        "repeating terms from one category must not inflate the score"
    )


def test_separate_categories_accumulate():
    combined = _compute_urgency_signals("Residents evacuated after a shooting")

    assert combined == (
        URGENCY_CATEGORY_WEIGHTS["evacuation"] + URGENCY_CATEGORY_WEIGHTS["active_violence"]
    )


def test_urgency_is_capped_at_ten():
    text = (
        "Residents evacuated amid tear gas and a shooting, with casualties, "
        "a fire, an earthquake, paramedics on scene, breaking news"
    )
    assert _compute_urgency_signals(text) == 10.0


def test_two_word_signals_match_across_punctuation():
    # Bigrams are built from the punctuation-normalized text, so "tear gas"
    # still matches when punctuation sits beside it.
    assert _compute_urgency_signals("Officers used (tear gas).") == (
        URGENCY_CATEGORY_WEIGHTS["hazardous_agent"]
    )


def test_the_three_word_evacuation_signal_can_never_match(monkeypatch):
    """Documents a bug rather than asserting desired behaviour.

    `_compute_urgency_signals` builds unigrams and bigrams only, but the
    evacuation category contains the three-word term "shelter in place". That
    term is therefore dead: a shelter-in-place order with no other signal scores
    zero urgency, which is close to the worst case for this function to miss.

    Filed separately. Pinned here so a fix has a target.
    """
    from src.endpoints.analysis import URGENCY_SIGNAL_CATEGORIES

    assert "shelter in place" in URGENCY_SIGNAL_CATEGORIES["evacuation"]
    assert _compute_urgency_signals("Officials ordered shelter in place") == 0.0, (
        "the three-word term now matches; the bug is fixed, update this test"
    )


def test_signals_are_case_insensitive():
    assert _compute_urgency_signals("EVACUATED") == URGENCY_CATEGORY_WEIGHTS["evacuation"]


def test_a_term_inside_a_longer_word_does_not_match():
    # Tokenized matching, not substring: "armada" must not trigger "armed".
    assert _compute_urgency_signals("The armada set sail") == 0.0


# --- _final_urgency ---------------------------------------------------------

def test_final_urgency_takes_whichever_signal_is_stronger():
    # The classifier is not reliably calibrated, so rules can only raise.
    assert _final_urgency("Council approves budget", 7.0) == 7.0
    assert _final_urgency("Residents evacuated", 1.0) == URGENCY_CATEGORY_WEIGHTS["evacuation"]


def test_final_urgency_is_rounded():
    assert _final_urgency("", 1.23456) == 1.23


# --- _scores_from_editorial_result -----------------------------------------

def editorial(labels, scores):
    return SimpleNamespace(full_result={"labels": labels, "scores": scores})


def test_editorial_scores_are_zero_without_a_usable_result():
    assert _scores_from_editorial_result(SimpleNamespace(full_result=None)) == (0.0, 0.0, None)
    assert _scores_from_editorial_result(SimpleNamespace(full_result={})) == (0.0, 0.0, None)
    assert _scores_from_editorial_result(editorial([], [])) == (0.0, 0.0, None)


def test_a_confident_breaking_story_scores_near_the_maximum():
    newsworthiness, urgency, label = _scores_from_editorial_result(
        editorial(["major breaking story", "not newsworthy"], [0.99, 0.01])
    )

    assert label == "major breaking story"
    assert newsworthiness > 9.0
    assert urgency > 9.0


def test_a_confident_non_story_scores_zero():
    newsworthiness, urgency, label = _scores_from_editorial_result(
        editorial(["not newsworthy", "major breaking story"], [0.99, 0.01])
    )

    assert label == "not newsworthy"
    assert newsworthiness < 1.0
    assert urgency < 1.0


def test_scores_are_a_weighted_blend_not_just_the_top_label():
    blended, _, label = _scores_from_editorial_result(
        editorial(["major breaking story", "not newsworthy"], [0.5, 0.5])
    )

    assert label == "major breaking story"
    top_only = EDITORIAL_SCORE_MAP["major breaking story"][0]
    assert blended < top_only, "an even split must pull the score down from the top label"


def test_unknown_labels_contribute_nothing_but_still_dilute():
    newsworthiness, _, label = _scores_from_editorial_result(
        editorial(["something new", "major breaking story"], [0.6, 0.4])
    )

    assert label == "something new"
    assert 0.0 < newsworthiness < EDITORIAL_SCORE_MAP["major breaking story"][0]


def test_zero_weight_scores_return_the_top_label_with_no_scores():
    newsworthiness, urgency, label = _scores_from_editorial_result(
        editorial(["major breaking story"], [0.0])
    )

    assert (newsworthiness, urgency) == (0.0, 0.0)
    assert label == "major breaking story", "the label survives even with no usable weight"


# --- _extract_detected_language --------------------------------------------

@pytest.mark.parametrize(
    "supplied,expected",
    [
        (None, None),
        ("es", "es"),
        ("  es  ", "es"),
        ("", None),
        ("   ", None),
        ({"lang": "es"}, "es"),
        ({"language": "pt"}, "pt"),
        ({"code": "fr"}, "fr"),
        ({"languages": ["de", "en"]}, "de"),
        ({"lang": "  es  "}, "es"),
        ({}, None),
        ({"lang": ""}, None),
        ({"lang": 42}, None),
        ({"languages": []}, None),
        ({"languages": [42]}, None),
        ({"languages": ["  "]}, None),
        (42, None),
        ([], None),
    ],
)
def test_detected_language_normalization(supplied, expected):
    assert _extract_detected_language(supplied) == expected


def test_lang_wins_over_language_and_code():
    assert _extract_detected_language({"lang": "es", "language": "pt", "code": "fr"}) == "es"


def test_an_explicit_key_wins_over_the_languages_list():
    assert _extract_detected_language({"code": "fr", "languages": ["de"]}) == "fr"


# --- _should_run_content_classification ------------------------------------

def test_classification_runs_when_no_editorial_label_is_available():
    assert _should_run_content_classification(None) is True
    assert _should_run_content_classification("") is True


@pytest.mark.parametrize("label", ["niche or low-priority item", "not newsworthy"])
def test_classification_is_skipped_for_low_value_stories(label):
    assert _should_run_content_classification(label) is False


@pytest.mark.parametrize(
    "label",
    ["major breaking story", "important story but not breaking", "routine publishable update"],
)
def test_classification_runs_for_publishable_stories(label):
    assert _should_run_content_classification(label) is True


def test_every_skipped_label_exists_in_the_editorial_map():
    # A typo here would silently re-enable expensive tagging for junk.
    for label in ("niche or low-priority item", "not newsworthy"):
        assert label in EDITORIAL_SCORE_MAP


# --- _classification_response ----------------------------------------------

def state(min_score=0.5, min_margin=0.1):
    return SimpleNamespace(config={"MIN_SCORE": min_score, "MIN_MARGIN": min_margin})


def test_classification_response_returns_the_top_label():
    response = _classification_response(
        {"labels": ["politics", "sports"], "scores": [0.9, 0.05]}, state()
    )

    assert response.label == "politics"
    assert response.score == pytest.approx(0.9)


@pytest.mark.parametrize("result", [{}, {"labels": []}, {"scores": []}, {"labels": ["a"]}])
def test_classification_response_degrades_on_an_unusable_result(result):
    response = _classification_response(result, state())

    assert response.label == "error"
    assert response.score == 0.0


def test_classification_response_is_uncertain_below_the_score_floor():
    response = _classification_response(
        {"labels": ["politics", "sports"], "scores": [0.4, 0.01]}, state(min_score=0.5)
    )

    assert response.label == "uncertain"
    assert response.score == pytest.approx(0.4), "the real score is still reported"


def test_classification_response_is_uncertain_on_a_narrow_margin():
    response = _classification_response(
        {"labels": ["politics", "sports"], "scores": [0.9, 0.89]}, state(min_margin=0.1)
    )
    assert response.label == "uncertain"


def test_classification_response_handles_a_lone_score():
    response = _classification_response({"labels": ["politics"], "scores": [0.9]}, state())
    assert response.label == "politics"


# --- timing recorders -------------------------------------------------------

def test_step_timing_accumulates_into_the_batch_total_and_the_item():
    totals = {"language_detection_ms": 0.0}
    timings = UnifiedAnalysisItemTimings(total_ms=0.0)

    _record_analyze_step_timing(
        "language_detection_ms", "language_detection", 0.0, totals, timings
    )

    assert totals["language_detection_ms"] > 0.0
    assert timings.language_detection_ms is not None


def test_step_timing_tolerates_an_item_without_timings():
    totals = {"language_detection_ms": 0.0}

    _record_analyze_step_timing("language_detection_ms", "language_detection", 0.0, totals, None)

    assert totals["language_detection_ms"] > 0.0


def test_shared_editorial_timing_is_split_between_both_scores():
    totals = {"newsworthiness_ms": 0.0, "urgency_ms": 0.0}
    timings = UnifiedAnalysisItemTimings(total_ms=0.0)

    _record_shared_editorial_timing(0.0, totals, timings)

    assert totals["newsworthiness_ms"] == pytest.approx(totals["urgency_ms"])
    assert timings.newsworthiness_ms == pytest.approx(timings.urgency_ms)


def test_shared_editorial_timing_tolerates_an_item_without_timings():
    totals = {"newsworthiness_ms": 0.0, "urgency_ms": 0.0}

    _record_shared_editorial_timing(0.0, totals, None)

    assert totals["newsworthiness_ms"] > 0.0
