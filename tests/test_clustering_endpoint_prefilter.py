"""Covers the candidate pre-filter in `src/endpoints/clustering.py`.

This runs before graph clustering and decides which posts are even eligible to
merge into an event. Dropping the wrong candidate loses a real event; keeping the
wrong one produces a false merge, which is the more visible failure.
"""

from types import SimpleNamespace

import pytest

from src.endpoints.clustering import (
    _cosine_similarity,
    _filter_candidate_posts,
    _has_conflicting_event_anchors,
    _normalise_anchor_text,
    _parse_post_time,
)
from src.models import PostData


def post(post_id, content, **overrides):
    fields = {
        "id": post_id,
        "content": content,
        "source": "rss",
        "createdAt": "2026-08-29T12:00:00Z",
        "hash": f"hash-{post_id}",
        "uuid": f"0000000-0000-0000-0000-{post_id:012d}",
    }
    fields.update(overrides)
    return PostData(**fields)


class FakeEmbedder:
    """Returns one unit vector per text, keyed by content."""

    def __init__(self, vectors=None, error=None):
        self.vectors = vectors or {}
        self.error = error
        self.calls = []

    def encode(self, texts, **_kwargs):
        self.calls.append(list(texts))
        if self.error is not None:
            raise self.error
        rows = [self.vectors.get(text, [0.0, 0.0, 1.0]) for text in texts]
        return SimpleNamespace(tolist=lambda: rows)


# --- _normalise_anchor_text -------------------------------------------------

def test_anchor_text_is_stripped_to_ascii_lowercase_tokens():
    assert _normalise_anchor_text("Córdoba — Quiniela Nocturna!") == "cordoba quiniela nocturna"
    assert _normalise_anchor_text("") == ""
    assert _normalise_anchor_text(None) == ""


def test_anchor_text_keeps_digits():
    assert _normalise_anchor_text("Sorteo 2026 #4521") == "sorteo 2026 4521"


# --- _parse_post_time -------------------------------------------------------

def test_post_time_prefers_posted_at_then_created_then_received():
    assert _parse_post_time(
        post(1, "x", posted_at="2026-08-29T10:00:00Z", received_at="2026-08-29T11:00:00Z")
    ).hour == 10

    assert _parse_post_time(post(1, "x", createdAt="2026-08-29T09:00:00Z")).hour == 9

    only_received = post(1, "x", createdAt="", received_at="2026-08-29T08:00:00Z")
    assert _parse_post_time(only_received).hour == 8


def test_post_time_is_none_without_any_timestamp():
    assert _parse_post_time(post(1, "x", createdAt="")) is None


def test_post_time_rejects_an_unparseable_timestamp():
    assert _parse_post_time(post(1, "x", createdAt="last Tuesday")) is None


def test_naive_timestamps_are_treated_as_utc():
    parsed = _parse_post_time(post(1, "x", createdAt="2026-08-29T12:00:00"))
    assert parsed.tzinfo is not None
    assert parsed.hour == 12


def test_offset_timestamps_are_converted_to_utc():
    parsed = _parse_post_time(post(1, "x", createdAt="2026-08-29T08:00:00-04:00"))
    assert parsed.hour == 12


# --- _has_conflicting_event_anchors ----------------------------------------

def test_unrelated_copy_has_no_conflicting_anchors():
    assert _has_conflicting_event_anchors("Budget passes", "Council meets") is False


def test_different_lottery_editions_conflict():
    assert _has_conflicting_event_anchors(
        "Quiniela Nocturna resultados", "Quiniela Matutina resultados"
    ) is True


def test_the_same_lottery_edition_does_not_conflict():
    assert _has_conflicting_event_anchors(
        "Quiniela Nocturna resultados", "Quiniela Nocturna numeros"
    ) is False


def test_a_lottery_without_an_edition_does_not_conflict():
    assert _has_conflicting_event_anchors("Quiniela resultados", "Quiniela numeros") is False


def test_recurring_reports_from_different_months_conflict():
    assert _has_conflicting_event_anchors(
        "Earnings report for January", "Earnings report for February"
    ) is True


def test_recurring_reports_from_the_same_month_do_not_conflict():
    assert _has_conflicting_event_anchors(
        "Earnings report for January", "January earnings report revised"
    ) is False


def test_month_aliases_match_across_languages():
    # "enero" and "january" are the same month, so these must not conflict.
    assert _has_conflicting_event_anchors(
        "Earnings report enero", "Earnings report January"
    ) is False


def test_months_without_a_recurring_marker_do_not_conflict():
    # Two ordinary stories in different months are not a recurring series.
    assert _has_conflicting_event_anchors(
        "Protests in January", "Protests in February"
    ) is False


# --- _cosine_similarity -----------------------------------------------------

def test_cosine_similarity_of_identical_vectors_is_one():
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_is_zero_for_empty_or_zero_vectors():
    assert _cosine_similarity([], [1.0]) == 0.0
    assert _cosine_similarity([1.0], []) == 0.0
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert _cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0


# --- _filter_candidate_posts ------------------------------------------------

def test_no_candidates_returns_nothing():
    assert _filter_candidate_posts(post(1, "main"), [], FakeEmbedder()) == []


def test_duplicate_hashes_and_content_are_dropped():
    main = post(1, "Main story", embeddings=[1.0, 0.0, 0.0])
    same_hash = post(2, "Different wording", hash="hash-1")
    same_content = post(3, "  MAIN STORY  ")
    distinct = post(4, "Another angle", embeddings=[1.0, 0.0, 0.0])

    filtered = _filter_candidate_posts(main, [same_hash, same_content, distinct], FakeEmbedder())

    assert [candidate.id for candidate in filtered] == [4]


def test_repeated_candidates_are_deduplicated_against_each_other():
    main = post(1, "Main story", embeddings=[1.0, 0.0, 0.0])
    first = post(2, "Another angle", embeddings=[1.0, 0.0, 0.0])
    duplicate = post(2, "Another angle", embeddings=[1.0, 0.0, 0.0])

    filtered = _filter_candidate_posts(main, [first, duplicate], FakeEmbedder())

    assert len(filtered) == 1


def test_candidates_outside_the_age_window_are_dropped():
    main = post(1, "Main", createdAt="2026-08-29T12:00:00Z", embeddings=[1.0, 0.0, 0.0])
    recent = post(2, "Recent", createdAt="2026-08-29T20:00:00Z", embeddings=[1.0, 0.0, 0.0])
    stale = post(3, "Stale", createdAt="2026-08-01T12:00:00Z", embeddings=[1.0, 0.0, 0.0])

    filtered = _filter_candidate_posts(main, [recent, stale], FakeEmbedder(), max_age_hours=24)

    assert [candidate.id for candidate in filtered] == [2]


def test_candidates_without_a_timestamp_are_dropped_when_an_age_window_applies():
    main = post(1, "Main", embeddings=[1.0, 0.0, 0.0])
    undated = post(2, "Undated", createdAt="", embeddings=[1.0, 0.0, 0.0])

    assert _filter_candidate_posts(main, [undated], FakeEmbedder(), max_age_hours=24) == []


def test_disabling_the_age_window_keeps_undated_candidates():
    main = post(1, "Main", embeddings=[1.0, 0.0, 0.0])
    undated = post(2, "Undated", createdAt="", embeddings=[1.0, 0.0, 0.0])

    filtered = _filter_candidate_posts(main, [undated], FakeEmbedder(), max_age_hours=None)

    assert [candidate.id for candidate in filtered] == [2]


def test_conflicting_event_anchors_are_dropped_before_similarity():
    main = post(1, "Quiniela Nocturna resultados", embeddings=[1.0, 0.0, 0.0])
    other_edition = post(2, "Quiniela Matutina resultados", embeddings=[1.0, 0.0, 0.0])

    assert _filter_candidate_posts(main, [other_edition], FakeEmbedder()) == []


def test_without_a_main_embedding_candidates_pass_through_capped():
    main = post(1, "Main")
    candidates = [post(index, f"Angle {index}") for index in range(2, 8)]

    filtered = _filter_candidate_posts(main, candidates, FakeEmbedder(), max_candidates=3)

    assert len(filtered) == 3, "with no embedding to compare against, keep the first N"


def test_candidates_below_the_similarity_floor_are_dropped():
    main = post(1, "Main", embeddings=[1.0, 0.0, 0.0])
    similar = post(2, "Similar", embeddings=[1.0, 0.0, 0.0])
    unrelated = post(3, "Unrelated", embeddings=[0.0, 0.0, 1.0])

    filtered = _filter_candidate_posts(
        main, [similar, unrelated], FakeEmbedder(), min_similarity=0.3
    )

    assert [candidate.id for candidate in filtered] == [2]


def test_candidates_are_returned_most_similar_first():
    main = post(1, "Main", embeddings=[1.0, 0.0, 0.0])
    partial = post(2, "Partial", embeddings=[0.7071, 0.7071, 0.0])
    exact = post(3, "Exact", embeddings=[1.0, 0.0, 0.0])

    filtered = _filter_candidate_posts(main, [partial, exact], FakeEmbedder(), min_similarity=0.1)

    assert [candidate.id for candidate in filtered] == [3, 2]


def test_candidates_without_embeddings_are_embedded_on_the_fly():
    main = post(1, "Main", embeddings=[1.0, 0.0, 0.0])
    missing = post(2, "Needs embedding")
    embedder = FakeEmbedder(vectors={"Needs embedding": [1.0, 0.0, 0.0]})

    filtered = _filter_candidate_posts(main, [missing], embedder, min_similarity=0.1)

    assert embedder.calls == [["Needs embedding"]]
    assert [candidate.id for candidate in filtered] == [2]
    assert filtered[0].embeddings == [1.0, 0.0, 0.0], "the computed embedding is cached on the post"


def test_precomputed_embeddings_are_not_recomputed():
    main = post(1, "Main", embeddings=[1.0, 0.0, 0.0])
    already = post(2, "Already embedded", embeddings=[1.0, 0.0, 0.0])
    embedder = FakeEmbedder()

    _filter_candidate_posts(main, [already], embedder, min_similarity=0.1)

    assert embedder.calls == [], "an existing embedding must not be recomputed"


def test_an_embedding_failure_falls_back_to_the_unique_candidates():
    main = post(1, "Main", embeddings=[1.0, 0.0, 0.0])
    candidates = [post(index, f"Angle {index}") for index in range(2, 8)]
    embedder = FakeEmbedder(error=RuntimeError("cuda oom"))

    filtered = _filter_candidate_posts(main, candidates, embedder, max_candidates=3)

    assert len(filtered) == 3, "a failed pre-filter must not drop every candidate"


def test_the_candidate_cap_is_applied_after_ranking():
    main = post(1, "Main", embeddings=[1.0, 0.0, 0.0])
    candidates = [
        post(2, "Weak", embeddings=[0.6, 0.8, 0.0]),
        post(3, "Strong", embeddings=[1.0, 0.0, 0.0]),
        post(4, "Medium", embeddings=[0.9, 0.4359, 0.0]),
    ]

    filtered = _filter_candidate_posts(
        main, candidates, FakeEmbedder(), min_similarity=0.1, max_candidates=2
    )

    assert [candidate.id for candidate in filtered] == [3, 4], (
        "the cap must keep the strongest matches, not the first N"
    )
