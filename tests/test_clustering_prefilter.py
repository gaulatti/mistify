from src.endpoints.clustering import (
    _filter_candidate_posts,
    _has_conflicting_event_anchors,
)
from src.models import PostData
import numpy as np


class DummyEmbedder:
    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0] for _ in texts])


def _post(post_id: int, content: str, embedding=None) -> PostData:
    return PostData(
        id=post_id,
        uuid=f"post-{post_id}",
        content=content,
        source="rss",
        createdAt="2026-07-17T00:00:00Z",
        hash=f"hash-{post_id}",
        embeddings=embedding,
    )


def test_prefilter_stores_computed_candidate_embeddings():
    main = _post(1, "Main post", [1.0, 0.0])
    candidate = _post(2, "Similar post")

    filtered = _filter_candidate_posts(
        main,
        [candidate],
        DummyEmbedder(),
        min_similarity=0.25,
        max_candidates=10,
    )

    assert filtered == [candidate]
    assert candidate.embeddings == [1.0, 0.0]


def test_prefilter_rejects_candidate_outside_event_window():
    main = _post(1, "August distribution", [1.0, 0.0])
    candidate = _post(2, "January distribution", [1.0, 0.0])
    candidate.createdAt = "2026-01-17T00:00:00Z"

    assert _filter_candidate_posts(main, [candidate], DummyEmbedder()) == []


def test_prefilter_rejects_different_lottery_editions():
    main = _post(
        1,
        "Quiniela Nacional: result of the Vespertina draw today",
        [1.0, 0.0],
    )
    candidate = _post(
        2,
        "Quiniela de Santa Fe: result of the Vespertina draw today",
        [1.0, 0.0],
    )

    assert _filter_candidate_posts(main, [candidate], DummyEmbedder()) == []


def test_anchor_guard_rejects_different_reporting_months():
    assert _has_conflicting_event_anchors(
        "Accelerate declares January 2026 cash distributions",
        "Accelerate announces August 2026 distributions",
    )


def test_anchor_guard_allows_same_breaking_event():
    assert not _has_conflicting_event_anchors(
        "Israeli minister calls for killing 30 to 40 Palestinians in Gaza nightly",
        "Ben Gvir calls for killing between 30 and 40 people every night in Gaza",
    )
