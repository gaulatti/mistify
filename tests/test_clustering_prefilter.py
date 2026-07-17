from src.endpoints.clustering import _filter_candidate_posts
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
