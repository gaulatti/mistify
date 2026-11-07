"""Covers `cluster_texts`, the `/cluster` endpoint in `src/endpoints/clustering.py`.

The endpoint pre-filters candidates, assembles the clustering configuration and
inputs, runs `_cluster_sync` under the clustering lock with a timeout, and maps
the result back onto the main post's group. `_cluster_sync` is replaced by a
recorder so these tests pin the endpoint's own contract: what reaches the
clustering layer and what the caller gets back, including the degraded paths.
"""

import asyncio
import time
from types import SimpleNamespace
from typing import List, Optional

import pytest
from fastapi import HTTPException

from src import metrics
from src.endpoints import clustering as endpoint
from src.models import CategoryRelation, ClusterGroup, PostData

SIMILAR = [1.0, 0.0, 0.0]
DISTANT = [0.0, 1.0, 0.0]


class Post(PostData):
    """PostData plus the optional `primary_entities` the endpoint probes for."""

    primary_entities: Optional[List[str]] = None


def post(post_id, content=None, embeddings=SIMILAR, created="2026-08-29T12:00:00Z", **extra):
    return Post(
        id=post_id,
        content=content or f"content {post_id}",
        source=extra.pop("source", "rss"),
        createdAt=created,
        hash=f"hash-{post_id}",
        uuid=f"0000000-0000-0000-0000-{post_id:012d}",
        embeddings=embeddings,
        **extra,
    )


def make_state(**config):
    cfg = {"TIMEOUT": 5}
    cfg.update(config)
    return SimpleNamespace(
        embedder=object(),
        nlp=object(),
        classifier=object(),
        config=cfg,
        clustering_lock=asyncio.Lock(),
        clustering_pool=None,
    )


def request_for(state):
    return SimpleNamespace(state=SimpleNamespace(app_state=state))


class SyncRecorder:
    """Stands in for `_cluster_sync`; records its inputs and returns `result`."""

    def __init__(self, result=None, error=None, delay=0.0, state=None):
        self.result = result
        self.error = error
        self.delay = delay
        self.state = state
        self.calls = []
        self.lock_was_held = None

    def __call__(self, texts, nlp, embedder, classifier, config, debug):
        self.calls.append(SimpleNamespace(
            texts=list(texts), nlp=nlp, embedder=embedder,
            classifier=classifier, config=config, debug=debug,
        ))
        if self.state is not None:
            self.lock_was_held = self.state.clustering_lock.locked()
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.result


def group(group_id, indices, **fields):
    return ClusterGroup(
        group_id=group_id,
        texts=[f"t{i}" for i in indices],
        indices=list(indices),
        size=len(indices),
        **fields,
    )


@pytest.fixture
def recorder(monkeypatch):
    rec = SyncRecorder(result={"groups": [group(7, [0, 1])]})
    monkeypatch.setattr(endpoint, "_cluster_sync", rec)
    return rec


def failures(failure_type):
    return metrics.OPERATION_FAILURES_TOTAL.labels(
        operation="cluster", failure_type=failure_type
    )._value.get()


# --- model availability -----------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["embedder", "nlp"])
async def test_missing_model_is_a_503(missing, recorder):
    state = make_state()
    setattr(state, missing, None)

    with pytest.raises(HTTPException) as caught:
        await endpoint.cluster_texts(post(1), request_for(state))

    assert caught.value.status_code == 503
    assert caught.value.detail == "Clustering models not available"
    assert recorder.calls == []


# --- short-circuit paths ----------------------------------------------------

@pytest.mark.asyncio
async def test_a_post_without_candidates_is_its_own_cluster(recorder):
    main = post(1, "main story")

    response = await endpoint.cluster_texts(main, request_for(make_state()))

    assert recorder.calls == []
    assert (response.total_posts, response.total_groups) == (1, 1)
    assert response.processing_time == 0.0
    assert response.group.group_id == 0 and response.group.size == 1
    assert response.group.avg_similarity == 1.0
    assert [(p.id, p.hash, p.content) for p in response.group.posts] == [(1, "hash-1", "main story")]


@pytest.mark.asyncio
async def test_candidates_dropped_by_the_prefilter_never_reach_clustering(recorder):
    main = post(1, similarPosts=[post(2, embeddings=DISTANT)])

    response = await endpoint.cluster_texts(main, request_for(make_state()))

    assert recorder.calls == []
    assert [p.id for p in response.group.posts] == [1]


# --- what reaches the clustering layer --------------------------------------

@pytest.mark.asyncio
async def test_main_post_comes_first_followed_by_the_surviving_candidates(recorder):
    state = make_state()
    main = post(1, "main", similarPosts=[
        post(2, "near one"), post(3, "far away", embeddings=DISTANT), post(4, "near two"),
    ])

    await endpoint.cluster_texts(main, request_for(state))

    call = recorder.calls[0]
    assert call.texts == ["main", "near one", "near two"]
    assert (call.nlp, call.embedder, call.classifier) == (state.nlp, state.embedder, state.classifier)
    assert call.debug is False


@pytest.mark.asyncio
async def test_clustering_runs_under_the_clustering_lock(monkeypatch):
    state = make_state()
    rec = SyncRecorder(result={"groups": [group(0, [0, 1])]}, state=state)
    monkeypatch.setattr(endpoint, "_cluster_sync", rec)

    await endpoint.cluster_texts(post(1, similarPosts=[post(2)]), request_for(state))

    assert rec.lock_was_held is True
    assert state.clustering_lock.locked() is False


@pytest.mark.asyncio
async def test_default_thresholds_are_used_when_config_does_not_override_them(recorder):
    await endpoint.cluster_texts(post(1, similarPosts=[post(2)]), request_for(make_state()))

    config = recorder.calls[0].config
    assert config["similarity_entity"] == 0.35
    assert config["similarity_global"] == 0.50
    assert config["avg_similarity_min"] == 0.40
    assert config["big_community_size"] == 25
    assert config["entity_context_weight"] == 0.20
    assert config["min_shared_entities"] == 1
    assert config["topic_strict_mode"] is False
    assert config["domain_filtering"] is False
    assert config["event_specific_mode"] is True


@pytest.mark.asyncio
async def test_thresholds_come_from_application_config(recorder):
    state = make_state(
        CLUSTERING_SIM_ENTITY=0.41, CLUSTERING_SIM_GLOBAL=0.62, CLUSTERING_AVG_SIM_MIN=0.33
    )

    await endpoint.cluster_texts(post(1, similarPosts=[post(2)]), request_for(state))

    config = recorder.calls[0].config
    assert (config["similarity_entity"], config["similarity_global"], config["avg_similarity_min"]) == (0.41, 0.62, 0.33)


@pytest.mark.asyncio
async def test_prefilter_bounds_come_from_application_config(recorder):
    candidates = [post(10 + i, f"story {i}") for i in range(4)]
    state = make_state(CLUSTERING_MAX_CANDIDATES=2)

    await endpoint.cluster_texts(post(1, "main", similarPosts=candidates), request_for(state))

    assert len(recorder.calls[0].texts) == 3  # main + two candidates


@pytest.mark.asyncio
async def test_event_window_limits_candidate_age_except_for_youtube(recorder):
    old = "2026-08-26T12:00:00Z"  # three days before the main post

    await endpoint.cluster_texts(
        post(1, similarPosts=[post(2, created=old)]), request_for(make_state())
    )
    assert recorder.calls == []

    await endpoint.cluster_texts(
        post(1, source="youtube", similarPosts=[post(2, created=old)]), request_for(make_state())
    )
    assert len(recorder.calls) == 1

    await endpoint.cluster_texts(
        post(1, similarPosts=[post(2, created=old)]),
        request_for(make_state(CLUSTERING_EVENT_WINDOW_HOURS=96)),
    )
    assert len(recorder.calls) == 2


@pytest.mark.asyncio
async def test_complete_primary_entities_are_forwarded_as_sets(recorder):
    main = post(1, primary_entities=["Alice", "Acme"], similarPosts=[
        post(2, primary_entities=["Alice"]),
    ])

    await endpoint.cluster_texts(main, request_for(make_state()))

    assert recorder.calls[0].config["precomputed_entities"] == [{"Alice", "Acme"}, {"Alice"}]


@pytest.mark.asyncio
async def test_incomplete_primary_entities_are_not_forwarded(recorder):
    main = post(1, primary_entities=["Alice"], similarPosts=[post(2)])

    await endpoint.cluster_texts(main, request_for(make_state()))

    assert recorder.calls[0].config["precomputed_entities"] is None


@pytest.mark.asyncio
async def test_embeddings_are_forwarded_and_gaps_are_left_for_backfill(recorder):
    main = post(1, similarPosts=[post(2, embeddings=SIMILAR)])

    await endpoint.cluster_texts(main, request_for(make_state()))

    assert recorder.calls[0].config["precomputed_embeddings"] == [SIMILAR, SIMILAR]


@pytest.mark.asyncio
async def test_no_embeddings_anywhere_forwards_none(recorder):
    # Without a main embedding the pre-filter keeps candidates unscored.
    main = post(1, embeddings=None, similarPosts=[post(2, embeddings=None)])

    await endpoint.cluster_texts(main, request_for(make_state()))

    assert recorder.calls[0].config["precomputed_embeddings"] is None


@pytest.mark.asyncio
async def test_partial_embeddings_forward_none_for_the_missing_posts(recorder):
    main = post(1, embeddings=None, similarPosts=[post(2, embeddings=SIMILAR)])

    await endpoint.cluster_texts(main, request_for(make_state()))

    assert recorder.calls[0].config["precomputed_embeddings"] == [None, SIMILAR]


@pytest.mark.asyncio
async def test_category_names_are_forwarded_only_when_any_post_has_them(recorder):
    tagged = [CategoryRelation(id=1, slug="sports", name="Sports", Tagging={})]
    await endpoint.cluster_texts(post(1, similarPosts=[post(2)]), request_for(make_state()))
    assert recorder.calls[0].config["precomputed_categories"] is None

    main = post(1, categories_relation=tagged, similarPosts=[post(2)])
    await endpoint.cluster_texts(main, request_for(make_state()))
    assert recorder.calls[1].config["precomputed_categories"] == [["Sports"], []]


# --- response assembly ------------------------------------------------------

@pytest.mark.asyncio
async def test_response_is_the_main_posts_group_with_its_members(monkeypatch):
    result = {
        "groups": [
            group(3, [2], primary_topic="sports"),
            group(5, [0, 1], primary_topic="politics", primary_entities=["Alice"], avg_similarity=0.87),
        ],
        "debug_info": {"edges": 1},
    }
    monkeypatch.setattr(endpoint, "_cluster_sync", SyncRecorder(result=result))
    main = post(1, "main", similarPosts=[post(2, "second"), post(3, "third")])

    response = await endpoint.cluster_texts(main, request_for(make_state()))

    assert (response.total_posts, response.total_groups) == (3, 2)
    assert response.group.group_id == 5 and response.group.size == 2
    assert [(p.id, p.hash, p.content) for p in response.group.posts] == [
        (1, "hash-1", "main"), (2, "hash-2", "second"),
    ]
    assert response.group.primary_topic == "politics"
    assert response.group.primary_entities == ["Alice"]
    assert response.group.avg_similarity == 0.87
    assert response.debug_info == {"edges": 1}
    assert response.processing_time >= 0.0


@pytest.mark.asyncio
async def test_response_lacking_the_main_post_degrades_to_a_single_group(monkeypatch):
    result = {"groups": [group(4, [1, 2])]}
    monkeypatch.setattr(endpoint, "_cluster_sync", SyncRecorder(result=result))
    main = post(1, "main", similarPosts=[post(2), post(3)])

    response = await endpoint.cluster_texts(main, request_for(make_state()))

    assert (response.total_posts, response.total_groups) == (3, 1)
    assert response.group.group_id == 0 and response.group.size == 1
    assert [p.id for p in response.group.posts] == [1]
    assert response.group.avg_similarity == 1.0
    assert response.processing_time is not None


# --- degraded paths ---------------------------------------------------------

@pytest.mark.asyncio
async def test_a_clustering_timeout_is_a_408_and_is_counted(monkeypatch):
    monkeypatch.setattr(endpoint, "_cluster_sync", SyncRecorder(result={"groups": []}, delay=0.3))
    before = failures("timeout")

    with pytest.raises(HTTPException) as caught:
        await endpoint.cluster_texts(
            post(1, similarPosts=[post(2)]), request_for(make_state(TIMEOUT=0.01))
        )

    assert caught.value.status_code == 408
    assert caught.value.detail == "Clustering request timed out"
    assert failures("timeout") == before + 1


@pytest.mark.asyncio
async def test_a_clustering_error_is_a_500_that_carries_the_cause(monkeypatch):
    monkeypatch.setattr(endpoint, "_cluster_sync", SyncRecorder(error=RuntimeError("louvain exploded")))
    state = make_state()
    before = failures("exception")

    with pytest.raises(HTTPException) as caught:
        await endpoint.cluster_texts(post(1, similarPosts=[post(2)]), request_for(state))

    assert caught.value.status_code == 500
    assert caught.value.detail == "Clustering failed: louvain exploded"
    assert failures("exception") == before + 1
    assert state.clustering_lock.locked() is False
