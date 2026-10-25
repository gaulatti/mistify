"""Termination and grouping contracts for the real community splitter."""

import networkx as nx
import numpy as np
import pytest

from src.helpers import clustering
from src.helpers.async_wrappers import _cluster_sync


def similarities(size, similarity=0.9):
    matrix = np.full((size, size), similarity)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def nonprogressing_partition(monkeypatch):
    calls = []

    def partition(graph, **kwargs):
        calls.append(set(graph))
        # Fail promptly on the broken implementation instead of exhausting the
        # interpreter stack or hanging a CI worker.
        assert len(calls) <= 8, "community splitting repeated without progress"
        return [set(graph)]

    monkeypatch.setattr(clustering, "louvain_communities", partition)
    return calls


def test_large_cohesive_group_stops_when_partition_makes_no_progress(monkeypatch):
    calls = nonprogressing_partition(monkeypatch)
    community = list(range(26))

    assert clustering.split_large_communities(community, similarities(26)) == [community]
    assert len(calls) == 1


def test_loose_group_stops_without_progress_and_keeps_existing_singleton_result(monkeypatch):
    calls = nonprogressing_partition(monkeypatch)
    matrix = similarities(4, 0.1)
    matrix[0, 1] = matrix[1, 0] = 0.6

    assert clustering.split_large_communities(list(range(4)), matrix) == [[0], [1], [2], [3]]
    assert len(calls) == 1


@pytest.mark.parametrize("similarity, expected", [(0.9, [[0, 1, 2]]), (0.1, [[0], [1], [2]])])
def test_both_recursion_paths_stop_at_the_existing_depth_boundary(monkeypatch, similarity, expected):
    def unexpected_partition(*args, **kwargs):
        pytest.fail("partitioning must stop at the recursion depth boundary")

    monkeypatch.setattr(clustering, "louvain_communities", unexpected_partition)

    assert clustering.split_large_communities(
        [0, 1, 2], similarities(3, similarity), {"big_community_size": 1}, depth=4
    ) == expected


def test_progressing_large_splits_are_still_depth_bounded(monkeypatch):
    calls = []

    def partition(graph, **kwargs):
        nodes = list(graph)
        calls.append(nodes)
        assert len(calls) <= 8, "progressing splits exceeded the depth boundary"
        return [set(nodes[:-1]), {nodes[-1]}]

    monkeypatch.setattr(clustering, "louvain_communities", partition)
    community = list(range(32))
    groups = clustering.split_large_communities(
        community, similarities(32), {"big_community_size": 1}
    )

    assert len(calls) == 4
    assert sorted(index for group in groups for index in group) == community
    assert len(groups[0]) == 28


def test_nonprogressing_split_preserves_original_noncontiguous_indices(monkeypatch):
    nonprogressing_partition(monkeypatch)
    community = [2, 4, 8, 16]

    assert clustering.split_large_communities(
        community, similarities(17), {"big_community_size": 1}
    ) == [community]


def test_existing_single_small_and_loose_grouping_cases_are_unchanged():
    assert clustering.split_large_communities([0], similarities(1)) == [[0]]
    assert clustering.split_large_communities([0, 1, 2], similarities(3)) == [[0, 1, 2]]
    assert clustering.split_large_communities([0, 1, 2], similarities(3, 0.1)) == [[0], [1], [2]]

    matrix = similarities(4, 0.1)
    matrix[0, 1] = matrix[1, 0] = 0.9
    matrix[2, 3] = matrix[3, 2] = 0.9
    groups = clustering.split_large_communities(list(range(4)), matrix)
    assert {frozenset(group) for group in groups} == {frozenset([0, 1]), frozenset([2, 3])}


def test_real_louvain_terminates_for_twenty_six_mutually_similar_posts():
    community = list(range(26))
    groups = clustering.split_large_communities(community, similarities(26))

    assert groups
    assert sorted(index for group in groups for index in group) == community


def test_clustering_worker_returns_group_without_recursion_failure_fallback(monkeypatch, caplog):
    size = 26
    texts = [f"Fictional council event report {index}" for index in range(size)]
    graph = nx.complete_graph(size)
    matrix = similarities(size)
    monkeypatch.setattr(
        clustering,
        "build_clustering_graph",
        lambda *args, **kwargs: (graph, matrix, {}, ["politics"] * size, [{"Council"}] * size),
    )
    monkeypatch.setattr(
        nx.algorithms.community, "louvain_communities", lambda graph, **kwargs: [set(graph)]
    )
    calls = nonprogressing_partition(monkeypatch)

    result = _cluster_sync(texts, None, None, None)

    assert len(result["groups"]) == 1
    group = result["groups"][0]
    assert group.indices == list(range(size))
    assert group.texts == texts
    assert group.size == size
    assert group.avg_similarity == pytest.approx(0.9)
    assert len(calls) == 1
    assert "Error splitting community" not in caplog.text
