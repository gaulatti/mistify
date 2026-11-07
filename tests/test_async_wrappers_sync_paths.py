"""Covers `_cluster_sync`, `_embed_sync` and the classification wrappers.

These are the synchronous bodies the endpoints run in worker threads. They
promise to tolerate partial failure (one bad community or group must not sink the
whole clustering result), to hand callers well-shaped output for degenerate input,
and to retry GPU out-of-memory failures with less work. Collaborators are fakes:
no model weights, no GPU, no network.
"""

import logging

import networkx as nx
import numpy as np
import pytest

from src.helpers import async_wrappers as wrappers
from src.helpers import clustering
from src.helpers.async_wrappers import (
    _classify_batch_sync,
    _classify_sync,
    _cluster_sync,
    _embed_sync,
)


# --- _cluster_sync ----------------------------------------------------------

def graph_of(size, edges):
    graph = nx.Graph()
    graph.add_nodes_from(range(size))
    graph.add_edges_from(edges)
    return graph


def sims_of(size, pairs=None, default=0.2):
    matrix = np.full((size, size), default)
    np.fill_diagonal(matrix, 1.0)
    for (i, j), value in (pairs or {}).items():
        matrix[i, j] = matrix[j, i] = value
    return matrix


def stub_graph(monkeypatch, graph, sims, why=None, topics=None, entities=None):
    size = len(sims)
    built = (
        graph,
        sims,
        why or {},
        topics or ["misc"] * size,
        entities or [{"__NOENT__"} for _ in range(size)],
    )
    calls = []

    def fake_build(texts, nlp, embedder, classifier, show_bar=False, config=None):
        calls.append(SimpleCall(texts, nlp, embedder, classifier, show_bar, config))
        return built

    monkeypatch.setattr(clustering, "build_clustering_graph", fake_build)
    return calls


class SimpleCall:
    def __init__(self, texts, nlp, embedder, classifier, show_bar, config):
        self.texts, self.nlp, self.embedder = texts, nlp, embedder
        self.classifier, self.show_bar, self.config = classifier, show_bar, config


TEXTS = ["t0", "t1", "t2", "t3", "t4"]


def test_an_edgeless_graph_yields_one_cluster_per_text(monkeypatch):
    stub_graph(monkeypatch, graph_of(3, []), sims_of(3))

    result = _cluster_sync(TEXTS[:3], "nlp", "emb", "clf")

    assert [g.indices for g in result["groups"]] == [[0], [1], [2]]
    assert [g.size for g in result["groups"]] == [1, 1, 1]
    assert [g.avg_similarity for g in result["groups"]] == [1.0, 1.0, 1.0]
    assert [g.group_id for g in result["groups"]] == [0, 1, 2]
    assert result["debug_info"] is None


def test_an_empty_graph_object_also_yields_individual_clusters(monkeypatch):
    stub_graph(monkeypatch, nx.Graph(), sims_of(2))

    result = _cluster_sync(TEXTS[:2], "nlp", "emb", "clf")

    assert [g.indices for g in result["groups"]] == [[0], [1]]


def test_connected_posts_form_communities_with_their_mean_similarity(monkeypatch):
    graph = graph_of(5, [(0, 1), (0, 2), (1, 2), (3, 4)])
    sims = sims_of(5, {(0, 1): 0.9, (0, 2): 0.8, (1, 2): 0.7, (3, 4): 0.85})
    stub_graph(monkeypatch, graph, sims)

    result = _cluster_sync(TEXTS, "nlp", "emb", "clf")

    groups = result["groups"]
    # Largest community first; ids follow that order.
    assert [(g.group_id, g.indices, g.size) for g in groups] == [(0, [0, 1, 2], 3), (1, [3, 4], 2)]
    assert groups[0].texts == ["t0", "t1", "t2"]
    assert groups[0].avg_similarity == pytest.approx(0.8)
    assert groups[1].avg_similarity == pytest.approx(0.85)


def test_arguments_reach_the_graph_builder_unchanged(monkeypatch):
    calls = stub_graph(monkeypatch, graph_of(2, []), sims_of(2))
    config = {"similarity_global": 0.7}

    _cluster_sync(TEXTS[:2], "nlp", "emb", "clf", config=config, debug=True)

    call = calls[0]
    assert (call.texts, call.nlp, call.embedder, call.classifier) == (TEXTS[:2], "nlp", "emb", "clf")
    assert call.show_bar is True and call.config is config


def test_primary_topic_is_the_most_common_and_entities_exclude_the_marker(monkeypatch):
    graph = graph_of(3, [(0, 1), (1, 2), (0, 2)])
    entities = [
        {"Alice", "__NOENT__"},
        {"Alice", "Bob"},
        {"Carol", "Dan", "Erin", "Frank", "Grace"},
    ]
    stub_graph(monkeypatch, graph, sims_of(3, default=0.9), topics=["sports", "politics", "sports"], entities=entities)

    group = _cluster_sync(TEXTS[:3], "nlp", "emb", "clf")["groups"][0]

    assert group.primary_topic == "sports"
    assert "__NOENT__" not in group.primary_entities
    assert len(group.primary_entities) == 5
    assert set(group.primary_entities) <= {"Alice", "Bob", "Carol", "Dan", "Erin", "Frank", "Grace"}


def test_debug_mode_reports_edge_reasons_topics_and_entity_counts(monkeypatch):
    graph = graph_of(2, [(0, 1)])
    why = {(0, 1): "EVENT_SEM 0.90"}
    entities = [{"Alice"}, {"Alice", "Bob"}]
    stub_graph(monkeypatch, graph, sims_of(2, {(0, 1): 0.9}), why, ["sports", "sports"], entities)

    debug_info = _cluster_sync(TEXTS[:2], "nlp", "emb", "clf", debug=True)["debug_info"]

    assert debug_info == {
        "edge_reasons": {"0-1": "EVENT_SEM 0.90"},
        "total_edges": 1,
        "topics": ["sports", "sports"],
        "entities_count": 3,
    }


def test_a_failing_split_keeps_the_original_community_and_others_still_split(monkeypatch):
    graph = graph_of(5, [(0, 1), (1, 2), (0, 2), (3, 4)])
    stub_graph(monkeypatch, graph, sims_of(5, default=0.9))
    real_split = clustering.split_large_communities

    def flaky_split(comm, sims, config=None, depth=0):
        if 3 in comm:
            raise RuntimeError("split exploded")
        return real_split(comm, sims, config, depth)

    monkeypatch.setattr(clustering, "split_large_communities", flaky_split)

    groups = _cluster_sync(TEXTS, "nlp", "emb", "clf")["groups"]

    assert sorted(g.indices for g in groups) == [[0, 1, 2], [3, 4]]


def test_out_of_range_groups_are_skipped(monkeypatch, caplog):
    graph = graph_of(2, [(0, 1)])
    stub_graph(monkeypatch, graph, sims_of(2, {(0, 1): 0.9}))

    def fake_split(comm, sims, config=None, depth=0):
        return [[0, 1], [0, 99]]

    monkeypatch.setattr(clustering, "split_large_communities", fake_split)

    with caplog.at_level(logging.WARNING, logger="src.helpers.async_wrappers"):
        groups = _cluster_sync(TEXTS[:2], "nlp", "emb", "clf")["groups"]

    assert [g.indices for g in groups] == [[0, 1]]
    assert "Invalid index 99" in caplog.text


def test_a_similarity_failure_defaults_the_group_to_zero(monkeypatch):
    graph = graph_of(2, [(0, 1)])

    class ExplodingSims(np.ndarray):
        def __getitem__(self, key):
            if isinstance(key, tuple) and key == (0, 1):
                raise IndexError("bad pair")
            return super().__getitem__(key)

    sims = sims_of(2, {(0, 1): 0.9}).view(ExplodingSims)
    stub_graph(monkeypatch, graph, sims)
    monkeypatch.setattr(clustering, "split_large_communities", lambda comm, s, c=None, depth=0: [[0, 1]])

    group = _cluster_sync(TEXTS[:2], "nlp", "emb", "clf")["groups"][0]

    assert group.avg_similarity == 0.0
    assert group.indices == [0, 1]


def test_a_group_that_cannot_be_built_is_dropped_without_losing_the_rest(monkeypatch):
    graph = graph_of(4, [(0, 1), (2, 3)])
    # Topics shorter than the posts: building any group touching index 3 fails.
    stub_graph(monkeypatch, graph, sims_of(4, default=0.9), topics=["misc", "misc", "misc"])

    groups = _cluster_sync(TEXTS[:4], "nlp", "emb", "clf")["groups"]

    assert [g.indices for g in groups] == [[0, 1]]


def test_a_graph_builder_failure_propagates_and_frees_gpu_memory(monkeypatch):
    def broken(*_a, **_k):
        raise ValueError("no models")

    emptied = []
    monkeypatch.setattr(clustering, "build_clustering_graph", broken)
    monkeypatch.setattr(wrappers.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(wrappers.torch.cuda, "empty_cache", lambda: emptied.append(True))

    with pytest.raises(ValueError, match="no models"):
        _cluster_sync(TEXTS[:2], "nlp", "emb", "clf")

    assert emptied == [True]


def test_a_graph_builder_failure_without_a_gpu_just_propagates(monkeypatch):
    monkeypatch.setattr(clustering, "build_clustering_graph", lambda *a, **k: (_ for _ in ()).throw(KeyError("x")))
    monkeypatch.setattr(wrappers.torch.cuda, "is_available", lambda: False)

    with pytest.raises(KeyError):
        _cluster_sync(TEXTS[:2], "nlp", "emb", "clf")


# --- _classify_sync ---------------------------------------------------------

class RecordingClassifier:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def cuda(monkeypatch):
    """Pretend a GPU exists and record cache flushes."""
    emptied = []
    monkeypatch.setattr(wrappers.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(wrappers.torch.cuda, "empty_cache", lambda: emptied.append(True))
    return emptied


def test_classification_passes_the_labels_and_returns_the_result():
    classifier = RecordingClassifier({"labels": ["sports"]})

    assert _classify_sync(classifier, "match report", ["sports", "politics"]) == {"labels": ["sports"]}
    assert classifier.calls == [("match report", {"candidate_labels": ["sports", "politics"]})]


def test_classification_truncates_long_text_to_512_characters():
    classifier = RecordingClassifier({})

    _classify_sync(classifier, "x" * 900, ["a"])

    assert len(classifier.calls[0][0]) == 512


@pytest.mark.parametrize("message", ["CUDA out of memory. Tried to allocate", "out of memory"])
def test_out_of_memory_retries_once_with_a_shorter_text(cuda, message):
    classifier = RecordingClassifier(RuntimeError(message), {"labels": ["ok"]})

    assert _classify_sync(classifier, "y" * 600, ["a"]) == {"labels": ["ok"]}
    assert [len(call[0]) for call in classifier.calls] == [512, 256]
    assert cuda == [True]


def test_out_of_memory_on_short_text_retries_without_cutting_it(cuda):
    classifier = RecordingClassifier(RuntimeError("out of memory"), {"labels": ["ok"]})

    _classify_sync(classifier, "short", ["a"])

    assert [call[0] for call in classifier.calls] == ["short", "short"]


def test_out_of_memory_without_a_gpu_still_retries(monkeypatch):
    monkeypatch.setattr(wrappers.torch.cuda, "is_available", lambda: False)
    classifier = RecordingClassifier(RuntimeError("out of memory"), {"labels": ["ok"]})

    assert _classify_sync(classifier, "z" * 400, ["a"]) == {"labels": ["ok"]}


def test_repeated_out_of_memory_raises_the_original_error(cuda):
    first = RuntimeError("out of memory (first)")
    classifier = RecordingClassifier(first, RuntimeError("out of memory (second)"))

    with pytest.raises(RuntimeError) as caught:
        _classify_sync(classifier, "z" * 400, ["a"])

    assert caught.value is first


def test_other_runtime_errors_are_not_retried():
    classifier = RecordingClassifier(RuntimeError("bad tensor shape"))

    with pytest.raises(RuntimeError, match="bad tensor shape"):
        _classify_sync(classifier, "text", ["a"])

    assert len(classifier.calls) == 1


# --- _classify_batch_sync ---------------------------------------------------

def test_an_empty_batch_never_calls_the_classifier():
    classifier = RecordingClassifier()

    assert _classify_batch_sync(classifier, [], ["a"]) == []
    assert classifier.calls == []


def test_a_batch_is_cleaned_truncated_and_sent_with_its_batch_size():
    classifier = RecordingClassifier([{"labels": ["a"]}, {"labels": ["b"]}, {"labels": ["c"]}])

    result = _classify_batch_sync(classifier, ["x" * 700, None, "ok"], ["a", "b"], batch_size=4)

    sent, kwargs = classifier.calls[0]
    assert [len(t) for t in sent] == [512, 0, 2]
    assert kwargs == {"candidate_labels": ["a", "b"], "batch_size": 4}
    assert result == [{"labels": ["a"]}, {"labels": ["b"]}, {"labels": ["c"]}]


def test_a_single_dict_result_is_wrapped_in_a_list():
    classifier = RecordingClassifier({"labels": ["a"]})

    assert _classify_batch_sync(classifier, ["only"], ["a"]) == [{"labels": ["a"]}]


def test_batch_out_of_memory_falls_back_to_serial_classification(cuda):
    classifier = RecordingClassifier(
        RuntimeError("CUDA Out Of Memory"), {"labels": ["one"]}, {"labels": ["two"]}
    )

    result = _classify_batch_sync(classifier, ["first", "second"], ["a"])

    assert result == [{"labels": ["one"]}, {"labels": ["two"]}]
    assert classifier.calls[0][0] == ["first", "second"]
    assert [call[0] for call in classifier.calls[1:]] == ["first", "second"]
    assert cuda == [True]


def test_batch_out_of_memory_without_a_gpu_still_falls_back(monkeypatch):
    monkeypatch.setattr(wrappers.torch.cuda, "is_available", lambda: False)
    classifier = RecordingClassifier(RuntimeError("out of memory"), {"labels": ["one"]})

    assert _classify_batch_sync(classifier, ["only"], ["a"]) == [{"labels": ["one"]}]


def test_other_batch_errors_propagate_without_a_serial_retry():
    classifier = RecordingClassifier(RuntimeError("model crashed"))

    with pytest.raises(RuntimeError, match="model crashed"):
        _classify_batch_sync(classifier, ["a", "b"], ["x"])

    assert len(classifier.calls) == 1


# --- _embed_sync ------------------------------------------------------------

class FakeEmbedder:
    def __init__(self, vectors=None, error=None):
        self.vectors = vectors
        self.error = error
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        if self.error is not None:
            raise self.error
        if self.vectors is not None:
            return self.vectors
        return np.arange(len(texts) * 3, dtype=np.float64).reshape(len(texts), 3)


def test_no_texts_yield_an_empty_matrix_without_calling_the_model():
    embedder = FakeEmbedder()

    result = _embed_sync(embedder, [], 8, True)

    assert result.shape == (0, 384) and result.dtype == np.float32
    assert embedder.calls == []


def test_only_blank_texts_yield_zero_rows_without_calling_the_model():
    embedder = FakeEmbedder()

    result = _embed_sync(embedder, ["", "   ", "\n"], 8, True)

    assert result.shape == (3, 384) and not result.any()
    assert embedder.calls == []


def test_texts_are_stripped_and_encoded_with_the_requested_options():
    embedder = FakeEmbedder()

    result = _embed_sync(embedder, ["  hello ", "world"], 16, False)

    texts, kwargs = embedder.calls[0]
    assert texts == ["hello", "world"]
    assert kwargs == {
        "batch_size": 16,
        "normalize_embeddings": False,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    assert result.dtype == np.float32 and result.shape == (2, 3)
    assert result.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_blank_texts_keep_their_position_as_zero_rows():
    embedder = FakeEmbedder()

    result = _embed_sync(embedder, ["a", "  ", "b", ""], 8, True)

    assert embedder.calls[0][0] == ["a", "b"]
    assert result.shape == (4, 3)
    assert result[0].tolist() == [0.0, 1.0, 2.0]
    assert result[2].tolist() == [3.0, 4.0, 5.0]
    assert not result[1].any() and not result[3].any()


def test_a_square_result_is_flagged_but_still_returned(caplog):
    square = np.eye(2)
    embedder = FakeEmbedder(vectors=square)

    with caplog.at_level(logging.ERROR, logger="src.helpers.async_wrappers"):
        result = _embed_sync(embedder, ["a", "b"], 8, True)

    assert "similarity matrix" in caplog.text
    assert result.tolist() == square.tolist()


def test_an_encoder_failure_returns_zero_embeddings(caplog):
    embedder = FakeEmbedder(error=RuntimeError("encoder offline"))

    with caplog.at_level(logging.WARNING, logger="src.helpers.async_wrappers"):
        result = _embed_sync(embedder, ["a", "b", "c"], 8, True)

    assert result.shape == (3, 384) and result.dtype == np.float32 and not result.any()
    assert "encoder offline" in caplog.text
