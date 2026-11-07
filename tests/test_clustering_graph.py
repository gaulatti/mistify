"""Covers `build_clustering_graph` in `src/helpers/clustering.py`.

The function turns texts into a similarity graph: it normalises entities,
gates pairs by topic, and applies a ladder of semantic thresholds that depends
on which entities two posts share. Everything runs on fakes (a spaCy-shaped
pipeline, an embedder returning caller-supplied unit vectors, a callable
classifier): no model weights, no network.

Default thresholds under test: entity 0.35, global 0.50, entity-context weight
0.25. A single shared non-generic entity raises the semantic floor to 0.55; a
generic-only overlap raises it to 0.60; a shared template without a specific
entity raises it to 0.75.
"""

import logging
import math

import pytest
import torch

from src.helpers.clustering import build_clustering_graph

DIMENSIONS = 32


# --- fakes ------------------------------------------------------------------

class FakeEntity:
    def __init__(self, text, label):
        self.text = text
        self.label_ = label


class FakeDoc:
    def __init__(self, entities):
        self.ents = [FakeEntity(text, label) for text, label in entities]


class FakeNlp:
    """Returns the entities registered for each text, in input order."""

    def __init__(self, entities_by_text):
        self._entities_by_text = entities_by_text

    def pipe(self, texts, batch_size=64):
        return (FakeDoc(self._entities_by_text.get(text, [])) for text in texts)


class FakeEmbedder:
    """Maps text to a caller-supplied vector; unknown text gets its own axis.

    Unknown strings land on mutually orthogonal axes, so entity names never
    alias unless a test gives two names (almost) the same vector.
    """

    def __init__(self, vectors=None):
        self._vectors = {k: list(v) for k, v in (vectors or {}).items()}
        self._next_axis = 8
        self.calls = []

    def encode(self, texts, **_kwargs):
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            if text not in self._vectors:
                vector = [0.0] * DIMENSIONS
                vector[self._next_axis] = 1.0
                self._next_axis += 1
                self._vectors[text] = vector
            row = self._vectors[text]
            rows.append(row + [0.0] * (DIMENSIONS - len(row)))
        return torch.tensor(rows, dtype=torch.float32)


def pair(similarity):
    """Two unit vectors whose cosine similarity is `similarity`."""
    return [1.0, 0.0], [similarity, math.sqrt(1.0 - similarity ** 2)]


def run(a, b, similarity, ents_a=(), ents_b=(), config=None, classifier=None, embedder=None):
    """Build the graph for two posts at a chosen semantic similarity."""
    va, vb = pair(similarity)
    embedder = embedder or FakeEmbedder()
    embedder._vectors[a] = va + [0.0] * (DIMENSIONS - 2)
    embedder._vectors[b] = vb + [0.0] * (DIMENSIONS - 2)
    nlp = FakeNlp({a: list(ents_a), b: list(ents_b)})
    return build_clustering_graph([a, b], nlp, embedder, classifier, config=config)


def edge_reason(result):
    graph, _sims, why, _topics, _entities = result
    assert graph.number_of_edges() in (0, 1)
    return why.get((0, 1))


ALICE = [("Alice Smith", "PERSON")]
ALICE_AND_ACME = [("Alice Smith", "PERSON"), ("Acme", "ORG")]


# --- preconditions and return shape -----------------------------------------

def test_missing_models_are_an_error():
    with pytest.raises(Exception, match="Clustering models not available"):
        build_clustering_graph(["a"], None, FakeEmbedder(), None)
    with pytest.raises(Exception, match="Clustering models not available"):
        build_clustering_graph(["a"], FakeNlp({}), None, None)


def test_result_carries_graph_similarities_reasons_topics_and_entities():
    graph, sims, why, topics, entities = run("p one", "p two", 0.9, ALICE, ALICE)

    assert sorted(graph.edges) == [(0, 1)]
    assert graph[0][1]["weight"] == pytest.approx(0.9, abs=1e-4)
    assert sims.shape == (2, 2)
    assert sims[0, 1] == pytest.approx(0.9, abs=1e-4)
    assert why == {(0, 1): "EVENT_ENT & 0.90 (ctx: 1.00)"}
    assert topics == ["misc", "misc"]
    assert entities == [{"Alice Smith"}, {"Alice Smith"}]


def test_a_single_text_builds_an_empty_graph():
    embedder = FakeEmbedder({"solo": [1.0]})
    graph, sims, why, _topics, entities = build_clustering_graph(
        ["solo"], FakeNlp({"solo": ALICE}), embedder, None
    )

    assert graph.number_of_edges() == 0 and why == {}
    assert sims.shape == (1, 1)
    assert entities == [{"Alice Smith"}]


# --- edge formation: event-specific mode ------------------------------------

def test_one_shared_specific_entity_needs_the_raised_semantic_floor():
    assert edge_reason(run("a", "b", 0.54, ALICE, ALICE)) is None
    assert edge_reason(run("a", "b", 0.56, ALICE, ALICE)).startswith("EVENT_ENT")


def test_two_shared_entities_fall_back_to_the_pure_semantic_threshold():
    # Below the entity floor (0.55) but above the global threshold (0.50).
    assert edge_reason(run("a", "b", 0.52, ALICE_AND_ACME, ALICE_AND_ACME)) == "EVENT_SEM 0.52"
    assert edge_reason(run("a", "b", 0.48, ALICE_AND_ACME, ALICE_AND_ACME)) is None


def test_posts_without_a_shared_entity_use_the_pure_semantic_threshold():
    bob = [("Bob Jones", "PERSON")]
    assert edge_reason(run("a", "b", 0.52, ALICE, bob)) == "EVENT_SEM 0.52"
    assert edge_reason(run("a", "b", 0.48, ALICE, bob)) is None


def test_shared_entities_with_weak_context_use_the_shared_entity_floor():
    """Context similarity is zeroed when too few entities are shared (min 3)."""
    config = {"similarity_global": 0.9, "min_shared_entities": 3}

    assert edge_reason(run("a", "b", 0.65, ALICE_AND_ACME, ALICE_AND_ACME, config)).startswith("EVENT_SHARED")
    assert edge_reason(run("a", "b", 0.55, ALICE_AND_ACME, ALICE_AND_ACME, config)) is None


def test_entity_aided_edge_requires_the_combined_score_to_reach_the_entity_threshold():
    """Passing the semantic floor is not enough when the blended score is low.

    With the context weight at 1.0 the blend equals the entity-context score,
    here 1/3 (one shared person of three), below the 0.35 entity threshold. The
    entity-aided rule claims the pair, so the looser rules are not consulted.
    """
    ents_a = [("Alice Smith", "PERSON"), ("Carol Diaz", "PERSON")]
    ents_b = [("Alice Smith", "PERSON"), ("Dave Chen", "PERSON")]
    config = {"entity_context_weight": 1.0}

    assert edge_reason(run("a", "b", 0.60, ents_a, ents_b, config)) is None
    assert edge_reason(run("a", "b", 0.60, ents_a, ents_b, {**config, "event_specific_mode": False})) is None
    assert edge_reason(run("a", "b", 0.60, ents_a, ents_b)).startswith("EVENT_ENT")


def test_edge_weight_is_the_semantic_similarity_not_the_blend():
    graph = run("a", "b", 0.6, ALICE_AND_ACME, ALICE_AND_ACME)[0]
    assert graph[0][1]["weight"] == pytest.approx(0.6, abs=1e-4)


# --- generic entities -------------------------------------------------------

def test_generic_only_overlap_needs_stronger_similarity_than_a_specific_one():
    trump = [("Trump", "PERSON")]
    # 0.58 clears a specific entity (0.55) but not a generic one (0.60).
    assert edge_reason(run("a", "b", 0.58, ALICE, ALICE)).startswith("EVENT_ENT")
    assert edge_reason(run("a", "b", 0.58, trump, trump)) is None
    assert edge_reason(run("a", "b", 0.64, trump, trump)).startswith("EVENT_ENT")


def test_generic_match_ignores_case_and_surrounding_whitespace():
    shouting = [("SUPREME COURT", "ORG")]
    assert edge_reason(run("a", "b", 0.58, shouting, shouting)) is None


def test_a_specific_entity_alongside_a_generic_one_is_not_generic_only():
    mixed = [("Trump", "PERSON"), ("Alice Smith", "PERSON")]
    assert edge_reason(run("a", "b", 0.52, mixed, mixed)) == "EVENT_SEM 0.52"


# --- template overlap -------------------------------------------------------

def test_format_siblings_without_a_shared_entity_need_very_high_similarity():
    a, b = "How to watch Brazil vs Japan", "How to watch Spain vs Italy"
    ents_a = [("Brazil", "GPE"), ("Japan", "GPE")]
    ents_b = [("Spain", "GPE"), ("Italy", "GPE")]

    assert edge_reason(run(a, b, 0.60, ents_a, ents_b)) is None
    assert edge_reason(run(a, b, 0.80, ents_a, ents_b)) == "EVENT_SEM 0.80"


def test_the_same_similarity_merges_non_template_posts():
    a, b = "Brazil beat Japan late", "Spain edge Italy late"
    ents_a = [("Brazil", "GPE"), ("Japan", "GPE")]
    ents_b = [("Spain", "GPE"), ("Italy", "GPE")]

    assert edge_reason(run(a, b, 0.60, ents_a, ents_b)) == "EVENT_SEM 0.60"


def test_a_shared_specific_entity_exempts_template_posts_from_the_penalty():
    a, b = "How to watch Alice Smith play", "How to watch Alice Smith win"
    assert edge_reason(run(a, b, 0.60, ALICE, ALICE)).startswith("EVENT_ENT")


# --- topic gating -----------------------------------------------------------

class SpyClassifier:
    def __init__(self, topics):
        self._topics = topics
        self.calls = 0

    def __call__(self, chunk, _labels, multi_label=True):
        self.calls += 1
        return [{"labels": [self._topics[text]]} for text in chunk]


def test_strict_topic_mode_blocks_cross_topic_edges_even_at_perfect_similarity():
    classifier = SpyClassifier({"a": "sports", "b": "politics"})
    result = run("a", "b", 0.99, ALICE, ALICE, {"topic_strict_mode": True}, classifier)

    assert edge_reason(result) is None
    assert result[3] == ["sports", "politics"]


def test_strict_topic_mode_allows_same_topic_edges():
    classifier = SpyClassifier({"a": "sports", "b": "sports"})
    result = run("a", "b", 0.99, ALICE, ALICE, {"topic_strict_mode": True}, classifier)

    assert edge_reason(result).startswith("EVENT_ENT")
    assert result[3] == ["sports", "sports"]


def test_default_mode_never_runs_the_topic_classifier():
    classifier = SpyClassifier({"a": "sports", "b": "politics"})
    result = run("a", "b", 0.99, ALICE, ALICE, None, classifier)

    assert classifier.calls == 0
    assert result[3] == ["misc", "misc"]
    assert edge_reason(result) is not None


def test_a_failing_topic_classifier_degrades_to_misc_and_still_gates():
    def broken(*_a, **_k):
        raise RuntimeError("gpu out of memory")

    result = run("a", "b", 0.99, ALICE, ALICE, {"topic_strict_mode": True}, broken)

    assert result[3] == ["misc", "misc"]
    assert edge_reason(result) is not None


# --- entity alias normalisation ---------------------------------------------

def _alias_embedder():
    near = lambda axis: [1.0 if i == axis else 0.0 for i in range(DIMENSIONS)]
    return FakeEmbedder({
        "acme corporation": near(2), "acme": near(2),
        "barack obama": near(3), "obama": near(3),
    })


def test_aliased_entities_produce_an_edge_that_raw_entities_would_not():
    ents_a = [("Acme Corporation", "ORG"), ("Barack Obama", "PERSON")]
    ents_b = [("Acme", "ORG"), ("Obama", "PERSON")]
    config = {"similarity_global": 0.9}

    aliased = run("a", "b", 0.6, ents_a, ents_b, config, embedder=_alias_embedder())
    raw = run("a", "b", 0.6, ents_a, ents_b, config, embedder=FakeEmbedder())

    assert edge_reason(aliased).startswith("EVENT_ENT")
    assert aliased[4][0] == aliased[4][1] == {"Acme", "Obama"}
    assert edge_reason(raw) is None
    assert raw[4][0] != raw[4][1]


def test_alias_normalisation_can_raise_the_floor_for_a_now_shared_entity():
    ents_a, ents_b = [("Acme Corporation", "ORG")], [("Acme", "ORG")]

    aliased = run("a", "b", 0.52, ents_a, ents_b, embedder=_alias_embedder())
    raw = run("a", "b", 0.52, ents_a, ents_b, embedder=FakeEmbedder())

    assert edge_reason(raw) == "EVENT_SEM 0.52"
    assert edge_reason(aliased) is None


# --- precomputed entities ---------------------------------------------------

def test_precomputed_entities_replace_extracted_ones_when_counts_match():
    config = {"precomputed_entities": [{"Cached One"}, {"Cached One"}]}
    result = run("a", "b", 0.9, [("Raw", "ORG")], [("Other", "ORG")], config)

    assert result[4] == [{"Cached One"}, {"Cached One"}]
    assert edge_reason(result) == "EVENT_SEM 0.90"


def test_precomputed_entities_with_the_wrong_count_are_ignored():
    config = {"precomputed_entities": [{"Cached One"}]}
    result = run("a", "b", 0.9, [("Raw", "ORG")], [("Other", "ORG")], config)

    assert result[4] == [{"Raw"}, {"Other"}]


# --- precomputed embeddings -------------------------------------------------

def _unit_vectors(similarity):
    va, vb = pair(similarity)
    return va, vb


def test_complete_precomputed_embeddings_skip_the_encoder():
    va, vb = _unit_vectors(0.9)
    embedder = FakeEmbedder()
    nlp = FakeNlp({"a": ALICE, "b": ALICE})
    # Entity alias discovery encodes entity names; texts must never be encoded.
    result = build_clustering_graph(
        ["a", "b"], nlp, embedder, None, config={"precomputed_embeddings": [va, vb]}
    )

    assert ["a", "b"] not in embedder.calls
    assert all("a" not in call or call == ["Alice Smith"] for call in embedder.calls)
    assert result[1][0, 1] == pytest.approx(0.9, abs=1e-4)
    assert (0, 1) in result[2]


def test_unnormalised_precomputed_embeddings_are_normalised_before_use():
    va, vb = _unit_vectors(0.9)
    scaled = [[x * 7 for x in va], [x * 7 for x in vb]]
    result = build_clustering_graph(
        ["a", "b"], FakeNlp({"a": ALICE, "b": ALICE}), FakeEmbedder(), None,
        config={"precomputed_embeddings": scaled},
    )

    assert result[1][0, 1] == pytest.approx(0.9, abs=1e-4)
    assert result[1][0, 0] == pytest.approx(1.0, abs=1e-4)


def test_embedding_count_mismatch_recomputes_everything(caplog):
    embedder = FakeEmbedder({"a": [1.0], "b": [1.0]})
    with caplog.at_level(logging.WARNING, logger="src.helpers.clustering"):
        result = build_clustering_graph(
            ["a", "b"], FakeNlp({}), embedder, None,
            config={"precomputed_embeddings": [[0.0, 1.0]]},
        )

    assert ["a", "b"] in embedder.calls
    assert result[1][0, 1] == pytest.approx(1.0, abs=1e-4)
    assert "doesn't match texts" in caplog.text


@pytest.mark.parametrize("unusable", [[None, None], [[], []], [(), None]])
def test_all_unusable_precomputed_embeddings_recompute_everything(unusable):
    embedder = FakeEmbedder({"a": [1.0], "b": [1.0]})
    result = build_clustering_graph(
        ["a", "b"], FakeNlp({}), embedder, None, config={"precomputed_embeddings": unusable}
    )

    assert ["a", "b"] in embedder.calls
    assert result[1][0, 1] == pytest.approx(1.0, abs=1e-4)


def test_inconsistent_embedding_dimensions_recompute_everything(caplog):
    embedder = FakeEmbedder({"a": [1.0], "b": [1.0]})
    with caplog.at_level(logging.WARNING, logger="src.helpers.clustering"):
        result = build_clustering_graph(
            ["a", "b"], FakeNlp({}), embedder, None,
            config={"precomputed_embeddings": [[1.0, 0.0], [1.0, 0.0, 0.0]]},
        )

    assert ["a", "b"] in embedder.calls
    assert result[1][0, 1] == pytest.approx(1.0, abs=1e-4)
    assert "inconsistent dimensions" in caplog.text


def test_missing_embeddings_are_encoded_and_merged_with_provided_ones():
    dim = DIMENSIONS
    provided = [1.0] + [0.0] * (dim - 1)
    embedder = FakeEmbedder({"b": provided})
    result = build_clustering_graph(
        ["a", "b"], FakeNlp({}), embedder, None,
        config={"precomputed_embeddings": [provided, None]},
    )

    # Only the missing text was encoded for the composite, and it matches.
    assert ["b"] in embedder.calls
    assert ["a", "b"] not in embedder.calls
    assert result[1][0, 1] == pytest.approx(1.0, abs=1e-4)


def test_model_dimension_mismatch_for_missing_embeddings_recomputes_everything(caplog):
    embedder = FakeEmbedder({"a": [1.0], "b": [1.0]})
    with caplog.at_level(logging.WARNING, logger="src.helpers.clustering"):
        result = build_clustering_graph(
            ["a", "b"], FakeNlp({}), embedder, None,
            config={"precomputed_embeddings": [[1.0, 0.0], None]},
        )

    assert ["b"] in embedder.calls and ["a", "b"] in embedder.calls
    assert result[1][0, 1] == pytest.approx(1.0, abs=1e-4)
    assert "recomputing all" in caplog.text


def test_unusable_precomputed_vectors_fall_back_to_encoding(caplog):
    embedder = FakeEmbedder({"a": [1.0], "b": [1.0]})
    with caplog.at_level(logging.WARNING, logger="src.helpers.clustering"):
        result = build_clustering_graph(
            ["a", "b"], FakeNlp({}), embedder, None,
            config={"precomputed_embeddings": [["x", "y"], ["x", "y"]]},
        )

    assert ["a", "b"] in embedder.calls
    assert result[1][0, 1] == pytest.approx(1.0, abs=1e-4)
    assert "Failed to use precomputed embeddings" in caplog.text


# --- non-event (original) mode ----------------------------------------------

LEGACY = {"event_specific_mode": False}


def test_legacy_mode_labels_each_edge_by_the_rule_that_admitted_it():
    bob = [("Bob Jones", "PERSON")]

    assert edge_reason(run("a", "b", 0.60, ALICE_AND_ACME, ALICE_AND_ACME, LEGACY)).startswith("ENT_CTX")
    assert edge_reason(run("a", "b", 0.52, ALICE, bob, LEGACY)) == "SEM 0.52"
    weak_context = {**LEGACY, "similarity_global": 0.9, "min_shared_entities": 3}
    assert edge_reason(run("a", "b", 0.65, ALICE_AND_ACME, ALICE_AND_ACME, weak_context)).startswith("ENT_FALL")


def test_legacy_mode_rejects_pairs_below_every_rule():
    bob = [("Bob Jones", "PERSON")]
    assert edge_reason(run("a", "b", 0.40, ALICE, bob, LEGACY)) is None


# --- rejection accounting ---------------------------------------------------

def test_the_summary_log_counts_each_rejection_reason(caplog):
    bob = [("Bob Jones", "PERSON")]
    with caplog.at_level(logging.DEBUG, logger="src.helpers.clustering"):
        run("a", "b", 0.2, ALICE, bob)
        run("a", "b", 0.54, ALICE, ALICE)
        classifier = SpyClassifier({"a": "sports", "b": "politics"})
        run("a", "b", 0.9, ALICE, ALICE, {"topic_strict_mode": True}, classifier)

    summaries = [r.getMessage() for r in caplog.records if "graph complete" in r.getMessage()]
    assert "1 rejected by entities" in summaries[0]
    assert "1 rejected by similarity" in summaries[1]
    assert "1 rejected by topic" in summaries[2]
