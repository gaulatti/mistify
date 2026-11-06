"""Covers `src/helpers/clustering.py`, which had no tests at all.

The module is the clustering decision layer: entity normalization, the
similarity and gating rules, and the recursive community splitter. Everything
here runs on fakes - no model weights, no network.
"""

import numpy as np
import pytest

from src.helpers.clustering import (
    CLUSTERING_AVG_SIM_MIN,
    ENTITY_TYPE_WEIGHTS,
    _template_overlap_score,
    calculate_category_similarity,
    calculate_entity_context_similarity,
    discover_aliases,
    get_entities,
    get_entities_with_types,
    get_primary_topics,
    grouper,
    scrub_text,
    should_cluster_by_domain,
    should_cluster_by_topic,
    split_large_communities,
)


# --- fakes ------------------------------------------------------------------

class _Vectors:
    """Mimics the tensor returned by SentenceTransformer.encode()."""

    def __init__(self, array):
        self._array = array

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class FakeEmbedder:
    """Returns caller-supplied unit vectors, keyed by the scrubbed text."""

    def __init__(self, vectors, dimensions=3):
        self._vectors = vectors
        self._dimensions = dimensions
        self.calls = []

    def encode(self, texts, **_kwargs):
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            vector = self._vectors.get(text)
            if vector is None:
                # Deterministic, mutually distant default.
                vector = [0.0] * self._dimensions
                vector[len(text) % self._dimensions] = 1.0
            rows.append(vector)
        return _Vectors(np.array(rows, dtype=np.float64))


class FakeEntity:
    def __init__(self, text, label):
        self.text = text
        self.label_ = label


class FakeDoc:
    def __init__(self, entities):
        self.ents = entities


# --- _template_overlap_score ------------------------------------------------

def test_shared_template_prefix_scores_one():
    assert _template_overlap_score(
        "How to watch the final", "How to watch the draw"
    ) == 1.0


def test_unrelated_sentences_share_no_template_phrasing():
    assert _template_overlap_score("Earthquake hits region", "Budget passes vote") == 0.0


def test_single_word_inputs_have_no_bigrams():
    assert _template_overlap_score("Earthquake", "Budget") == 0.0
    assert _template_overlap_score("", "") == 0.0


def test_content_only_overlap_is_not_counted_as_template():
    """Bigrams without a function word must not register as boilerplate."""
    assert _template_overlap_score("mango papaya", "mango papaya") == 0.0


def test_partial_function_word_overlap_scores_between_zero_and_one():
    score = _template_overlap_score(
        "the vote is delayed again", "the vote is expected tomorrow"
    )
    assert 0.0 < score < 1.0


# --- scrub_text / grouper ---------------------------------------------------

def test_scrub_text_strips_punctuation_and_filler_tokens():
    assert scrub_text("The United States, of America!") == "united states america"


def test_scrub_text_can_empty_a_string_entirely():
    assert scrub_text("the a an of") == ""


@pytest.mark.parametrize(
    "size,expected",
    [
        (2, [[1, 2], [3, 4], [5]]),
        (5, [[1, 2, 3, 4, 5]]),
        (10, [[1, 2, 3, 4, 5]]),
    ],
)
def test_grouper_chunks_without_dropping_items(size, expected):
    assert [list(chunk) for chunk in grouper([1, 2, 3, 4, 5], size)] == expected


def test_grouper_on_empty_sequence_yields_nothing():
    assert list(grouper([], 3)) == []


# --- discover_aliases -------------------------------------------------------

def test_discover_aliases_without_an_embedder_returns_nothing():
    assert discover_aliases([{"Joe Biden"}], None) == {}


def test_discover_aliases_ignores_the_no_entity_marker():
    embedder = FakeEmbedder({})
    assert discover_aliases([{"__NOENT__"}, {"__NOENT__"}], embedder) == {}
    assert embedder.calls == [], "must not embed when there is nothing to cluster"


def test_discover_aliases_needs_at_least_two_entities():
    """AgglomerativeClustering raises below two samples, so it is short-circuited."""
    embedder = FakeEmbedder({})
    assert discover_aliases([{"Joe Biden"}], embedder) == {}
    assert embedder.calls == []


def test_near_identical_entities_collapse_to_the_shortest_form():
    embedder = FakeEmbedder(
        {
            "joe biden": [1.0, 0.0, 0.0],
            "biden": [1.0, 0.0, 0.0],
            "tokyo": [0.0, 1.0, 0.0],
        }
    )
    aliases = discover_aliases([{"Joe Biden", "Biden", "Tokyo"}], embedder)

    assert aliases["Joe Biden"] == "Biden"
    assert aliases["Biden"] == "Biden"
    assert aliases["Tokyo"] == "Tokyo", "a distant entity keeps its own canonical form"


# --- get_entities -----------------------------------------------------------

def test_get_entities_keeps_only_the_supported_labels():
    doc = FakeDoc(
        [
            FakeEntity("Tokyo", "GPE"),
            FakeEntity("Toyota", "ORG"),
            FakeEntity("Tuesday", "DATE"),
            FakeEntity("12 percent", "PERCENT"),
        ]
    )
    assert get_entities([doc], None) == [{"Tokyo", "Toyota"}]


def test_documents_without_entities_get_the_no_entity_marker():
    assert get_entities([FakeDoc([])], None) == [{"__NOENT__"}]


def test_get_entities_applies_discovered_aliases():
    embedder = FakeEmbedder(
        {"joe biden": [1.0, 0.0, 0.0], "biden": [1.0, 0.0, 0.0]}
    )
    docs = [
        FakeDoc([FakeEntity("Joe Biden", "PERSON")]),
        FakeDoc([FakeEntity("Biden", "PERSON")]),
    ]
    assert get_entities(docs, embedder) == [{"Biden"}, {"Biden"}]


# --- get_entities_with_types ------------------------------------------------

def test_entities_are_bucketed_by_type():
    doc = FakeDoc(
        [
            FakeEntity("Tokyo", "GPE"),
            FakeEntity("Toyota", "ORG"),
            FakeEntity("Akio Toyoda", "PERSON"),
            FakeEntity("Tuesday", "DATE"),
        ]
    )
    [buckets] = get_entities_with_types([doc], None)

    assert buckets["GPE"] == {"Tokyo"}
    assert buckets["ORG"] == {"Toyota"}
    assert buckets["PERSON"] == {"Akio Toyoda"}
    assert buckets["LOC"] == set()
    assert "DATE" not in buckets


def test_typed_entities_mark_documents_with_nothing_found():
    [buckets] = get_entities_with_types([FakeDoc([])], None)
    assert buckets["__NOENT__"] == {"__NOENT__"}


def test_typed_alias_discovery_is_applied_per_type():
    embedder = FakeEmbedder(
        {"joe biden": [1.0, 0.0, 0.0], "biden": [1.0, 0.0, 0.0]}
    )
    docs = [
        FakeDoc([FakeEntity("Joe Biden", "PERSON")]),
        FakeDoc([FakeEntity("Biden", "PERSON")]),
    ]
    buckets = get_entities_with_types(docs, embedder)
    assert buckets[0]["PERSON"] == {"Biden"}
    assert buckets[1]["PERSON"] == {"Biden"}


# --- calculate_entity_context_similarity ------------------------------------

def test_similarity_of_empty_entity_maps_is_zero():
    assert calculate_entity_context_similarity({}, {"ORG": {"Toyota"}}) == 0.0
    assert calculate_entity_context_similarity({"ORG": {"Toyota"}}, {}) == 0.0


def test_identical_single_type_entities_score_the_full_weight():
    score = calculate_entity_context_similarity(
        {"ORG": {"Toyota"}}, {"ORG": {"Toyota"}}
    )
    assert score == pytest.approx(1.0)


def test_person_entities_outrank_geopolitical_ones():
    """PERSON is the most specific signal; GPE the least."""
    assert ENTITY_TYPE_WEIGHTS["PERSON"] > ENTITY_TYPE_WEIGHTS["GPE"]

    shared_person = calculate_entity_context_similarity(
        {"PERSON": {"Ada"}, "GPE": {"Tokyo", "Osaka"}},
        {"PERSON": {"Ada"}, "GPE": {"Tokyo", "Kyoto"}},
    )
    # Both types contribute, so the result sits between the two Jaccard scores.
    assert 0.0 < shared_person < 1.0


def test_no_shared_entities_scores_zero():
    assert (
        calculate_entity_context_similarity(
            {"ORG": {"Toyota"}}, {"ORG": {"Honda"}}
        )
        == 0.0
    )


def test_the_no_entity_marker_never_counts_as_a_shared_entity():
    assert (
        calculate_entity_context_similarity(
            {"ORG": {"__NOENT__"}}, {"ORG": {"__NOENT__"}}
        )
        == 0.0
    )


def test_minimum_shared_entity_threshold_is_configurable():
    pair = ({"ORG": {"Toyota", "Honda"}}, {"ORG": {"Toyota", "Honda"}})
    assert calculate_entity_context_similarity(*pair, config={"min_shared_entities": 2}) > 0.0
    assert calculate_entity_context_similarity(*pair, config={"min_shared_entities": 3}) == 0.0


# --- topic and domain gating ------------------------------------------------

def test_topic_gating_is_permissive_by_default():
    """The five coarse DistilBART labels are too noisy to block same-event merges."""
    assert should_cluster_by_topic("sports", "politics") is True
    assert should_cluster_by_topic("sports", "politics", config={}) is True


def test_strict_topic_mode_requires_a_match():
    config = {"topic_strict_mode": True}
    assert should_cluster_by_topic("sports", "sports", config=config) is True
    assert should_cluster_by_topic("sports", "politics", config=config) is False


def test_domain_filtering_is_off_by_default():
    assert should_cluster_by_domain(["sports"], ["politics"]) is True


def test_domain_filtering_blocks_incompatible_domains_when_enabled():
    config = {"domain_filtering": True}
    assert should_cluster_by_domain(["sports"], ["politics"], config=config) is False
    assert should_cluster_by_domain(["sports"], ["Racing"], config=config) is True


def test_domain_filtering_allows_posts_without_category_data():
    config = {"domain_filtering": True}
    assert should_cluster_by_domain([], ["politics"], config=config) is True
    assert should_cluster_by_domain(["politics"], [], config=config) is True


def test_domain_filtering_allows_unrecognised_categories():
    """An unmapped category yields no domain, so it must not block clustering."""
    config = {"domain_filtering": True}
    assert should_cluster_by_domain(["weather"], ["politics"], config=config) is True


@pytest.mark.parametrize(
    "left,right,expected",
    [
        # Each of the three recognised domain groups, on both sides.
        (["Cine y Series"], ["politics"], False),
        (["politics"], ["Actualidad Cultural"], False),
        (["Ministerio de las Culturas"], ["Copa Sudamericana"], False),
        (["Cine y Series"], ["seleccion-tendencias"], True),
        (["politics"], ["conflict"], True),
        (["Alianza Lima"], ["sports"], True),
    ],
)
def test_domain_filtering_maps_every_recognised_domain_group(left, right, expected):
    config = {"domain_filtering": True}
    assert should_cluster_by_domain(left, right, config=config) is expected


# --- calculate_category_similarity ------------------------------------------

@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([], ["a"], 0.0),
        (["a"], [], 0.0),
        (["a"], ["a"], 1.0),
        (["a", "b"], ["a"], 0.5),
        (["a"], ["b"], 0.0),
    ],
)
def test_category_similarity_is_jaccard(left, right, expected):
    assert calculate_category_similarity(left, right) == pytest.approx(expected)


# --- split_large_communities ------------------------------------------------

def _sims(size, pairs, default=0.0):
    matrix = np.full((size, size), default, dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)
    for (i, j), value in pairs.items():
        matrix[i, j] = matrix[j, i] = value
    return matrix


def test_a_single_member_community_is_returned_as_is():
    assert split_large_communities([0], _sims(1, {})) == [[0]]
    assert split_large_communities([], _sims(1, {})) == [[]]


def test_a_tight_small_community_is_kept_whole():
    sims = _sims(2, {(0, 1): 0.9})
    assert split_large_communities([0, 1], sims) == [[0, 1]]


def test_a_loose_community_with_no_strong_pair_splits_into_singletons():
    sims = _sims(2, {(0, 1): 0.1})
    assert split_large_communities([0, 1], sims) == [[0], [1]]


def test_a_loose_community_splits_into_its_tight_subgroups():
    sims = _sims(4, {(0, 1): 0.9, (2, 3): 0.9})
    groups = split_large_communities([0, 1, 2, 3], sims)

    assert sorted(sorted(group) for group in groups) == [[0, 1], [2, 3]]


def test_recursion_is_bounded_by_depth():
    """Past the depth guard a loose community degrades to singletons.

    One tight pair among three members averages 0.3, below the 0.4 split floor,
    so without the guard this would keep recursing on its tight subgroup.
    """
    sims = _sims(3, {(0, 1): 0.9})
    assert split_large_communities([0, 1, 2], sims, None, depth=4) == [[0], [1], [2]]
    # Below the guard the same input splits into the tight pair plus a singleton.
    shallow = split_large_communities([0, 1, 2], sims, None, depth=0)
    assert sorted(sorted(group) for group in shallow) == [[0, 1], [2]]


def test_a_large_but_coherent_community_is_still_split():
    sims = _sims(4, {(0, 1): 0.9, (2, 3): 0.9}, default=0.45)
    groups = split_large_communities(
        [0, 1, 2, 3], sims, config={"big_community_size": 2}
    )

    assert sorted(sorted(group) for group in groups) == [[0, 1], [2, 3]]


def test_split_thresholds_are_configurable():
    sims = _sims(2, {(0, 1): 0.45})
    kept = split_large_communities([0, 1], sims, config={"avg_similarity_min": 0.2})
    split = split_large_communities([0, 1], sims, config={"avg_similarity_min": 0.9})

    assert kept == [[0, 1]]
    assert split == [[0], [1]]
    assert CLUSTERING_AVG_SIM_MIN == 0.40, "default documented in the README"


# --- get_primary_topics -----------------------------------------------------

def test_without_a_classifier_everything_is_misc():
    assert get_primary_topics(["a", "b"], None) == ["misc", "misc"]


def test_primary_topic_is_the_classifier_top_label():
    def classifier(chunk, _labels, multi_label=False):
        return [{"labels": ["sports", "misc"]} for _ in chunk]

    assert get_primary_topics(["a", "b"], classifier) == ["sports", "sports"]


def test_a_failing_classifier_degrades_the_chunk_to_misc():
    def classifier(_chunk, _labels, multi_label=False):
        raise RuntimeError("model unavailable")

    assert get_primary_topics(["a", "b", "c"], classifier) == ["misc"] * 3


def test_topics_are_classified_in_chunks_of_sixteen():
    seen = []

    def classifier(chunk, _labels, multi_label=False):
        seen.append(len(chunk))
        return [{"labels": ["misc"]} for _ in chunk]

    get_primary_topics([f"text-{i}" for i in range(20)], classifier, show_bar=False)
    assert seen == [16, 4]
