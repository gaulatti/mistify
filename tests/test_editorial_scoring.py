import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.endpoints.analysis import (
    EDITORIAL_PRIORITY_LABELS,
    EDITORIAL_SCORE_MAP,
    _should_run_content_classification,
    _scores_from_editorial_result,
    _compute_urgency_signals,
    _final_urgency,
    score_editorial_priority,
    unified_analysis,
)
from src.models import (
    ClassificationResponse,
    UnifiedAnalysisRequest,
    UnifiedAnalysisItemRequest,
)


class DummyClassifier:
    """A fake DistilBART zero-shot classifier for unit tests."""

    def __init__(self, labels, scores, call_assertions=None):
        self.labels = list(labels)
        self.scores = list(scores)
        self.calls = []
        self.call_assertions = call_assertions or []

    def __call__(self, text, candidate_labels, **kwargs):
        self.calls.append({"text": text, "labels": candidate_labels, "kwargs": kwargs})
        for assertion in self.call_assertions:
            assertion(text, candidate_labels, kwargs)
        return {"labels": list(self.labels), "scores": list(self.scores)}


class FailingClassifier:
    def __call__(self, text, candidate_labels, **kwargs):
        raise RuntimeError("CUDA OOM")


@pytest.fixture
def app_state():
    """Minimal app_state fixture for analysis tests."""
    return SimpleNamespace(
        fasttext_model=None,
        translator=None,
        embedder=None,
        nlp=None,
        classifier=None,
        classification_lock=asyncio.Lock(),
        translation_lock=asyncio.Lock(),
        editorial_lock=asyncio.Lock(),
        embedding_lock=asyncio.Lock(),
        thread_pool=None,
        classification_pool=None,
        translation_pool=None,
        embedding_pool=None,
        clustering_pool=None,
        config={"TIMEOUT": 10, "EMBEDDING_TIMEOUT": 10, "DEFAULT_CLASSIFICATION_LABELS": []},
    )


@pytest.fixture
def http_request(app_state):
    """Fake HTTP request with app_state attached."""
    return SimpleNamespace(state=SimpleNamespace(app_state=app_state))


# ---------------------------------------------------------------------------
# _scores_from_editorial_result tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label,newsworthiness,urgency",
    [
        ("major breaking story", 10.0, 10.0),
        ("important story but not breaking", 8.5, 5.0),
        ("routine publishable update", 6.0, 1.5),
        ("niche or low-priority item", 2.5, 0.5),
        ("not newsworthy", 0.0, 0.0),
    ],
)
def test_scores_from_editorial_result(label, newsworthiness, urgency):
    resp = ClassificationResponse(label=label, score=1.0, full_result={"labels": [label], "scores": [1.0]})
    n, u, top = _scores_from_editorial_result(resp)

    assert top == label
    assert n == newsworthiness
    assert u == urgency


def test_unified_analysis_item_drops_non_string_categories():
    item = UnifiedAnalysisItemRequest.model_validate({
        "id": "post-1",
        "source": "rss",
        "uri": "https://example.com/story",
        "content": "Example content",
        "createdAt": "2026-07-17T00:00:00Z",
        "hash": "hash-1",
        "categories": [
            {"$": {"domain": "west_asia"}, "_": "آسیای غربی"},
            {"name": "Politics"},
            "World",
        ],
    })

    assert item.categories == ["آسیای غربی", "Politics", "World"]


def test_scores_from_editorial_result_weighted_distribution():
    # 70% major breaking, 30% important story -> weighted average
    resp = ClassificationResponse(
        label="major breaking story",
        score=0.7,
        full_result={
            "labels": ["major breaking story", "important story but not breaking"],
            "scores": [0.7, 0.3],
        },
    )
    n, u, top = _scores_from_editorial_result(resp)

    assert top == "major breaking story"
    assert n > 8.5  # closer to 10 than to 8.5
    assert n < 10.0
    assert u > 5.0  # closer to 10 than to 5
    assert u < 10.0


# ---------------------------------------------------------------------------
# _should_run_content_classification tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label,should_run",
    [
        ("major breaking story", True),
        ("important story but not breaking", True),
        ("routine publishable update", True),
        ("niche or low-priority item", False),
        ("not newsworthy", False),
        (None, True),
    ],
)
def test_should_run_content_classification(label, should_run):
    assert _should_run_content_classification(label) is should_run


# ---------------------------------------------------------------------------
# score_editorial_priority tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_score_editorial_priority_no_classifier_returns_zero(http_request):
    http_request.state.app_state.classifier = None
    n, u = await score_editorial_priority("Major earthquake hits city", http_request)
    assert n == 0.0
    assert u == 0.0


@pytest.mark.asyncio
async def test_score_editorial_priority_major_breaking(app_state, http_request):
    app_state.classifier = DummyClassifier(["major breaking story"], [1.0])
    n, u = await score_editorial_priority("Major earthquake hits city", http_request)
    assert n == 10.0
    assert u == 10.0


@pytest.mark.asyncio
async def test_score_editorial_priority_not_newsworthy(app_state, http_request):
    app_state.classifier = DummyClassifier(["not newsworthy"], [1.0])
    n, u = await score_editorial_priority("Someone had lunch", http_request)
    assert n == 0.0
    assert u == 0.0


@pytest.mark.asyncio
async def test_score_editorial_priority_important_story(app_state, http_request):
    app_state.classifier = DummyClassifier(["important story but not breaking"], [1.0])
    n, u = await score_editorial_priority("Senator announces new policy", http_request)
    assert n == 8.5
    assert u == 5.0


@pytest.mark.asyncio
async def test_score_editorial_priority_routine_update(app_state, http_request):
    app_state.classifier = DummyClassifier(["routine publishable update"], [1.0])
    n, u = await score_editorial_priority("City council meeting notes", http_request)
    assert n == 6.0
    assert u == 1.5


@pytest.mark.asyncio
async def test_score_editorial_priority_niche(app_state, http_request):
    app_state.classifier = DummyClassifier(["niche or low-priority item"], [1.0])
    n, u = await score_editorial_priority("Local bake sale announced", http_request)
    assert n == 2.5
    assert u == 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "label,should_wake",
    [
        ("major breaking story", True),   # urgency 10.0 >= 8
        ("important story but not breaking", False),  # urgency 5.0 < 8
        ("routine publishable update", False),  # urgency 1.5 < 8
        ("niche or low-priority item", False),  # urgency 0.5 < 8
        ("not newsworthy", False),  # urgency 0.0 < 8
    ],
)
async def test_wake_threshold(app_state, http_request, label, should_wake):
    app_state.classifier = DummyClassifier([label], [1.0])
    _, u = await score_editorial_priority("some post", http_request)
    assert (u >= 8.0) is should_wake


@pytest.mark.asyncio
async def test_score_editorial_priority_uses_probability_distribution(app_state, http_request):
    # 60% major breaking, 40% important -> continuous score, not hard 10 or 5.
    app_state.classifier = DummyClassifier(
        ["major breaking story", "important story but not breaking"],
        [0.6, 0.4],
    )
    n, u = await score_editorial_priority("some post", http_request)
    assert 8.5 < n < 10.0
    assert 5.0 < u < 10.0


@pytest.mark.asyncio
async def test_score_editorial_priority_failure_returns_zero(app_state, http_request):
    app_state.classifier = FailingClassifier()
    n, u = await score_editorial_priority("Major earthquake hits city", http_request)
    assert n == 0.0
    assert u == 0.0


@pytest.mark.asyncio
async def test_score_editorial_priority_timeout_returns_zero(app_state, http_request):
    app_state.classifier = DummyClassifier(["major breaking story"], [1.0])
    with patch("src.endpoints.analysis.asyncio.wait_for", side_effect=asyncio.TimeoutError):
        n, u = await score_editorial_priority("Major earthquake hits city", http_request)
    assert n == 0.0
    assert u == 0.0


# ---------------------------------------------------------------------------
# unified_analysis integration tests
# ---------------------------------------------------------------------------

@pytest.fixture
def make_analysis_request():
    def _make(content: str):
        return UnifiedAnalysisRequest(
            items=[
                UnifiedAnalysisItemRequest(
                    id="post-1",
                    source="test",
                    uri="https://example.com/1",
                    content=content,
                    createdAt="2026-06-28T00:00:00Z",
                    hash="abc123",
                )
            ],
            detect_language=False,
            classify_content=True,
            translate_to_english=False,
            include_timings=False,
        )

    return _make


@pytest.mark.asyncio
async def test_unified_analysis_no_models_skips_editorial(
    app_state, http_request, make_analysis_request
):
    app_state.classifier = None
    req = make_analysis_request("Some content")

    resp = await unified_analysis(req, http_request)

    assert resp.results[0].newsworthiness is None
    assert resp.results[0].urgency is None


@pytest.mark.asyncio
async def test_unified_analysis_sets_urgency_for_major_breaking(
    app_state, http_request, make_analysis_request
):
    app_state.classifier = DummyClassifier(["major breaking story"], [1.0])

    req = make_analysis_request("Major earthquake and tsunami warning issued")
    resp = await unified_analysis(req, http_request)

    assert resp.results[0].newsworthiness == 10.0
    assert resp.results[0].urgency == 10.0


@pytest.mark.asyncio
async def test_unified_analysis_low_newsworthiness_skips_content_classification(
    app_state, http_request, make_analysis_request
):
    app_state.classifier = DummyClassifier(["not newsworthy"], [1.0])

    # Mock classifier so we can verify it is NOT called for low-value posts.
    mock_classifier_response = ClassificationResponse(
        label="politics", score=0.9, full_result={"labels": ["politics"], "scores": [0.9]}
    )
    app_state.classifier_content = MagicMock()

    with patch("src.endpoints.analysis.classify_content", new=AsyncMock(return_value=mock_classifier_response)) as mock_classify:
        req = make_analysis_request("Someone posted a picture of their cat")
        resp = await unified_analysis(req, http_request)

        assert resp.results[0].newsworthiness == 0.0
        assert resp.results[0].urgency == 0.0
        assert resp.results[0].content_classification is None
        mock_classify.assert_not_awaited()


@pytest.mark.asyncio
async def test_unified_analysis_high_newsworthiness_runs_content_classification(
    app_state, http_request, make_analysis_request
):
    app_state.classifier = DummyClassifier(["major breaking story"], [1.0])

    mock_classifier_response = ClassificationResponse(
        label="politics", score=0.9, full_result={"labels": ["politics"], "scores": [0.9]}
    )

    with patch("src.endpoints.analysis.classify_content", new=AsyncMock(return_value=mock_classifier_response)) as mock_classify:
        req = make_analysis_request("War declared between two nations")
        resp = await unified_analysis(req, http_request)

        assert resp.results[0].newsworthiness == 10.0
        assert resp.results[0].urgency == 10.0
        assert resp.results[0].content_classification == mock_classifier_response
        mock_classify.assert_awaited_once()


@pytest.mark.asyncio
async def test_unified_analysis_editorial_failure_does_not_break_pipeline(
    app_state, http_request, make_analysis_request
):
    app_state.classifier = FailingClassifier()

    req = make_analysis_request("Some content")
    resp = await unified_analysis(req, http_request)

    # Failure falls back to safe low scores; the pipeline continues.
    assert resp.results[0].newsworthiness == 0.0
    assert resp.results[0].urgency == 0.0


# ---------------------------------------------------------------------------
# Rule-based urgency signal tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected_boost",
    [
        ("Manhattan hotel evacuated after mace sprayed inside, FDNY says", 7.5),
        ("NYC hotel evacuated after guest uses bear spray during fight: NYPD", 7.5),
        ("Police respond to shooting at downtown mall", 5.5),
        ("Major earthquake hits city", 3.5),
        ("Brazil break Japan hearts with last-gasp winner", 0.0),
        ("City council meeting notes", 0.0),
    ],
)
def test_compute_urgency_signals(text, expected_boost):
    assert _compute_urgency_signals(text) == expected_boost


@pytest.mark.asyncio
async def test_score_editorial_priority_public_safety_boost(app_state, http_request):
    # Classifier thinks it is niche, but public-safety signals override urgency.
    app_state.classifier = DummyClassifier(["niche or low-priority item"], [1.0])
    text = "Manhattan hotel evacuated after mace sprayed inside, FDNY says"
    n, u = await score_editorial_priority(text, http_request)
    assert n == 2.5
    assert u == 7.5


@pytest.mark.asyncio
async def test_score_editorial_priority_classifier_high_urgency_preserved(
    app_state, http_request
):
    # Existing high classifier urgency should not be reduced by signal logic.
    app_state.classifier = DummyClassifier(["major breaking story"], [1.0])
    n, u = await score_editorial_priority("Brazil break Japan hearts", http_request)
    assert n == 10.0
    assert u == 10.0
