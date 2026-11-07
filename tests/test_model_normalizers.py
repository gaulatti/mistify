"""Covers the request-model validators in `src/models.py`, `src/operations/models.py`
and `src/operations/media_models.py`.

Upstream feeds send noisy payloads (media as objects, categories as XML-ish
dicts, ids as structured values). The validators must reduce them to the stable
shapes the rest of the service assumes, and reject malformed input with an error
rather than passing it on.
"""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.models import (
    MediaItem,
    PostData,
    UnifiedAnalysisItemRequest,
    _normalize_id,
    _normalize_media_to_urls,
    _normalize_string_list,
)
from src.operations.media_models import (
    DiarizedSegment,
    MediaSource,
    SpeakerState,
    TimeRange,
)
from src.operations.models import GrpcCallback, HttpCallback, OperationEnvelope


# --- media normalisation ----------------------------------------------------

def test_media_none_and_non_lists_pass_through_unchanged():
    assert _normalize_media_to_urls(None) is None
    assert _normalize_media_to_urls("https://x/y.png") == "https://x/y.png"


def test_media_entries_are_reduced_to_trimmed_urls():
    value = [
        " https://a/1.png ",
        "   ",
        {"url": " https://b/2.png ", "type": "image/png"},
        {"url": ""},
        {"type": "image"},
        MediaItem(url="https://c/3.png", type="image"),
        SimpleNamespace(url="  "),
        SimpleNamespace(url=5),
        42,
    ]

    assert _normalize_media_to_urls(value) == ["https://a/1.png", "https://b/2.png", "https://c/3.png"]


def test_post_and_analysis_requests_normalise_media_on_input():
    common = dict(id="1", content="c", source="rss", hash="h", createdAt="2026-01-01T00:00:00Z")
    media = [{"url": "https://a/1.png"}, "https://b/2.png"]

    post = PostData(**{**common, "id": 1}, uuid="u", media=media)
    request = UnifiedAnalysisItemRequest(**common, uri="https://e/1", media=media)

    assert post.media == request.media == ["https://a/1.png", "https://b/2.png"]


# --- string-list normalisation ----------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        (None, []),
        ("  World ", ["World"]),
        ("   ", []),
        (7, []),
        ({"name": "x"}, []),
        ([" a ", "", "  ", "b"], ["a", "b"]),
        ([{"name": " N "}, {"label": "L"}, {"slug": "s"}, {"_": "u"}, {"name": 5}, {}], ["N", "L", "s", "u"]),
        (["keep", 3, None, ["nested"]], ["keep"]),
    ],
)
def test_string_lists_keep_only_useful_labels(value, expected):
    assert _normalize_string_list(value) == expected


def test_a_dict_label_prefers_name_then_label_then_slug():
    assert _normalize_string_list([{"name": "N", "label": "L", "slug": "S"}]) == ["N"]
    assert _normalize_string_list([{"label": "L", "slug": "S"}]) == ["L"]


# --- id normalisation -------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("abc", "abc"),
        (12, "12"),
        (1.5, "1.5"),
        (True, "True"),
        ({"id": " 77 "}, "77"),
        ({"guid": "g-1"}, "g-1"),
        ({"url": "https://e/1", "_": "ignored"}, "https://e/1"),
        ({"id": "  ", "link": "L"}, "L"),
        ({"_": "text"}, "text"),
        (object, str(object)),
    ],
)
def test_ids_are_reduced_to_strings(value, expected):
    assert _normalize_id(value) == expected


def test_an_id_dict_without_a_usable_key_is_serialised_deterministically():
    value = {"b": 2, "a": {"x": 1}}

    assert _normalize_id(value) == json.dumps(value, sort_keys=True, separators=(",", ":"))
    assert _normalize_id({"a": 1, "b": 2}) == _normalize_id({"b": 2, "a": 1})


def test_analysis_items_accept_structured_ids():
    item = UnifiedAnalysisItemRequest(
        id={"guid": "g-9"}, source="rss", uri="https://e", content="c",
        createdAt="2026-01-01T00:00:00Z", hash="h", categories="Politics",
    )

    assert item.id == "g-9" and item.categories == ["Politics"]


# --- operation callbacks and envelope ---------------------------------------

def test_http_callbacks_only_support_post_in_any_case():
    assert HttpCallback(url="https://e/cb", method=" post ").method == "POST"
    with pytest.raises(ValidationError, match="Only POST callbacks are supported"):
        HttpCallback(url="https://e/cb", method="PUT")


def test_grpc_callback_defaults_target_the_monitor_ingest_service():
    callback = GrpcCallback(target="monitor:50051")

    assert (callback.service, callback.method) == ("monitor.ingest.MonitorIngest", "ReceiveAnalysisResult")


def test_envelope_requires_a_type_and_trims_it():
    assert OperationEnvelope(operation_type="  embed_text ", payload={}).operation_type == "embed_text"
    with pytest.raises(ValidationError, match="operation_type is required"):
        OperationEnvelope(operation_type="   ", payload={})


# --- media request models ---------------------------------------------------

CHECKSUM = "AB" * 32


def test_media_source_lowercases_the_checksum():
    source = MediaSource(uri="file:///media/a.wav", checksum_sha256=CHECKSUM, media_type="audio/wav")

    assert source.checksum_sha256 == "ab" * 32


@pytest.mark.parametrize(
    "uri",
    ["relative/path.wav", "https://user:pw@host/a.wav", "https://host/a.wav?x=1", "https://host/a.wav#frag"],
)
def test_media_source_rejects_relative_or_decorated_uris(uri):
    with pytest.raises(ValidationError):
        MediaSource(uri=uri, checksum_sha256=CHECKSUM, media_type="audio/wav")


def test_time_ranges_must_run_forward():
    assert TimeRange(start_ms=0, end_ms=1).end_ms == 1
    with pytest.raises(ValidationError, match="end_ms must be greater than start_ms"):
        TimeRange(start_ms=5, end_ms=5)


def segment(**fields):
    base = dict(start_ms=0, end_ms=10, text="hello", speaker_state=SpeakerState.IDENTIFIED)
    return DiarizedSegment(**{**base, **fields})


def test_identified_speakers_need_a_label_and_confidence():
    assert segment(speaker_label="Ana", speaker_confidence=0.9).speaker_label == "Ana"
    for fields in ({}, {"speaker_label": "Ana"}, {"speaker_confidence": 0.9}):
        with pytest.raises(ValidationError, match="identified speakers require"):
            segment(**fields)


def test_unknown_speakers_cannot_carry_identity():
    assert segment(speaker_state=SpeakerState.UNKNOWN).speaker_label is None
    for fields in ({"speaker_label": "Ana"}, {"speaker_confidence": 0.5}):
        with pytest.raises(ValidationError, match="unknown speakers cannot"):
            segment(speaker_state=SpeakerState.UNKNOWN, **fields)
