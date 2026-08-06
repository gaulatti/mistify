import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from src.helpers.scout_queries import (
    _keyword_query,
    generate_scout_queries,
    rank_scout_candidates,
)


def test_keyword_query_removes_publisher_and_keeps_topic_terms():
    query = _keyword_query(
        "Assisted death law in New York reignites debate over ethical limits | Telemundo"
    )

    assert query == "Assisted death law New York reignites debate ethical limits"


@pytest.mark.asyncio
async def test_generate_scout_queries_falls_back_without_loaded_models():
    app_state = SimpleNamespace(
        fasttext_model=None,
        translator=None,
        nlp=None,
    )

    queries, translated_title, language = await generate_scout_queries(
        app_state,
        "Assisted death law in New York | Publisher",
    )

    assert queries == ["Assisted death law New York"]
    assert translated_title == "Assisted death law in New York | Publisher"
    assert language == ""


@pytest.mark.asyncio
async def test_rank_scout_candidates_filters_semantic_noise(monkeypatch):
    monkeypatch.setattr(
        "src.helpers.scout_queries._embed_sync",
        lambda *_args: np.array(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.2, 0.98],
            ],
            dtype=np.float32,
        ),
    )
    app_state = SimpleNamespace(
        embedder=object(),
        embedding_lock=asyncio.Lock(),
        embedding_pool=None,
    )

    ranked = await rank_scout_candidates(
        app_state,
        "New York assisted dying law",
        [("related", "New York assisted suicide"), ("noise", "Gladiator audiobook")],
        min_score=0.55,
    )

    assert ranked == [("related", pytest.approx(0.8))]
