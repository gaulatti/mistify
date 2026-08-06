from types import SimpleNamespace

import pytest

from src.helpers.scout_queries import _keyword_query, generate_scout_queries


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
