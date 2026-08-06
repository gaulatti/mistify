import asyncio
import logging
import re
from types import SimpleNamespace
from typing import Optional

from src.endpoints.language import detect_language
from src.endpoints.translation import translate_text
from src.helpers.async_wrappers import _embed_sync
from src.models import LanguageDetectionRequest, TranslationRequest

logger = logging.getLogger("mistify")

_STOPWORDS = {
    "about", "after", "and", "are", "for", "from", "has", "have", "into",
    "its", "over", "that", "the", "their", "this", "through", "with",
}


def _headline_core(title: str) -> str:
    return re.split(r"\s(?:\||—|–|-)\s", title, maxsplit=1)[0].strip()


def _keyword_query(text: str, nlp=None, *, limit: int = 10) -> str:
    core = _headline_core(text)
    entities: list[str] = []
    if nlp is not None:
        try:
            entities = [entity.text.strip() for entity in nlp(core).ents if entity.text.strip()]
        except Exception as exc:
            logger.warning("Scout entity extraction failed: %s", exc)

    entity_words = {
        word.casefold()
        for entity in entities
        for word in re.findall(r"[^\W_]+", entity, flags=re.UNICODE)
    }
    tokens = [
        token
        for token in re.findall(r"[^\W_]+", core, flags=re.UNICODE)
        if len(token) > 2 and token.casefold() not in _STOPWORDS
    ]
    ordered = [
        *(
            word
            for entity in entities
            for word in re.findall(r"[^\W_]+", entity, flags=re.UNICODE)
        ),
        *(token for token in tokens if token.casefold() not in entity_words),
    ]
    unique: list[str] = []
    seen: set[str] = set()
    for token in ordered:
        normalized = token.casefold()
        if normalized not in seen:
            unique.append(token)
            seen.add(normalized)
    return " ".join(unique[:limit]) or core


async def generate_scout_queries(
    app_state,
    title: str,
    source_language: Optional[str] = None,
    max_queries: int = 2,
) -> tuple[list[str], str, str]:
    """Generate compact translated YouTube queries using Mistify's local models."""
    request = SimpleNamespace(state=SimpleNamespace(app_state=app_state))
    detected_language = (source_language or "").strip()
    if not detected_language and app_state.fasttext_model is not None:
        try:
            detected = await detect_language(
                LanguageDetectionRequest(text=title, k=1),
                request,
            )
            if detected.languages:
                detected_language = detected.languages[0]
        except Exception as exc:
            logger.warning("Scout language detection failed: %s", exc)

    translated_title = title
    if (
        detected_language.casefold() not in {"", "en", "eng"}
        and app_state.translator is not None
    ):
        try:
            translated = await translate_text(
                TranslationRequest(
                    text=title,
                    source_language=detected_language,
                    target_language="eng",
                ),
                request,
            )
            translated_title = translated.translated_text or title
        except Exception as exc:
            logger.warning("Scout translation failed; using original title: %s", exc)

    queries = [_keyword_query(translated_title, app_state.nlp)]
    if translated_title != title:
        broader = _keyword_query(translated_title, app_state.nlp, limit=6)
        if broader.casefold() != queries[0].casefold():
            queries.append(broader)

    return queries[: max(1, min(3, max_queries))], translated_title, detected_language


async def rank_scout_candidates(
    app_state,
    seed_text: str,
    candidates: list[tuple[str, str]],
    min_score: float = 0.55,
    max_candidates: int = 15,
) -> list[tuple[str, float]]:
    """Rank candidate video titles against a translated seed using MiniLM."""
    usable = [(candidate_id, title) for candidate_id, title in candidates if title.strip()]
    if not seed_text.strip() or not usable or app_state.embedder is None:
        return []

    texts = [seed_text, *(title for _, title in usable)]
    async with app_state.embedding_lock:
        vectors = await asyncio.get_running_loop().run_in_executor(
            app_state.embedding_pool,
            _embed_sync,
            app_state.embedder,
            texts,
            32,
            True,
        )
    seed_vector = vectors[0]
    ranked = sorted(
        (
            (candidate_id, float(vector @ seed_vector))
            for (candidate_id, _), vector in zip(usable, vectors[1:])
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    threshold = min(1.0, max(-1.0, min_score or 0.55))
    limit = max(1, min(25, max_candidates or 15))
    return [item for item in ranked if item[1] >= threshold][:limit]
