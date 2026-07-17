import asyncio
import logging
import time
from fastapi import APIRouter, HTTPException, Request
from src.models import TranslationRequest, TranslationResponse
from src.helpers.async_wrappers import _translate_sync
from src import metrics

router = APIRouter()
logger = logging.getLogger("mistify")

SUPPORTED_TRANSLATION_LANGUAGES = {
    "af", "afr", "am", "amh", "ar", "arb", "ary", "arz", "as", "asm",
    "az", "azj", "be", "bel", "ben", "bg", "bn", "bos", "bs", "bul",
    "ca", "cat", "ceb", "ces", "ckb", "cmn", "cmn-hant", "cs", "cy", "cym",
    "da", "dan", "de", "deu", "el", "ell", "en", "eng", "es", "est",
    "et", "eu", "eus", "fa", "ff", "fr", "fra", "fuv", "ga", "gaz",
    "gle", "gl", "glg", "gu", "guj", "he", "heb", "hi", "hin", "hr",
    "hrv", "hu", "hun", "hy", "hye", "id", "ibo", "ig", "ind", "is",
    "isl", "it", "ita", "ja", "jav", "jpn", "jv", "ka", "kan", "kat",
    "kaz", "kk", "khk", "khm", "kir", "km", "kn", "ko", "kor", "ky",
    "lao", "lo", "lt", "lit", "lug", "luo", "lv", "lvs", "mai", "mal",
    "mar", "mk", "mkd", "ml", "mlt", "mn", "mni", "mr", "ms", "mt",
    "my", "mya", "ne", "nld", "nl", "nn", "nno", "no", "nob", "npi",
    "ny", "nya", "or", "ory", "pa", "pan", "pbt", "pes", "pl", "pol",
    "por", "ps", "pt", "ro", "ron", "ru", "rus", "sat", "sd", "sk",
    "sl", "slk", "slv", "sn", "sna", "snd", "so", "som", "spa", "sr",
    "srp", "sv", "sw", "swe", "swh", "ta", "tam", "te", "tel", "tg",
    "tgk", "tgl", "th", "tha", "tl", "tr", "tur", "uk", "ukr", "ur",
    "urd", "uz", "uzn", "vi", "vie", "yo", "yor", "yue", "zh", "zlm",
    "zu", "zul",
}

TRANSLATION_LANGUAGE_ALIASES = {
    "arb": "ar", "ara": "ar", "arabic": "ar",
    "cmn": "zh", "chi": "zh", "chinese": "zh", "zh-cn": "zh", "zh-tw": "cmn-hant", "zho": "zh",
    "deu": "de", "ger": "de", "german": "de",
    "eng": "en", "english": "en",
    "fas": "fa", "farsi": "fa", "per": "fa", "persian": "fa", "pes": "fa",
    "fra": "fr", "fre": "fr", "french": "fr",
    "heb": "he", "hebrew": "he", "iw": "he",
    "japanese": "ja", "jp": "ja", "jpn": "ja",
    "spa": "es", "spanish": "es",
}

UNSUPPORTED_TRANSLATION_FALLBACKS = {
    "ast": "es",  # Asturian is not supported by Seamless; Spanish is the closest supported fallback.
    "lim": "nl",  # Limburgish is not supported by Seamless; Dutch is the closest supported fallback.
    "vec": "it",  # Venetian is not supported by Seamless; Italian is the closest supported fallback.
}


def _normalize_translation_language(language: str | None) -> str | None:
    if not language:
        return None

    cleaned = language.strip().lower().replace("__label__", "")
    cleaned = cleaned.split(".", 1)[0].replace("_", "-")
    if cleaned in TRANSLATION_LANGUAGE_ALIASES:
        return TRANSLATION_LANGUAGE_ALIASES[cleaned]
    if cleaned in UNSUPPORTED_TRANSLATION_FALLBACKS:
        return UNSUPPORTED_TRANSLATION_FALLBACKS[cleaned]

    base = cleaned.split("-", 1)[0]
    if base in UNSUPPORTED_TRANSLATION_FALLBACKS:
        return UNSUPPORTED_TRANSLATION_FALLBACKS[base]
    return TRANSLATION_LANGUAGE_ALIASES.get(base, base)


@router.post("/translate", response_model=TranslationResponse)
async def translate_text(req: TranslationRequest, http_request: Request):
    """Translate text to English using Seamless M4T v2"""
    app_state = http_request.state.app_state
    if app_state.translator is None:
        raise HTTPException(status_code=503, detail="Translation model not available")

    # Track translation request
    metrics.POSTS_PROCESSED_TOTAL.labels(endpoint="translate").inc()
    original_source_language = req.source_language
    source_language = _normalize_translation_language(req.source_language)
    if original_source_language and source_language and original_source_language.lower() != source_language:
        logger.info(
            "Normalized source language for translation: %s -> %s",
            original_source_language,
            source_language,
        )

    if app_state.fasttext_model is not None and (
        not source_language or source_language not in SUPPORTED_TRANSLATION_LANGUAGES
    ):
        try:
            cleaned_text = ' '.join(req.text.replace('\n', ' ').replace('\r', ' ').strip().split())
            if cleaned_text:
                labels, probs = app_state.fasttext_model.predict(cleaned_text, k=10)
                candidates = [
                    (_normalize_translation_language(label), float(prob))
                    for label, prob in zip(labels, probs)
                ]
                supported_candidate = next(
                    (lang for lang, _ in candidates if lang in SUPPORTED_TRANSLATION_LANGUAGES),
                    None,
                )
                if supported_candidate:
                    if source_language and source_language != supported_candidate:
                        logger.warning(
                            "Replacing unsupported source language %s with supported FastText candidate %s; candidates=%s",
                            source_language,
                            supported_candidate,
                            candidates,
                        )
                    source_language = supported_candidate
                    logger.info(
                        "Detected source language for translation: %s (confidence=%.4f)",
                        source_language,
                        next(prob for lang, prob in candidates if lang == source_language),
                    )
        except Exception as e:
            logger.warning("Source language detection before translation failed: %s", e)

    if not source_language:
        source_language = "en"
        logger.warning(
            "Source language is unknown; falling back to %s for translation",
            source_language,
        )
    elif source_language not in SUPPORTED_TRANSLATION_LANGUAGES:
        logger.warning(
            "No supported FastText fallback found for source language %s; falling back to en",
            source_language,
        )
        source_language = "en"

    with metrics.record_operation("translate"):
        queue_wait_start = time.perf_counter()
        async with app_state.translation_lock:
            queue_wait_duration = time.perf_counter() - queue_wait_start
            metrics.MODEL_OPERATION_PHASE_DURATION_SECONDS.labels(
                operation="translate", phase="queue_wait"
            ).observe(queue_wait_duration)

            execution_start = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        app_state.translation_pool,
                        _translate_sync,
                        app_state.translator,
                        req.text,
                        source_language,
                        req.target_language,
                        getattr(app_state, 'translator_model_name', None)
                    ),
                    timeout=app_state.config["TIMEOUT"] * 2,
                )
            except asyncio.TimeoutError:
                metrics.OPERATION_FAILURES_TOTAL.labels(operation="translate", failure_type="timeout").inc()
                raise HTTPException(status_code=408, detail="Translation request timed out")
            except Exception as e:
                metrics.OPERATION_FAILURES_TOTAL.labels(operation="translate", failure_type="exception").inc()
                logger.error("❌ Translation error: %s", e)
                raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
            finally:
                execution_duration = time.perf_counter() - execution_start
                metrics.MODEL_OPERATION_PHASE_DURATION_SECONDS.labels(
                    operation="translate", phase="execute"
                ).observe(execution_duration)

        if result is None:
            metrics.OPERATION_FAILURES_TOTAL.labels(operation="translate", failure_type="null_result").inc()
            raise HTTPException(status_code=500, detail="Translation failed")

        try:
            # Handle different response formats from different models
            if isinstance(result, list) and len(result) > 0:
                # Standard transformers pipeline format
                translated_text = result[0].get('translation_text', '')
                if not translated_text:
                    # Try alternative key names
                    translated_text = result[0].get('generated_text', '') or result[0].get('text', '')
            elif isinstance(result, dict):
                # Single result format
                translated_text = result.get('translation_text', '') or result.get('generated_text', '') or result.get(
                    'text', '')
            else:
                # Fallback to string conversion
                translated_text = str(result)

            # Validate translation result
            if not translated_text or translated_text.strip() == "":
                logger.warning("Translation result was empty, returning original text")
                translated_text = req.text

            logger.info("✓ Translation completed: %d -> %d chars", len(req.text), len(translated_text))
            return TranslationResponse(
                original_text=req.text,
                translated_text=translated_text,
                source_language=source_language,
                target_language=req.target_language
            )
        except Exception as e:
            metrics.OPERATION_FAILURES_TOTAL.labels(operation="translate", failure_type="parse_error").inc()
            logger.error("❌ Failed to parse translation result: %s, result type: %s", e, type(result))
            raise HTTPException(status_code=500, detail="Failed to parse translation result")
