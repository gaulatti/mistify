import asyncio
import logging
import re
import time
from fastapi import APIRouter, Request
from typing import Any, Dict, List, Optional, Tuple
from src.models import (
    UnifiedAnalysisRequest, UnifiedAnalysisResponse, UnifiedAnalysisItemResponse,
    LanguageDetectionRequest, LanguageDetectionResponse, TranslationRequest, ClassificationRequest,
    ClassificationResponse, TranslationResponse, UnifiedAnalysisItemTimings, UnifiedAnalysisTimings
)
from .language import detect_language
from .translation import translate_text
from .classification import classify_content
from src.helpers.async_wrappers import _classify_sync, _embed_sync
from src import metrics

router = APIRouter()
logger = logging.getLogger("mistify")

EDITORIAL_PRIORITY_LABELS = [
    "major breaking story",
    "important story but not breaking",
    "routine publishable update",
    "niche or low-priority item",
    "not newsworthy",
]

EDITORIAL_SCORE_MAP = {
    "major breaking story": (10.0, 10.0),
    "important story but not breaking": (8.5, 5.0),
    "routine publishable update": (6.0, 1.5),
    "niche or low-priority item": (2.5, 0.5),
    "not newsworthy": (0.0, 0.0),
}

# Rule-based urgency signals. DistilBART zero-shot is not reliably calibrated
# for timeliness/harm, so we add a lightweight signal layer that boosts urgency
# when the text contains markers of active public-safety incidents, harm,
# emergency response, or large-scale developing events.
URGENCY_SIGNAL_CATEGORIES = {
    "evacuation": {"evacuated", "evacuation", "evacuees", "lockdown", "shelter in place"},
    "hazardous_agent": {"mace", "bear spray", "pepper spray", "tear gas", "chemical", "toxic", "radioactive"},
    "active_violence": {
        "shooting", "shooter", "gunman", "gunfire", "armed", "hostage", "barricaded",
        "stabbing", "stabbed", "knife", "machete", "attack", "attacker", "bomb",
        "bomber", "explosion", "exploded", "blast", "shootings",
    },
    "casualties": {
        "killed", "deaths", "dead", "died", "dying", "injured", "injuries",
        "casualty", "casualties", "wounded", "hurt", "hospitalised", "hospitalized",
    },
    "fire": {"fire", "blaze", "inferno", "wildfire"},
    "natural_disaster": {"earthquake", "tsunami", "hurricane", "tornado", "flood", "flooding"},
    "emergency_response": {
        "fdny", "nypd", "police", "firefighter", "firefighters", "emergency",
        "ems", "ambulance", "rescue", "first responder", "first responders",
    },
    "breaking": {"breaking", "developing", "ongoing", "underway"},
    "scale": {"mass", "multiple", "dozens", "hundreds", "thousands", "widespread", "numerous"},
    "threat": {"threat", "threatens", "threatened", "warning", "warned", "imminent", "danger"},
}
URGENCY_CATEGORY_WEIGHTS = {
    "evacuation": 3.0,
    "hazardous_agent": 3.0,
    "active_violence": 4.0,
    "casualties": 3.0,
    "fire": 3.0,
    "natural_disaster": 3.5,
    "emergency_response": 1.5,
    "breaking": 1.0,
    "scale": 1.0,
    "threat": 1.5,
}


def _compute_urgency_signals(text: str) -> float:
    """
    Compute a 0-10 urgency boost from explicit public-safety/incident signals.

    Each matched category contributes its weight once, regardless of how many
    terms from that category appear. The result is capped at 10.
    """
    if not text:
        return 0.0

    lowered = text.lower()
    # Normalize punctuation so multi-word signals match reliably.
    normalized = re.sub(r"[^\w\s]", " ", lowered)
    tokens = set(normalized.split())
    bigrams = {f"{w1} {w2}" for w1, w2 in zip(normalized.split(), normalized.split()[1:])}

    score = 0.0
    for category, terms in URGENCY_SIGNAL_CATEGORIES.items():
        if any(term in tokens or term in bigrams for term in terms):
            score += URGENCY_CATEGORY_WEIGHTS.get(category, 1.0)

    return min(score, 10.0)


def _final_urgency(text: str, base_urgency: float) -> float:
    """Combine classifier urgency with rule-based urgency signals."""
    signal_urgency = _compute_urgency_signals(text)
    return round(max(base_urgency, signal_urgency), 2)


def _record_analyze_step_timing(
    step_field: str,
    metric_step: str,
    started_at: float,
    batch_totals_ms: Dict[str, float],
    item_timings: Optional[UnifiedAnalysisItemTimings],
) -> None:
    duration_seconds = time.perf_counter() - started_at
    duration_ms = duration_seconds * 1000.0

    metrics.ANALYZE_STEP_DURATION_SECONDS.labels(step=metric_step).observe(duration_seconds)
    batch_totals_ms[step_field] += duration_ms

    if item_timings is not None:
        setattr(item_timings, step_field, round(duration_ms, 2))


async def score_editorial_priority(text: str, http_request: Request) -> Tuple[float, float]:
    """
    Compute both newsworthiness and urgency from one editorial triage decision.

    Uses DistilBART zero-shot classification to score the post against editorial
    priority labels, then computes weighted urgency/newsworthiness across the
    full probability distribution. This yields continuous 0-10 scores instead
    of hard 0/10 buckets.
    """
    app_state = http_request.state.app_state
    if app_state.classifier is None:
        logger.warning(
            "Editorial scoring skipped: classifier is not loaded. "
            "Set LOAD_MODELS_ON_STARTUP=true to enable urgency/newsworthiness scoring."
        )
        return 0.0, 0.0

    try:
        async with app_state.editorial_lock:
            result = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    app_state.classification_pool,
                    _classify_sync,
                    app_state.classifier,
                    text,
                    EDITORIAL_PRIORITY_LABELS,
                ),
                timeout=app_state.config["TIMEOUT"],
            )
        class_resp = ClassificationResponse(
            label=result["labels"][0],
            score=result["scores"][0],
            full_result=result,
        )
        newsworthiness, urgency, _ = _scores_from_editorial_result(class_resp)
        urgency = _final_urgency(text, urgency)
        return newsworthiness, urgency
    except Exception as e:
        logger.error("❌ Editorial scoring failed: %s", e)
        return 0.0, 0.0


def _scores_from_editorial_result(class_resp) -> Tuple[float, float, Optional[str]]:
    if not class_resp.full_result or "labels" not in class_resp.full_result:
        return 0.0, 0.0, None

    result_labels = class_resp.full_result["labels"]
    result_scores = class_resp.full_result["scores"]

    if not result_labels or not result_scores:
        return 0.0, 0.0, None

    top_label = result_labels[0]
    weighted_newsworthiness = 0.0
    weighted_urgency = 0.0
    total_weight = 0.0

    for label, score in zip(result_labels, result_scores):
        newsworthiness_weight, urgency_weight = EDITORIAL_SCORE_MAP.get(label, (0.0, 0.0))
        # Slightly sharpen the distribution so the top labels matter more
        # without collapsing everything to a single hard class.
        weight = float(score) ** 1.5
        weighted_newsworthiness += newsworthiness_weight * weight
        weighted_urgency += urgency_weight * weight
        total_weight += weight

    if total_weight <= 0.0:
        return 0.0, 0.0, top_label

    newsworthiness = weighted_newsworthiness / total_weight
    urgency = weighted_urgency / total_weight

    return round(newsworthiness, 2), round(urgency, 2), top_label


def _record_shared_editorial_timing(
    started_at: float,
    batch_totals_ms: Dict[str, float],
    item_timings: Optional[UnifiedAnalysisItemTimings],
) -> None:
    duration_seconds = time.perf_counter() - started_at
    duration_ms = duration_seconds * 1000.0
    shared_ms = duration_ms / 2.0

    metrics.ANALYZE_STEP_DURATION_SECONDS.labels(step="editorial_scoring").observe(duration_seconds)
    batch_totals_ms["newsworthiness_ms"] += shared_ms
    batch_totals_ms["urgency_ms"] += shared_ms

    if item_timings is not None:
        item_timings.newsworthiness_ms = round(shared_ms, 2)
        item_timings.urgency_ms = round(shared_ms, 2)


def _extract_detected_language(lang_value: Any) -> Optional[str]:
    """
    Normalize externally provided language values.

    Accepts:
      - "es"
      - {"lang": "es"}
      - {"language": "es"}
      - {"code": "es"}
      - {"languages": ["es", ...]}
    """
    if lang_value is None:
        return None

    if isinstance(lang_value, str):
        lang = lang_value.strip()
        return lang if lang else None

    if isinstance(lang_value, dict):
        for key in ("lang", "language", "code"):
            val = lang_value.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        langs = lang_value.get("languages")
        if isinstance(langs, list) and langs:
            first = langs[0]
            if isinstance(first, str) and first.strip():
                return first.strip()

    return None


def _should_run_content_classification(editorial_label: Optional[str]) -> bool:
    """Skip expensive taxonomy tagging for low-value stories."""
    if not editorial_label:
        return True

    return editorial_label not in {
        "niche or low-priority item",
        "not newsworthy",
    }


async def unified_analysis(req: UnifiedAnalysisRequest, http_request: Request):
    """Perform language detection, content classification, and translation on multiple input items"""
    app_state = http_request.state.app_state
    batch_size = len(req.items)
    metrics.POSTS_BATCH_SIZE.labels(endpoint="analyze").observe(batch_size)
    metrics.POSTS_PROCESSED_TOTAL.labels(endpoint="analyze").inc(batch_size)
    analysis_started_at = time.perf_counter()
    with metrics.record_operation("analyze"):
        results = []
        metrics.ANALYZE_BATCH_ITEMS.observe(batch_size)

        batch_totals_ms = {
            "language_detection_ms": 0.0,
            "translation_ms": 0.0,
            "content_classification_ms": 0.0,
            "newsworthiness_ms": 0.0,
            "urgency_ms": 0.0,
        }

        for item_req in req.items:
            item_started_at = time.perf_counter()
            item = UnifiedAnalysisItemResponse(
                id=item_req.id,
                content=item_req.content,
                hash=item_req.hash
            )
            detected_language = _extract_detected_language(item_req.lang)
            item_timings = UnifiedAnalysisItemTimings(total_ms=0.0) if req.include_timings else None
            editorial_label = None

            # If language was already detected upstream, expose it in a consistent shape.
            if detected_language:
                item.language_detection = LanguageDetectionResponse(
                    languages=[detected_language],
                    probabilities=[1.0],
                )

            if req.detect_language and app_state.fasttext_model and not detected_language:
                step_started_at = time.perf_counter()
                try:
                    lang_req = LanguageDetectionRequest(text=item_req.content, k=req.language_count)
                    lang_resp = await detect_language(lang_req, http_request)
                    item.language_detection = lang_resp
                    if lang_resp.languages:
                        detected_language = lang_resp.languages[0]
                except Exception as e:
                    logger.error("❌ Language detection failed in unified analysis: %s", e)
                finally:
                    _record_analyze_step_timing(
                        "language_detection_ms",
                        "language_detection",
                        step_started_at,
                        batch_totals_ms,
                        item_timings,
                    )

            if req.translate_to_english and app_state.translator and detected_language and detected_language.lower() not in ['en', 'eng']:
                step_started_at = time.perf_counter()
                try:
                    trans_req = TranslationRequest(text=item_req.content, source_language=detected_language)
                    trans_resp = await translate_text(trans_req, http_request)
                    item.translation = trans_resp
                    # Update content to translated text
                    item.content = trans_resp.translated_text
                except Exception as e:
                    logger.error("❌ Translation failed in unified analysis: %s", e)
                finally:
                    _record_analyze_step_timing(
                        "translation_ms",
                        "translation",
                        step_started_at,
                        batch_totals_ms,
                        item_timings,
                    )
            elif req.translate_to_english and detected_language and detected_language.lower() in ['en', 'eng']:
                # Already in English, just record that
                item.translation = TranslationResponse(
                    original_text=item_req.content,
                    translated_text=item_req.content,
                    source_language=detected_language,
                    target_language="eng"
                )

            # Compute embedding on the (possibly translated) English content.
            # This replaces any upstream embedding so downstream vector search
            # operates in a single language space.
            if app_state.embedder is None:
                logger.warning(
                    "Embedding generation skipped: embedder is not loaded. "
                    "Set LOAD_MODELS_ON_STARTUP=true to enable embedding output."
                )

            try:
                if app_state.embedder:
                    embedding_timeout = app_state.config.get(
                        "EMBEDDING_TIMEOUT",
                        app_state.config["TIMEOUT"],
                    )
                    async with app_state.embedding_lock:
                        emb = await asyncio.wait_for(
                            asyncio.get_running_loop().run_in_executor(
                                app_state.embedding_pool,
                                _embed_sync,
                                app_state.embedder,
                                [item.content],
                                64,
                                True,
                            ),
                            timeout=embedding_timeout,
                        )
                    if emb is not None and emb.shape[0] > 0:
                        item.embedding = emb[0].tolist()
            except asyncio.TimeoutError:
                logger.error(
                    "❌ Embedding generation timed out in unified analysis after %ss",
                    app_state.config.get("EMBEDDING_TIMEOUT", app_state.config["TIMEOUT"]),
                )
            except Exception as e:
                logger.error(
                    "❌ Embedding generation failed in unified analysis: %s: %s",
                    type(e).__name__,
                    e,
                )

            # Compute newsworthiness and urgency from one editorial scoring pass.
            # This uses the DistilBART zero-shot classifier, which returns a
            # probability distribution across all editorial labels.
            if app_state.classifier is None:
                logger.warning(
                    "Editorial scoring skipped: classifier is not loaded. "
                    "Set LOAD_MODELS_ON_STARTUP=true to enable urgency/newsworthiness scoring."
                )

            text_to_classify = item.content
            step_started_at = time.perf_counter()
            try:
                if app_state.classifier:
                    async with app_state.editorial_lock:
                        result = await asyncio.wait_for(
                            asyncio.get_running_loop().run_in_executor(
                                app_state.classification_pool,
                                _classify_sync,
                                app_state.classifier,
                                text_to_classify,
                                EDITORIAL_PRIORITY_LABELS,
                            ),
                            timeout=app_state.config["TIMEOUT"],
                        )
                    editorial_resp = ClassificationResponse(
                        label=result["labels"][0],
                        score=result["scores"][0],
                        full_result=result,
                    )
                    item.newsworthiness, base_urgency, editorial_label = _scores_from_editorial_result(
                        editorial_resp
                    )
                    item.urgency = _final_urgency(item.content, base_urgency)
            except Exception as e:
                logger.error("❌ Editorial scoring failed in unified analysis: %s", e)
                item.newsworthiness = 0.0
                item.urgency = 0.0
                editorial_label = "not newsworthy"
            finally:
                _record_shared_editorial_timing(step_started_at, batch_totals_ms, item_timings)

            if req.classify_content and app_state.classifier and _should_run_content_classification(editorial_label):
                step_started_at = time.perf_counter()
                try:
                    # Use the (potentially translated) content for classification
                    text_to_classify = item.content

                    class_req = ClassificationRequest(text=text_to_classify, labels=req.classification_labels)
                    class_resp = await classify_content(class_req, http_request)
                    item.content_classification = class_resp
                except Exception as e:
                    logger.error("❌ Content classification failed in unified analysis: %s", e)
                finally:
                    _record_analyze_step_timing(
                        "content_classification_ms",
                        "content_classification",
                        step_started_at,
                        batch_totals_ms,
                        item_timings,
                    )

            item_total_seconds = time.perf_counter() - item_started_at
            metrics.ANALYZE_STEP_DURATION_SECONDS.labels(step="item_total").observe(item_total_seconds)

            if item_timings is not None:
                item_timings.total_ms = round(item_total_seconds * 1000.0, 2)
                item.timings = item_timings

            results.append(item)

        total_ms = (time.perf_counter() - analysis_started_at) * 1000.0
        avg_item_ms = total_ms / len(req.items) if req.items else 0.0

        if total_ms >= 1000.0:
            logger.warning(
                "Slow /analyze batch: items=%d total_ms=%.2f avg_item_ms=%.2f "
                "language_detection_ms=%.2f translation_ms=%.2f "
                "content_classification_ms=%.2f newsworthiness_ms=%.2f urgency_ms=%.2f",
                batch_size,
                total_ms,
                avg_item_ms,
                batch_totals_ms["language_detection_ms"],
                batch_totals_ms["translation_ms"],
                batch_totals_ms["content_classification_ms"],
                batch_totals_ms["newsworthiness_ms"],
                batch_totals_ms["urgency_ms"],
            )

        timings = None
        if req.include_timings:
            timings = UnifiedAnalysisTimings(
                total_ms=round(total_ms, 2),
                item_count=batch_size,
                avg_item_ms=round(avg_item_ms, 2),
                language_detection_ms=round(batch_totals_ms["language_detection_ms"], 2),
                translation_ms=round(batch_totals_ms["translation_ms"], 2),
                content_classification_ms=round(batch_totals_ms["content_classification_ms"], 2),
                newsworthiness_ms=round(batch_totals_ms["newsworthiness_ms"], 2),
                urgency_ms=round(batch_totals_ms["urgency_ms"], 2),
            )

        return UnifiedAnalysisResponse(results=results, timings=timings)
