import logging
import time
from fastapi import APIRouter, Request
from typing import Dict, List, Optional, Tuple
from src.models import (
    UnifiedAnalysisRequest, UnifiedAnalysisResponse, UnifiedAnalysisItemResponse,
    LanguageDetectionRequest, TranslationRequest, ClassificationRequest,
    TranslationResponse, UnifiedAnalysisItemTimings, UnifiedAnalysisTimings
)
from .language import detect_language
from .translation import translate_text
from .classification import classify_content
from src import metrics

router = APIRouter()
logger = logging.getLogger("mistify")


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


def _blend_towards_midpoint(base_score: float, confidence: float, midpoint: float) -> float:
    """Shrink uncertain scores toward a midpoint so noisy labels do less damage."""
    if confidence >= 0.7:
        return base_score
    if confidence >= 0.4:
        blend_factor = (0.7 - confidence) / 0.3 * 0.3
        return base_score * (1 - blend_factor) + midpoint * blend_factor

    blend_factor = min(0.5, (0.4 - confidence) / 0.4 * 0.5)
    return base_score * (1 - blend_factor) + midpoint * blend_factor


async def score_editorial_priority(text: str, http_request: Request) -> Tuple[float, float]:
    """
    Compute both newsworthiness and urgency from one editorial triage decision.

    The classifier is asked to pick the best newsroom action, which maps to:
    - newsworthiness: "should we publish this?"
    - urgency: "should this wake an editor right now?"
    """
    labels = [
        "publish immediately and wake the editor",
        "publish quickly at the next editorial pass",
        "publish during regular hours; useful but not urgent",
        "publish only if it strongly fits our niche or coverage plan",
        "do not publish; not meaningfully newsworthy"
    ]

    score_map = {
        "publish immediately and wake the editor": (10.0, 10.0),
        "publish quickly at the next editorial pass": (8.5, 6.5),
        "publish during regular hours; useful but not urgent": (7.0, 2.5),
        "publish only if it strongly fits our niche or coverage plan": (3.5, 1.0),
        "do not publish; not meaningfully newsworthy": (0.0, 0.0),
    }

    try:
        class_req = ClassificationRequest(text=text, labels=labels)
        class_resp = await classify_content(class_req, http_request)

        if not class_resp.full_result or "labels" not in class_resp.full_result:
            return 0.0, 0.0

        result_labels = class_resp.full_result["labels"]
        result_scores = class_resp.full_result["scores"]

        if not result_labels or not result_scores:
            return 0.0, 0.0

        top_label = result_labels[0]
        top_confidence = result_scores[0]
        base_newsworthiness, base_urgency = score_map.get(top_label, (0.0, 0.0))

        newsworthiness = _blend_towards_midpoint(base_newsworthiness, top_confidence, 5.0)
        urgency = _blend_towards_midpoint(base_urgency, top_confidence, 3.0)

        return round(newsworthiness, 2), round(urgency, 2)
    except Exception as e:
        logger.error("❌ Editorial scoring failed: %s", e)
        return 0.0, 0.0


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


@router.post("/analyze", response_model=UnifiedAnalysisResponse)
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
            detected_language = None
            item_timings = UnifiedAnalysisItemTimings(total_ms=0.0) if req.include_timings else None

            if req.detect_language and app_state.fasttext_model:
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

            if req.classify_content and app_state.classifier:
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

            # Compute newsworthiness and urgency from one editorial scoring pass
            if app_state.classifier:
                text_to_classify = item.content

                step_started_at = time.perf_counter()
                try:
                    item.newsworthiness, item.urgency = await score_editorial_priority(
                        text_to_classify, http_request
                    )
                except Exception as e:
                    logger.error("❌ Editorial scoring failed in unified analysis: %s", e)
                finally:
                    _record_shared_editorial_timing(step_started_at, batch_totals_ms, item_timings)

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
