import logging
from fastapi import APIRouter, Request
from typing import Dict, List
from src.models import (
    UnifiedAnalysisRequest, UnifiedAnalysisResponse, UnifiedAnalysisItemResponse,
    LanguageDetectionRequest, TranslationRequest, ClassificationRequest,
    TranslationResponse
)
from .language import detect_language
from .translation import translate_text
from .classification import classify_content

router = APIRouter()
logger = logging.getLogger("mistify")


async def _compute_weighted_classification_score(
    text: str,
    http_request: Request,
    labels: List[str],
    weights: Dict[str, float],
    score_name: str
) -> float:
    """
    Generic helper to compute a weighted score from classification probabilities.

    Args:
        text: Text to classify
        http_request: FastAPI request object
        labels: List of classification labels
        weights: Dict mapping each label to its weight
        score_name: Name of the score (for logging)

    Returns:
        Weighted score as a float
    """
    try:
        class_req = ClassificationRequest(text=text, labels=labels)
        class_resp = await classify_content(class_req, http_request)

        if not class_resp.full_result or "labels" not in class_resp.full_result:
            return 0.0

        result_labels = class_resp.full_result["labels"]
        result_scores = class_resp.full_result["scores"]

        # Create a mapping from label to probability
        label_probs = dict(zip(result_labels, result_scores))

        # Calculate weighted score
        score = sum(weights.get(label, 0.0) * label_probs.get(label, 0.0) for label in labels)

        return round(score, 2)
    except Exception as e:
        logger.error("❌ %s scoring failed: %s", score_name, e)
        return 0.0


async def score_newsworthiness(text: str, http_request: Request) -> float:
    """
    Compute newsworthiness score (0-10) based on classification probabilities.

    Uses labels:
      - "highly newsworthy hard news"
      - "moderately newsworthy"
      - "low newsworthiness"
      - "not newsworthy"

    Formula: 10*P(highly) + 6*P(moderate) + 2*P(low) + 0*P(not newsworthy)
    """
    labels = [
        "highly newsworthy hard news",
        "moderately newsworthy",
        "low newsworthiness",
        "not newsworthy"
    ]

    weights = {
        "highly newsworthy hard news": 10.0,
        "moderately newsworthy": 6.0,
        "low newsworthiness": 2.0,
        "not newsworthy": 0.0
    }

    return await _compute_weighted_classification_score(
        text, http_request, labels, weights, "Newsworthiness"
    )


async def score_urgency(text: str, http_request: Request) -> float:
    """
    Compute urgency score (0-10) based on classification probabilities.

    Uses labels:
      - "breaking or highly time-sensitive"
      - "time-sensitive but not breaking"
      - "not time-sensitive / evergreen"

    Formula: 10*P(breaking/high) + 6*P(time-sensitive) + 1*P(evergreen)
    """
    labels = [
        "breaking or highly time-sensitive",
        "time-sensitive but not breaking",
        "not time-sensitive / evergreen"
    ]

    weights = {
        "breaking or highly time-sensitive": 10.0,
        "time-sensitive but not breaking": 6.0,
        "not time-sensitive / evergreen": 1.0
    }

    return await _compute_weighted_classification_score(
        text, http_request, labels, weights, "Urgency"
    )


@router.post("/analyze", response_model=UnifiedAnalysisResponse)
async def unified_analysis(req: UnifiedAnalysisRequest, http_request: Request):
    """Perform language detection, content classification, and translation on multiple input items"""
    app_state = http_request.state.app_state
    results = []

    for item_req in req.items:
        item = UnifiedAnalysisItemResponse(
            id=item_req.id,
            content=item_req.content,
            hash=item_req.hash
        )
        detected_language = None

        if req.detect_language and app_state.fasttext_model:
            try:
                lang_req = LanguageDetectionRequest(text=item_req.content, k=req.language_count)
                lang_resp = await detect_language(lang_req, http_request)
                item.language_detection = lang_resp
                if lang_resp.languages:
                    detected_language = lang_resp.languages[0]
            except Exception as e:
                logger.error("❌ Language detection failed in unified analysis: %s", e)

        # Always translate non-English content
        if app_state.translator and detected_language and detected_language.lower() not in ['en', 'eng']:
            try:
                trans_req = TranslationRequest(text=item_req.content, source_language=detected_language)
                trans_resp = await translate_text(trans_req, http_request)
                item.translation = trans_resp
                # Update content to translated text
                item.content = trans_resp.translated_text
            except Exception as e:
                logger.error("❌ Translation failed in unified analysis: %s", e)
        elif detected_language and detected_language.lower() in ['en', 'eng']:
            # Already in English, just record that
            item.translation = TranslationResponse(
                original_text=item_req.content,
                translated_text=item_req.content,
                source_language=detected_language,
                target_language="eng"
            )

        if req.classify_content and app_state.classifier:
            try:
                # Use the (potentially translated) content for classification
                text_to_classify = item.content
                
                class_req = ClassificationRequest(text=text_to_classify, labels=req.classification_labels)
                class_resp = await classify_content(class_req, http_request)
                item.content_classification = class_resp
            except Exception as e:
                logger.error("❌ Content classification failed in unified analysis: %s", e)

        # Compute newsworthiness and urgency scores
        if app_state.classifier:
            text_to_classify = item.content

            try:
                item.newsworthiness = await score_newsworthiness(text_to_classify, http_request)
            except Exception as e:
                logger.error("❌ Newsworthiness scoring failed in unified analysis: %s", e)

            try:
                item.urgency = await score_urgency(text_to_classify, http_request)
            except Exception as e:
                logger.error("❌ Urgency scoring failed in unified analysis: %s", e)

        results.append(item)

    return UnifiedAnalysisResponse(results=results)
