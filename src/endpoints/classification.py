import asyncio
import logging
from fastapi import APIRouter, HTTPException, Request
from src.models import ClassificationRequest, ClassificationResponse
from src.helpers.async_wrappers import _classify_sync

router = APIRouter()
logger = logging.getLogger("unified-text-analysis")

@router.post("/classify", response_model=ClassificationResponse)
async def classify_content(req: ClassificationRequest, http_request: Request):
    """Classify the content of the input text using BART"""
    app_state = http_request.state.app_state
    if app_state.classifier is None:
        raise HTTPException(status_code=503, detail="Classification model not available")

    cleaned_text = ' '.join(req.text.replace('\n', ' ').replace('\r', ' ').strip().split())
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Empty text provided for classification")

    labels = req.labels or app_state.config["DEFAULT_CLASSIFICATION_LABELS"]
    if len(labels) < 2:
        labels.append("other")

    async with app_state.classification_lock:
        try:
            result = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    app_state.thread_pool, _classify_sync, app_state.classifier, cleaned_text, labels
                ),
                timeout=app_state.config["TIMEOUT"],
            )
        except asyncio.TimeoutError:
            return ClassificationResponse(label="timeout", score=0.0, full_result={})
        except Exception as e:
            logger.error("❌ Classification error: %s", e)
            return ClassificationResponse(label="error", score=0.0, full_result={"error": str(e)})

    if not result or "scores" not in result or "labels" not in result:
        return ClassificationResponse(label="error", score=0.0, full_result={})

    scores = result["scores"]
    labels_result = result["labels"]
    best_score = scores[0]
    best_label = labels_result[0]

    if best_score < app_state.config["MIN_SCORE"] or (best_score - (scores[1] if len(scores) > 1 else 0.0)) < app_state.config["MIN_MARGIN"]:
        best_label = "uncertain"

    logger.info("✓ Classification: label=%s, score=%.2f", best_label, best_score)
    return ClassificationResponse(label=best_label, score=float(best_score), full_result=result)
