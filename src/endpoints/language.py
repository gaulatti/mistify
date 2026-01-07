import logging
from fastapi import APIRouter, HTTPException, Request
from src.models import LanguageDetectionRequest, LanguageDetectionResponse
from src import metrics

router = APIRouter()
logger = logging.getLogger("mistify")


@router.post("/detect", response_model=LanguageDetectionResponse)
async def detect_language(req: LanguageDetectionRequest, http_request: Request):
    """Detect the language(s) of the input text using FastText"""
    app_state = http_request.state.app_state
    if app_state.fasttext_model is None:
        raise HTTPException(status_code=503, detail="FastText model not available")

    try:
        # Track language detection request
        metrics.POSTS_PROCESSED_TOTAL.labels(endpoint="detect").inc()
        
        cleaned_text = ' '.join(req.text.replace('\n', ' ').replace('\r', ' ').strip().split())
        if not cleaned_text:
            raise HTTPException(status_code=400, detail="Empty text provided")

        with metrics.record_operation("language_detect"):
            labels, probs = app_state.fasttext_model.predict(cleaned_text, k=req.k)
        languages = [label.replace("__label__", "") for label in labels]
        probabilities = [round(float(p), 4) for p in probs]

        logger.info("✓ Language detection completed: %s", languages)
        return LanguageDetectionResponse(languages=languages, probabilities=probabilities)
    except HTTPException:
        raise
    except Exception as e:
        metrics.OPERATION_FAILURES_TOTAL.labels(operation="language_detect", failure_type="exception").inc()
        logger.error("❌ Language detection error: %s", e)
        raise HTTPException(status_code=500, detail=f"Language detection failed: {str(e)}")
