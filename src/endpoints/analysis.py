import logging
from fastapi import APIRouter, Request
from src.models import (
    UnifiedAnalysisRequest, UnifiedAnalysisResponse,
    LanguageDetectionRequest, TranslationRequest, ClassificationRequest,
    TranslationResponse
)
from .language import detect_language
from .translation import translate_text
from .classification import classify_content

router = APIRouter()
logger = logging.getLogger("mistify")


@router.post("/analyze", response_model=UnifiedAnalysisResponse)
async def unified_analysis(req: UnifiedAnalysisRequest, http_request: Request):
    """Perform language detection, content classification, and translation on the input text"""
    app_state = http_request.state.app_state
    response = UnifiedAnalysisResponse(text=req.text)
    detected_language = None

    if req.detect_language and app_state.fasttext_model:
        try:
            lang_req = LanguageDetectionRequest(text=req.text, k=req.language_count)
            lang_resp = await detect_language(lang_req, http_request)
            response.language_detection = lang_resp
            if lang_resp.languages:
                detected_language = lang_resp.languages[0]
        except Exception as e:
            logger.error("❌ Language detection failed in unified analysis: %s", e)

    if req.translate_to_english and app_state.translator:
        try:
            if not detected_language or detected_language.lower() not in ['en', 'eng']:
                trans_req = TranslationRequest(text=req.text, source_language=detected_language)
                trans_resp = await translate_text(trans_req, http_request)
                response.translation = trans_resp
            else:
                response.translation = TranslationResponse(
                    original_text=req.text,
                    translated_text=req.text,
                    source_language=detected_language,
                    target_language="eng"
                )
        except Exception as e:
            logger.error("❌ Translation failed in unified analysis: %s", e)

    if req.classify_content and app_state.classifier:
        try:
            text_to_classify = response.translation.translated_text if response.translation else req.text
            class_req = ClassificationRequest(text=text_to_classify, labels=req.classification_labels)
            class_resp = await classify_content(class_req, http_request)
            response.content_classification = class_resp
        except Exception as e:
            logger.error("❌ Content classification failed in unified analysis: %s", e)

    return response
