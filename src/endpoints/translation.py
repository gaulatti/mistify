import asyncio
import logging
from fastapi import APIRouter, HTTPException, Request
from src.models import TranslationRequest, TranslationResponse
from src.helpers.async_wrappers import _translate_sync

router = APIRouter()
logger = logging.getLogger("unified-text-analysis")

@router.post("/translate", response_model=TranslationResponse)
async def translate_text(req: TranslationRequest, http_request: Request):
    """Translate text to English using Seamless M4T v2"""
    app_state = http_request.state.app_state
    if app_state.translator is None:
        raise HTTPException(status_code=503, detail="Translation model not available")

    async with app_state.translation_lock:
        try:
            result = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    app_state.thread_pool, 
                    _translate_sync, 
                    app_state.translator,
                    req.text, 
                    req.source_language,
                    req.target_language,
                    getattr(app_state, 'translator_model_name', None)
                ),
                timeout=app_state.config["TIMEOUT"] * 2,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Translation request timed out")
        except Exception as e:
            logger.error("❌ Translation error: %s", e)
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

    if result is None:
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
            translated_text = result.get('translation_text', '') or result.get('generated_text', '') or result.get('text', '')
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
            source_language=req.source_language,
            target_language=req.target_language
        )
    except Exception as e:
        logger.error("❌ Failed to parse translation result: %s, result type: %s", e, type(result))
        raise HTTPException(status_code=500, detail="Failed to parse translation result")
