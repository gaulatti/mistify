import asyncio
import logging
from fastapi import APIRouter, HTTPException, Request
from src.models import TextGenerationRequest, TextGenerationResponse
from src.helpers.async_wrappers import _generate_text_sync

router = APIRouter()
logger = logging.getLogger("mistify")


@router.post("/generate/text")
async def generate_text(req: TextGenerationRequest, http_request: Request) -> TextGenerationResponse:
    """Generate text using Flan-T5-XL model based on a prompt."""
    app_state = http_request.state.app_state
    if app_state.text_generator is None:
        raise HTTPException(status_code=503, detail="Text generation model not available")

    try:
        # Generate text using the pipeline directly with the provided prompt
        generated_text = await asyncio.get_running_loop().run_in_executor(
            app_state.thread_pool, _generate_text_sync, app_state.text_generator, req.prompt
        )

        return TextGenerationResponse(
            prompt=req.prompt,
            generated_text=generated_text
        )

    except Exception as e:
        logger.error("❌ Text generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")
