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
        # Sanitize and clamp parameters
        params = {
            "max_new_tokens": int(min(800, max(16, req.max_new_tokens))),
            "min_new_tokens": int(max(0, min(req.min_new_tokens, 512))),
            "num_beams": int(max(1, min(req.num_beams, 6))),
            "temperature": float(req.temperature),
            "do_sample": bool(req.do_sample),
            "no_repeat_ngram_size": int(max(0, min(req.no_repeat_ngram_size, 10))),
            "length_penalty": float(min(2.0, max(0.1, req.length_penalty))),
        }
        # Adjust sampling/beam settings interplay
        if not params["do_sample"] and params["temperature"] != 0.7:
            # Temperature irrelevant when not sampling; keep for transparency
            pass
        if params["do_sample"] and params["num_beams"] > 1:
            # Beam search + sampling can be expensive; allow but could warn
            logger.debug("Beam search with sampling enabled; potential latency increase")

        generated_text = await asyncio.get_running_loop().run_in_executor(
            app_state.thread_pool, _generate_text_sync, app_state.text_generator, req.prompt, params
        )

        return TextGenerationResponse(
            prompt=req.prompt,
            generated_text=generated_text,
            used_params={k: str(v) for k, v in params.items()}
        )

    except Exception as e:
        logger.error("❌ Text generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")
