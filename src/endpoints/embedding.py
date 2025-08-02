import asyncio
import logging
from fastapi import APIRouter, HTTPException, Request
from typing import List
from src.models import EmbeddingItem
from src.helpers.async_wrappers import _embed_sync

router = APIRouter()
logger = logging.getLogger("unified-text-analysis")

@router.post("/embed")
async def embed_items(req: List[EmbeddingItem], http_request: Request) -> List[dict]:
    """Compute sentence embeddings for an array of items."""
    app_state = http_request.state.app_state
    if app_state.embedder is None:
        raise HTTPException(status_code=503, detail="Embeddings model not available")

    items = [item.dict() for item in req] or []
    if not items:
        return []

    texts = [str(it.get("content", "") or "") for it in items]
    if not texts:
        return items

    try:
        vecs = await asyncio.get_running_loop().run_in_executor(
            app_state.thread_pool, _embed_sync, app_state.embedder, texts, 64, True
        )
    except Exception as e:
        logger.error("❌ Embedding failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    for it, v in zip(items, vecs):
        it["embeddings"] = v.tolist()
    return items
