import asyncio
import logging
from fastapi import APIRouter, HTTPException, Request
from typing import List
from src.models import EmbeddingItem
from src.helpers.async_wrappers import _embed_sync
from src import metrics

router = APIRouter()
logger = logging.getLogger("mistify")


@router.post("/embed")
async def embed_items(req: EmbeddingItem, http_request: Request) -> dict:
    """Compute sentence embedding for a single item."""
    app_state = http_request.state.app_state
    if app_state.embedder is None:
        raise HTTPException(status_code=503, detail="Embeddings model not available")

    # Track embedding request
    metrics.POSTS_PROCESSED_TOTAL.labels(endpoint="embed").inc()

    item = req.dict()
    if not item:
        return {}

    text = str(item.get("content", "") or "")
    if not text:
        return item

    try:
        with metrics.record_operation("embed"):
            # Call embedding function with single text in a list
            vecs = await asyncio.get_running_loop().run_in_executor(
                app_state.thread_pool, _embed_sync, app_state.embedder, [text], 64, True
            )

    except Exception as e:
        metrics.OPERATION_FAILURES_TOTAL.labels(operation="embed", failure_type="exception").inc()
        logger.error("❌ Embedding failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    # Ensure vecs is 2D array with shape (1, embedding_dim)
    # Validate dimensions
    if vecs.ndim != 2:
        logger.error(f"❌ Expected 2D embeddings array, got {vecs.ndim}D with shape {vecs.shape}")
        raise HTTPException(status_code=500, detail=f"Invalid embeddings shape: expected 2D, got {vecs.ndim}D")

    if vecs.shape[0] != 1:
        logger.error(f"❌ Expected 1 embedding, got {vecs.shape[0]}")
        raise HTTPException(status_code=500, detail=f"Expected 1 embedding, got {vecs.shape[0]}")

    # Extract the single embedding vector
    embedding_vector = vecs[0]  # Get first (and only) row
    # Add embeddings to the item
    item["embeddings"] = embedding_vector.tolist()

    return item
