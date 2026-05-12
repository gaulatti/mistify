import asyncio
import logging
import os
from fastapi import APIRouter, HTTPException, Request
from typing import List
import torch
from sentence_transformers import SentenceTransformer
from src.models import EmbeddingItem
from src.helpers.async_wrappers import _embed_sync
from src import metrics

router = APIRouter()
logger = logging.getLogger("mistify")


def _load_embedder(cache_folder: str):
    """Create sentence-transformer embedder synchronously for thread execution."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return SentenceTransformer(
        "all-MiniLM-L6-v2",
        device=device,
        cache_folder=cache_folder,
    )


async def _ensure_embedder_available(app_state) -> None:
    if app_state.embedder is not None:
        return

    async with app_state.embedding_lock:
        if app_state.embedder is not None:
            return

        load_models_on_startup = os.getenv("LOAD_MODELS_ON_STARTUP", "true").strip().lower() in {"1", "true", "yes"}
        logger.warning(
            "⚠️ /embed called with embedder unavailable. "
            "Attempting lazy load (LOAD_MODELS_ON_STARTUP=%s).",
            load_models_on_startup,
        )

        try:
            app_state.embedder = await asyncio.get_running_loop().run_in_executor(
                app_state.thread_pool,
                _load_embedder,
                str(app_state.config["HF_CACHE"]),
            )
            logger.info("✓ Lazy-loaded sentence transformer embedder for /embed")
        except Exception as e:
            logger.exception("❌ Failed to initialize embedder lazily for /embed")
            raise HTTPException(
                status_code=503,
                detail=(
                    "Embeddings model not available. "
                    f"Lazy initialization failed: {type(e).__name__}: {e}"
                ),
            )


@router.post("/embed")
async def embed_items(req: EmbeddingItem, http_request: Request) -> dict:
    """Compute sentence embedding for a single item."""
    app_state = http_request.state.app_state
    await _ensure_embedder_available(app_state)

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
        logger.exception("❌ Embedding failed for /embed (content_len=%d)", len(text))
        raise HTTPException(status_code=500, detail=f"Embedding failed: {type(e).__name__}: {str(e)}")

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
