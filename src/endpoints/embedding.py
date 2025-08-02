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
        logger.info("🎯 About to call _embed_sync from embedding endpoint")
        logger.info(f"🎯 Function being called: {_embed_sync.__name__}")
        logger.info(f"🎯 Input texts count: {len(texts)}")
        
        vecs = await asyncio.get_running_loop().run_in_executor(
            app_state.thread_pool, _embed_sync, app_state.embedder, texts, 64, True
        )
        
        logger.info("🎯 _embed_sync call completed")
        logger.info(f"🎯 Returned object type: {type(vecs)}")
        logger.info(f"🎯 Returned object shape (if numpy): {getattr(vecs, 'shape', 'N/A')}")
        
    except Exception as e:
        logger.error("❌ Embedding failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    # Ensure vecs is 2D array with shape (num_texts, embedding_dim)
    logger.info(f"📊 Embeddings shape: {vecs.shape}, Items count: {len(items)}")
    logger.info(f"📊 Expected shape: ({len(items)}, 384) for all-MiniLM-L6-v2")
    
    # CRITICAL CHECK: Detect if we got a similarity matrix instead of embeddings
    if vecs.shape[0] == vecs.shape[1] == len(items):
        logger.error(f"🚨🚨🚨 FOUND THE BUG! Got square matrix {vecs.shape} instead of embeddings!")
        logger.error(f"🚨 This is a {len(items)}×{len(items)} = {vecs.shape[0] * vecs.shape[1]} similarity matrix!")
        logger.error(f"🚨 Expected: ({len(items)}, 384) embedding matrix")
        logger.error(f"🚨 This means clustering code ran instead of embedding code!")
        raise HTTPException(status_code=500, detail=f"BUG: Got similarity matrix {vecs.shape} instead of embeddings ({len(items)}, 384)")
    
    # Validate that we have the right number of embeddings
    if len(vecs) != len(items):
        logger.error(f"❌ Mismatch: {len(items)} items but {len(vecs)} embeddings")
        raise HTTPException(status_code=500, detail=f"Embedding count mismatch: {len(items)} items vs {len(vecs)} embeddings")
    
    # Ensure vecs is 2D and iterate properly
    if vecs.ndim != 2:
        logger.error(f"❌ Expected 2D embeddings array, got {vecs.ndim}D with shape {vecs.shape}")
        raise HTTPException(status_code=500, detail=f"Invalid embeddings shape: expected 2D, got {vecs.ndim}D")
    
    # Log detailed shape information
    num_embeddings, embedding_dim = vecs.shape
    logger.info(f"📊 Processing {num_embeddings} embeddings, each with {embedding_dim} dimensions")
    total_elements = num_embeddings * embedding_dim
    logger.info(f"📊 Total elements in embedding array: {total_elements}")
    
    # Safely assign embeddings - each row is one item's embedding vector
    for i, (item, embedding_vector) in enumerate(zip(items, vecs)):
        vector_length = len(embedding_vector)
        item["embeddings"] = embedding_vector.tolist()
        logger.debug(f"Item {i}: embedding length = {vector_length}")
        
        # Check if this is the source of the issue
        if i == 0:  # Log details for first item
            logger.info(f"📊 First embedding vector shape: {embedding_vector.shape}")
            logger.info(f"📊 First embedding vector length after tolist(): {len(item['embeddings'])}")
    
    logger.info(f"✓ Successfully processed {len(items)} items with embeddings")
    
    # Final validation of response structure
    response_items = len(items)
    logger.info(f"📊 Final response contains {response_items} items")
    
    # Check the structure of the first item if available
    if items:
        first_item = items[0]
        has_embeddings = "embeddings" in first_item
        embeddings_length = len(first_item.get("embeddings", [])) if has_embeddings else 0
        logger.info(f"📊 First item has embeddings: {has_embeddings}, length: {embeddings_length}")
        
        # Log all keys in the first item for debugging
        logger.info(f"📊 First item keys: {list(first_item.keys())}")
    
    return items
