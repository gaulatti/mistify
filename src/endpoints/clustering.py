import asyncio
import logging
import time
from fastapi import APIRouter, HTTPException, Request
from src.models import ClusteringRequest, ClusteringResponse, ClusterGroup
from src.helpers.async_wrappers import _cluster_sync

router = APIRouter()
logger = logging.getLogger("unified-text-analysis")

@router.post("/cluster", response_model=ClusteringResponse)
async def cluster_texts(req: ClusteringRequest, http_request: Request):
    """Cluster a list of texts using entity-aware, topic-gated community detection"""
    app_state = http_request.state.app_state
    if not app_state.embedder or not app_state.nlp:
        raise HTTPException(status_code=503, detail="Clustering models not available")
    
    if not req.texts or len(req.texts) < 2:
        return ClusteringResponse(
            total_texts=len(req.texts),
            total_groups=len(req.texts),
            groups=[ClusterGroup(group_id=i, texts=[t], indices=[i], size=1, avg_similarity=1.0) for i, t in enumerate(req.texts)],
            processing_time=0.0
        )

    cluster_config = {
        "similarity_entity": req.similarity_entity,
        "similarity_global": req.similarity_global,
        "big_community_size": req.big_community_size,
        "avg_similarity_min": req.avg_similarity_min
    }
    start_time = time.time()

    async with app_state.clustering_lock:
        try:
            result = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    app_state.thread_pool, 
                    _cluster_sync, 
                    req.texts, 
                    app_state.nlp, app_state.embedder, app_state.classifier,
                    cluster_config,
                    req.debug
                ),
                timeout=app_state.config["TIMEOUT"] * 4,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Clustering request timed out")
        except Exception as e:
            logger.error("❌ Clustering error: %s", e)
            raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

    processing_time = time.time() - start_time
    logger.info("✓ Clustering completed: %d texts -> %d groups in %.2fs", len(req.texts), len(result["groups"]), processing_time)
    return ClusteringResponse(
        total_texts=len(req.texts),
        total_groups=len(result["groups"]),
        groups=result["groups"],
        processing_time=processing_time,
        debug_info=result.get("debug_info")
    )
