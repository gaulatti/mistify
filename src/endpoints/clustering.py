import asyncio
import logging
import time
from typing import List, Optional, Union
from fastapi import APIRouter, HTTPException, Request
from src.models import (
    ClusteringRequest, ClusteringResponse, ClusterGroup,
    PostClusteringRequest, PostClusteringResponse, PostClusterGroup,
    PostData, ClusteredPost
)
from src.helpers.async_wrappers import _cluster_sync
from src import metrics

router = APIRouter()
logger = logging.getLogger("mistify")


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math

    if not a or not b:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _filter_candidate_posts(
    main_post: PostData,
    candidates: List[PostData],
    embedder,
    min_similarity: float = 0.45,
    max_candidates: int = 10,
) -> List[PostData]:
    """
    Pre-filter candidate posts before expensive graph clustering.

    Uses cosine similarity between the main post embedding and each candidate.
    Candidates without precomputed embeddings are embedded on the fly. This
    removes low-similarity noise that otherwise causes false merges or
    dilutes real event clusters.
    """
    if not candidates:
        return []

    main_embedding = getattr(main_post, "embeddings", None) or getattr(main_post, "embedding", None)
    if not main_embedding:
        logger.warning("⚠️ Main post has no embedding; skipping candidate pre-filter")
        return candidates[:max_candidates]

    contents_to_embed = []
    candidate_indices = []
    precomputed = []

    for idx, candidate in enumerate(candidates):
        cand_emb = getattr(candidate, "embeddings", None) or getattr(candidate, "embedding", None)
        if cand_emb:
            precomputed.append((idx, cand_emb))
        else:
            contents_to_embed.append(candidate.content)
            candidate_indices.append(idx)

    computed_embeddings = []
    if contents_to_embed:
        try:
            computed_embeddings = embedder.encode(
                contents_to_embed,
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).tolist()
        except Exception as e:
            logger.warning("⚠️ Failed to embed candidate posts for pre-filter: %s", e)
            return candidates[:max_candidates]

    idx_to_emb = dict(precomputed)
    for idx, emb in zip(candidate_indices, computed_embeddings):
        idx_to_emb[idx] = emb

    scored = []
    for idx, candidate in enumerate(candidates):
        emb = idx_to_emb.get(idx)
        if emb is None:
            continue
        sim = _cosine_similarity(main_embedding, emb)
        scored.append((sim, candidate))
        logger.debug("🔍 Candidate pre-filter: post=%s sim=%.3f", candidate.id, sim)

    scored.sort(key=lambda x: x[0], reverse=True)
    filtered = [c for sim, c in scored if sim >= min_similarity]

    if len(filtered) > max_candidates:
        filtered = filtered[:max_candidates]

    logger.info(
        "🔍 Pre-filter: %d candidates -> %d after similarity filter (min=%.2f, max=%d)",
        len(candidates),
        len(filtered),
        min_similarity,
        max_candidates,
    )
    return filtered


@router.post("/cluster", response_model=PostClusteringResponse)
async def cluster_texts(req: PostData, http_request: Request):
    """Cluster a single post with its similar posts using entity-aware, topic-gated community detection"""
    app_state = http_request.state.app_state
    if not app_state.embedder or not app_state.nlp:
        raise HTTPException(status_code=503, detail="Clustering models not available")

    # Debug: Log the incoming request structure
    logger.info(f"🔍 Clustering request - Post ID: {req.id}, Similar posts: {type(req.similarPosts)}")

    # Pre-filter candidates to remove low-similarity noise before graph clustering.
    # This keeps the graph focused on plausible same-event posts and reduces both
    # under-clustering (diluted by noise) and over-clustering (weak same-subject
    # candidates that only share broad entities).
    candidate_posts = req.similarPosts or []
    filtered_candidates = _filter_candidate_posts(
        req,
        candidate_posts,
        app_state.embedder,
        min_similarity=0.45,
        max_candidates=10,
    )

    # Prepare texts for clustering: main post + similar posts
    texts = [req.content]  # Main post first
    all_posts = [req]  # Keep reference to all posts for building response

    # Add filtered similar posts to the clustering candidates
    if filtered_candidates:
        logger.info(f"🔍 Processing {len(filtered_candidates)} filtered similar posts")
        for similar_post in filtered_candidates:
            texts.append(similar_post.content)
            all_posts.append(similar_post)
    else:
        logger.warning(f"⚠️ No similar posts passed pre-filter for post {req.id}")
    
    # Track batch size and posts processed
    batch_size = len(texts)
    metrics.POSTS_BATCH_SIZE.labels(endpoint="cluster").observe(batch_size)
    metrics.POSTS_PROCESSED_TOTAL.labels(endpoint="cluster").inc(batch_size)
    
    logger.info(f"🔍 Total texts for clustering: {len(texts)}")
    
    # If we only have the main post (no similar posts), return single cluster
    if len(texts) == 1:
        return PostClusteringResponse(
            total_posts=1,
            total_groups=1,
            group=PostClusterGroup(
                group_id=0,
                posts=[ClusteredPost(
                    id=req.id,
                    hash=req.hash,
                    content=req.content
                )],
                size=1,
                primary_topic=None,
                primary_entities=None,
                avg_similarity=1.0
            ),
            processing_time=0.0
        )

    # Extract precomputed entities from the posts if available
    precomputed_entities = []
    for post in all_posts:
        if hasattr(post, 'primary_entities') and post.primary_entities:
            # Convert list to set for compatibility with clustering algorithm
            precomputed_entities.append(set(post.primary_entities))
        else:
            precomputed_entities.append(None)  # Will trigger entity extraction

    # Check if we have complete entity data
    has_complete_entities = all(entities is not None for entities in precomputed_entities)

    # Extract precomputed embeddings and category information if available
    precomputed_embeddings = []
    precomputed_categories = []

    for post in all_posts:
        # Extract embeddings if available
        if hasattr(post, 'embeddings') and post.embeddings:
            precomputed_embeddings.append(post.embeddings)
        else:
            precomputed_embeddings.append(None)  # Will trigger embedding computation

        # Extract category information if available
        if hasattr(post, 'categories_relation') and post.categories_relation:
            categories = [cat.name for cat in post.categories_relation]
            precomputed_categories.append(categories)
        else:
            precomputed_categories.append([])

    # Track outsourced embedding coverage; helper will backfill missing entries.
    has_any_embeddings = any(emb is not None for emb in precomputed_embeddings)
    has_complete_embeddings = all(emb is not None for emb in precomputed_embeddings)
    has_category_data = any(len(cats) > 0 for cats in precomputed_categories)

    logger.info(
        "🔍 Precomputed data: embeddings_any=%s embeddings_complete=%s categories=%s",
        has_any_embeddings,
        has_complete_embeddings,
        has_category_data,
    )

    # Clustering configuration tuned for same-event detection while avoiding
    # same-subject/different-event merges.
    cluster_config = {
        "similarity_entity": 0.35,  # Base threshold for entity-aided clustering
        "similarity_global": 0.55,  # Pure-semantic same-event threshold (cross-lingual safe)
        "big_community_size": 20,   # Allow slightly larger event communities
        "avg_similarity_min": 0.45, # Prevent over-splitting legitimate event clusters
        "topic_strict_mode": False, # Coarse topic labels are too noisy to gate on
        "entity_context_weight": 0.20, # Moderate weight for event-specific entities
        "min_shared_entities": 1,   # Require at least one shared key entity
        "domain_filtering": False,  # Disable domain filtering - focus on events not domains
        "event_specific_mode": True, # Enable event-specific clustering
        "precomputed_entities": precomputed_entities if has_complete_entities else None,
        "precomputed_embeddings": precomputed_embeddings if has_any_embeddings else None,
        "precomputed_categories": precomputed_categories if has_category_data else None,
    }
    debug = False
    start_time = time.time()

    with metrics.record_operation("cluster"):
        async with app_state.clustering_lock:
            try:
                result = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        app_state.thread_pool,
                        _cluster_sync,
                        texts,
                        app_state.nlp, 
                        app_state.embedder, 
                        app_state.classifier,
                        cluster_config,
                        debug
                    ),
                    timeout=app_state.config["TIMEOUT"] * 4,
                )
            except asyncio.TimeoutError:
                metrics.OPERATION_FAILURES_TOTAL.labels(operation="cluster", failure_type="timeout").inc()
                raise HTTPException(status_code=408, detail="Clustering request timed out")
            except Exception as e:
                metrics.OPERATION_FAILURES_TOTAL.labels(operation="cluster", failure_type="exception").inc()
                logger.error("❌ Clustering error: %s", e)
                raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

    processing_time = time.time() - start_time
    logger.info("✓ Clustering completed: %d posts -> %d groups in %.2fs", len(texts), len(result["groups"]), processing_time)
    
    # Find which cluster contains the main post (index 0)
    main_post_group = None
    for group in result["groups"]:
        if 0 in group.indices:  # Main post is at index 0
            main_post_group = group
            break
    
    # If no group found (shouldn't happen), create a single-item group
    if main_post_group is None:
        return PostClusteringResponse(
            total_posts=len(texts),
            total_groups=1,
            group=PostClusterGroup(
                group_id=0,
                posts=[ClusteredPost(
                    id=req.id,
                    hash=req.hash,
                    content=req.content
                )],
                size=1,
                primary_topic=None,
                primary_entities=None,
                avg_similarity=1.0
            ),
            processing_time=processing_time
        )

    # Build the clustered posts array from the group indices
    clustered_posts = []
    for index in main_post_group.indices:
        # Get the post from our all_posts array using the index
        post = all_posts[index]
        clustered_posts.append(ClusteredPost(
            id=post.id,
            hash=post.hash,
            content=post.content
        ))

    # Create the response with the group containing the main post and clustered similar posts
    return PostClusteringResponse(
        total_posts=len(texts),
        total_groups=len(result["groups"]),
        group=PostClusterGroup(
            group_id=main_post_group.group_id,
            posts=clustered_posts,
            size=main_post_group.size,
            primary_topic=main_post_group.primary_topic,
            primary_entities=main_post_group.primary_entities,
            avg_similarity=main_post_group.avg_similarity
        ),
        processing_time=processing_time,
        debug_info=result.get("debug_info")
    )
