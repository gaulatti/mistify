import asyncio
import logging
import time
from typing import Union
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


@router.post("/cluster", response_model=PostClusteringResponse)
async def cluster_texts(req: PostData, http_request: Request):
    """Cluster a single post with its similar posts using entity-aware, topic-gated community detection"""
    app_state = http_request.state.app_state
    if not app_state.embedder or not app_state.nlp:
        raise HTTPException(status_code=503, detail="Clustering models not available")

    # Debug: Log the incoming request structure
    logger.info(f"🔍 Clustering request - Post ID: {req.id}, Similar posts: {type(req.similarPosts)}")
    
    # Prepare texts for clustering: main post + similar posts
    texts = [req.content]  # Main post first
    all_posts = [req]  # Keep reference to all posts for building response
    
    # Add similar posts to the clustering candidates (with null check)
    if req.similarPosts is not None:
        logger.info(f"🔍 Processing {len(req.similarPosts)} similar posts")
        for similar_post in req.similarPosts:
            texts.append(similar_post.content)
            all_posts.append(similar_post)
    else:
        logger.warning(f"⚠️ No similar posts provided for post {req.id}")
    
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

    # Check if we have complete embedding data
    has_complete_embeddings = all(emb is not None for emb in precomputed_embeddings)
    has_category_data = any(len(cats) > 0 for cats in precomputed_categories)

    logger.info(f"🔍 Precomputed data: embeddings={has_complete_embeddings}, categories={has_category_data}")

    # Clustering configuration balanced for same-event detection
    cluster_config = {
        "similarity_entity": 0.35,  # Moderate entity weight for event-specific entities
        "similarity_global": 0.70,  # High but not extreme threshold for same-event clustering
        "big_community_size": 15,   # Small communities for tighter event clustering
        "avg_similarity_min": 0.65, # Moderate minimum for same-event stories
        "topic_strict_mode": False, # Allow cross-topic clustering for same events
        "entity_context_weight": 0.20, # Moderate weight for event-specific entities
        "min_shared_entities": 1,   # Require at least one shared key entity
        "domain_filtering": False,  # Disable domain filtering - focus on events not domains
        "event_specific_mode": True, # Enable event-specific clustering
        "precomputed_entities": precomputed_entities if has_complete_entities else None,
        "precomputed_embeddings": precomputed_embeddings if has_complete_embeddings else None,
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
                raise HTTPException(status_code=408, detail="Clustering request timed out")
            except Exception as e:
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
