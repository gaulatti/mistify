# service/src/helpers/async_wrappers.py

import asyncio
import logging
import torch
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

async def run_in_executor(loop, executor, func, *args):
    """Helper to run sync functions in a thread pool."""
    return await loop.run_in_executor(executor, func, *args)

def _cluster_sync(texts: List[str], nlp, embedder, classifier, config: Dict = None, debug: bool = False):
    """Synchronous clustering function for thread execution"""
    logger.error("🚨🚨🚨 CLUSTERING FUNCTION CALLED - THIS SHOULD NOT HAPPEN FOR EMBEDDING ENDPOINT! 🚨🚨🚨")
    logger.error(f"🚨 _cluster_sync called with {len(texts)} texts - debug={debug}")
    logger.error("🚨 If you see this message during /embed request, there's a bug in the function call!")
    from .clustering import build_clustering_graph, split_large_communities
    from src.models import ClusterGroup
    import itertools
    import numpy as np
    from networkx.algorithms.community import louvain_communities

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        G, sims, why, topics, entities = build_clustering_graph(texts, nlp, embedder, classifier, show_bar=debug, config=config)
        logger.info(f"🎯 Clustering generated similarity matrix with shape: {sims.shape}")
        logger.info(f"🎯 Similarity matrix total elements: {sims.size}")
        comms = louvain_communities(G, weight=None, resolution=1.0)
        groups = []
        for c in comms:
            groups.extend(split_large_communities(list(c), sims, config))
        groups.sort(key=lambda g: (-len(g), min(g)))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cluster_groups = []
        for gid, idx in enumerate(groups):
            group_texts = [texts[i] for i in sorted(idx)]
            group_topics = [topics[i] for i in idx]
            primary_topic = max(set(group_topics), key=group_topics.count) if group_topics else None
            group_entities = set()
            for i in idx:
                group_entities.update(entities[i])
            group_entities.discard("__NOENT__")
            primary_entities = list(group_entities)[:5]

            if len(idx) > 1:
                group_sims = [sims[i, j] for i, j in itertools.combinations(idx, 2)]
                avg_similarity = float(np.mean(group_sims)) if group_sims else 0.0
            else:
                avg_similarity = 1.0
            
            cluster_groups.append(ClusterGroup(
                group_id=gid,
                texts=group_texts,
                indices=sorted(idx),
                size=len(idx),
                primary_topic=primary_topic,
                primary_entities=primary_entities,
                avg_similarity=avg_similarity
            ))

        result = {
            "groups": cluster_groups,
            "debug_info": {
                "edge_reasons": {f"{i}-{j}": reason for (i, j), reason in why.items()},
                "total_edges": len(why),
                "topics": topics,
                "entities_count": sum(len(e) for e in entities)
            } if debug else None
        }
        return result
    except Exception as e:
        logger.error("❌ Clustering error: %s", e)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e

def _classify_sync(classifier, text: str, labels: List[str]):
    """Synchronous classification function for thread execution"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        max_text_length = 512
        if len(text) > max_text_length:
            text = text[:max_text_length]
        result = classifier(text, candidate_labels=labels)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "out of memory" in str(e):
            logger.warning("GPU out of memory for classification, retrying with shorter text")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            short_text = text[:256] if len(text) > 256 else text
            try:
                return classifier(short_text, candidate_labels=labels)
            except RuntimeError:
                logger.error("Still out of memory even with shorter text")
                raise e
        else:
            raise e

def _translate_sync(translator, text: str, source_lang: Optional[str] = None, target_lang: str = "eng", model_name: str = None):
    """Synchronous translation function for thread execution"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Truncate text if too long
        max_text_length = 400
        if len(text) > max_text_length:
            text = text[:max_text_length]
            logger.warning(f"Text truncated to {max_text_length} characters for translation")
        
        # Determine if this is a Seamless M4T model
        is_seamless = model_name and "seamless" in model_name.lower()
        
        # Language code mapping for Seamless M4T
        seamless_lang_mapping = {
            "en": "eng", "es": "spa", "fr": "fra", "de": "deu", "it": "ita",
            "pt": "por", "ru": "rus", "zh": "cmn", "ja": "jpn", "ko": "kor",
            "ar": "arb", "hi": "hin", "tr": "tur", "pl": "pol", "nl": "nld",
            "he": "heb", "sv": "swe", "da": "dan", "no": "nor", "fi": "fin"
        }
        
        # Standard language mapping for other models
        standard_lang_mapping = {
            "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it",
            "pt": "pt", "ru": "ru", "zh": "zh", "ja": "ja", "ko": "ko",
            "ar": "ar", "hi": "hi", "tr": "tr", "pl": "pl", "nl": "nl",
            "he": "he", "sv": "sv", "da": "da", "no": "no", "fi": "fi"
        }
        
        # Choose appropriate language mapping
        lang_mapping = seamless_lang_mapping if is_seamless else standard_lang_mapping
        
        # Map target language
        mapped_target = lang_mapping.get(target_lang[:2].lower(), "eng" if is_seamless else "en")
        
        # Try translation with proper parameters
        translation_params = {
            "max_length": 512,
            "do_sample": False,
            "num_beams": 1
        }
        
        if is_seamless:
            # Seamless M4T requires both src_lang and tgt_lang
            if source_lang:
                mapped_source = lang_mapping.get(source_lang[:2].lower(), "eng")
            else:
                # Default to English if no source language detected
                mapped_source = "eng"
                
            result = translator(
                text, 
                src_lang=mapped_source, 
                tgt_lang=mapped_target,
                **translation_params
            )
        else:
            # Other models might work with just target language
            if source_lang:
                mapped_source = lang_mapping.get(source_lang[:2].lower(), source_lang)
                # For Helsinki models, use standard format
                if model_name and "helsinki" in model_name.lower():
                    result = translator(text, **translation_params)
                else:
                    result = translator(
                        text, 
                        src_lang=mapped_source, 
                        tgt_lang=mapped_target,
                        **translation_params
                    )
            else:
                result = translator(text, **translation_params)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    except Exception as e:
        logger.error("❌ Translation error: %s", e)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Try with minimal parameters as fallback
        try:
            logger.warning("Retrying translation with minimal parameters")
            if is_seamless:
                # For Seamless M4T, both src_lang and tgt_lang are required
                fallback_source = seamless_lang_mapping.get(source_lang[:2].lower() if source_lang else "en", "eng")
                fallback_target = seamless_lang_mapping.get(target_lang[:2].lower(), "eng")
                result = translator(
                    text,
                    src_lang=fallback_source,
                    tgt_lang=fallback_target,
                    max_length=400
                )
            else:
                # For other models, try without source language specification
                result = translator(text, max_length=400)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return result
        except Exception as fallback_e:
            logger.error("❌ Fallback translation also failed: %s", fallback_e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

def _embed_sync(embedder, texts: List[str], batch_size: int, normalize: bool):
    """Synchronous embedding function."""
    logger.info("🔥 ENTERING _embed_sync function - THIS SHOULD BE THE ONLY FUNCTION CALLED FOR EMBEDDING")
    logger.info(f"🔥 _embed_sync called with {len(texts)} texts")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Log input information
    logger.info(f"📊 Embedding {len(texts)} texts with batch_size={batch_size}")
    
    # Validate inputs
    if not texts:
        logger.warning("⚠️ Empty texts list provided to embedder")
        result = torch.empty((0, 384), dtype=torch.float32).numpy()  # all-MiniLM-L6-v2 has 384 dimensions
        logger.info(f"🔥 EXITING _embed_sync early - empty input, returning shape: {result.shape}")
        return result
    
    # Filter out empty texts and keep track of indices
    non_empty_texts = []
    original_indices = []
    for i, text in enumerate(texts):
        if text and text.strip():
            non_empty_texts.append(text.strip())
            original_indices.append(i)
        else:
            logger.warning(f"⚠️ Empty or whitespace-only text at index {i}")
    
    if not non_empty_texts:
        logger.warning("⚠️ No valid texts after filtering empty ones")
        result = torch.zeros((len(texts), 384), dtype=torch.float32).numpy()  # all-MiniLM-L6-v2 has 384 dimensions
        logger.info(f"🔥 EXITING _embed_sync early - no valid texts, returning shape: {result.shape}")
        return result
    
    try:
        vecs = embedder.encode(
            non_empty_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        ).astype("float32")
        
        # Log output information
        logger.info(f"📊 Generated embeddings shape: {vecs.shape} for {len(non_empty_texts)} non-empty texts")
        logger.info(f"🔥 CRITICAL: This should be ({len(non_empty_texts)}, 384) for all-MiniLM-L6-v2")
        
        # Verify this is NOT a similarity matrix
        if vecs.shape[0] == vecs.shape[1]:
            logger.error(f"🚨 DANGER: Got square matrix {vecs.shape} - this looks like a similarity matrix!")
            logger.error("🚨 This suggests clustering code is running instead of embedding code!")
        
        # If we had empty texts, we need to create a full array with zeros for empty texts
        if len(non_empty_texts) != len(texts):
            full_vecs = torch.zeros((len(texts), vecs.shape[1]), dtype=torch.float32).numpy()
            for i, orig_idx in enumerate(original_indices):
                full_vecs[orig_idx] = vecs[i]
            vecs = full_vecs
            logger.info(f"📊 Padded embeddings to full shape: {vecs.shape}")
        
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        # Return zero embeddings as fallback
        vecs = torch.zeros((len(texts), 384), dtype=torch.float32).numpy()  # all-MiniLM-L6-v2 has 384 dimensions
        logger.warning(f"⚠️ Returning zero embeddings with shape: {vecs.shape}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info(f"🔥 EXITING _embed_sync - final result shape: {vecs.shape}")
    logger.info(f"🔥 EXPECTED: ({len(texts)}, 384), GOT: {vecs.shape}")
    
    return vecs
