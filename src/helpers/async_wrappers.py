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
    logger.info(f"� Starting clustering with {len(texts)} texts")
    from .clustering import build_clustering_graph, split_large_communities
    from src.models import ClusterGroup
    import itertools
    import numpy as np
    from networkx.algorithms.community import louvain_communities

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("🔧 Building clustering graph...")
        G, sims, why, topics, entities = build_clustering_graph(texts, nlp, embedder, classifier, show_bar=debug,
                                                                config=config)
        logger.info(f"🔧 Graph built with {len(G.nodes())} nodes and {len(G.edges())} edges")
        
        # Check if graph is empty (no connections between texts)
        if len(G.nodes()) == 0 or len(G.edges()) == 0:
            logger.warning("⚠️ Empty graph detected - no connections between texts, creating individual clusters")
            # Create individual clusters for each text
            comms = [frozenset([i]) for i in range(len(texts))]
        else:
            logger.info("🔧 Running Louvain community detection...")
            comms = louvain_communities(G, weight=None, resolution=1.0)
        logger.info(f"🔧 Found {len(comms)} initial communities")
        
        logger.info("🔧 Splitting large communities...")
        groups = []
        for i, c in enumerate(comms):
            logger.debug(f"🔧 Processing community {i} with {len(c)} members")
            try:
                split_groups = split_large_communities(list(c), sims, config)
                groups.extend(split_groups)
                logger.debug(f"🔧 Community {i} split into {len(split_groups)} groups")
            except Exception as split_e:
                logger.error(f"❌ Error splitting community {i}: {split_e}")
                # Add the original community as-is if splitting fails
                groups.append(list(c))
        
        groups.sort(key=lambda g: (-len(g), min(g)))
        logger.info(f"🔧 Final result: {len(groups)} groups after splitting")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("🔧 Creating cluster groups...")
        cluster_groups = []
        for gid, idx in enumerate(groups):
            try:
                logger.debug(f"🔧 Processing group {gid} with indices: {idx}")
                
                # Validate indices
                if not idx:
                    logger.warning(f"⚠️ Empty group {gid}, skipping")
                    continue
                    
                # Check if all indices are valid
                max_idx = max(idx)
                if max_idx >= len(texts):
                    logger.error(f"❌ Invalid index {max_idx} in group {gid}, max allowed: {len(texts)-1}")
                    continue
                
                group_texts = [texts[i] for i in sorted(idx)]
                group_topics = [topics[i] for i in idx]
                primary_topic = max(set(group_topics), key=group_topics.count) if group_topics else None
                group_entities = set()
                for i in idx:
                    group_entities.update(entities[i])
                group_entities.discard("__NOENT__")
                primary_entities = list(group_entities)[:5]

                # Safe similarity calculation
                if len(idx) > 1:
                    try:
                        group_sims = [sims[i, j] for i, j in itertools.combinations(idx, 2)]
                        if group_sims:
                            avg_similarity = float(np.mean(group_sims))
                        else:
                            logger.warning(f"⚠️ No similarities found for group {gid}")
                            avg_similarity = 0.0
                    except Exception as sim_e:
                        logger.error(f"❌ Error calculating similarities for group {gid}: {sim_e}")
                        avg_similarity = 0.0
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
                logger.debug(f"✓ Group {gid} created successfully")
                
            except Exception as group_e:
                logger.error(f"❌ Error creating group {gid}: {group_e}")
                # Continue with other groups
                continue

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


def _translate_sync(translator, text: str, source_lang: Optional[str] = None, target_lang: str = "eng",
                    model_name: str = None):
    """Synchronous translation function with progressive fallback strategies"""

    # Initialize variables
    is_seamless = model_name and "seamless" in model_name.lower()
    input_length = len(text.split())

    # Model-specific limits based on actual capabilities
    if is_seamless:
        # Seamless M4T v2 can handle longer sequences (up to ~1024 tokens)
        max_input_tokens = 400  # More conservative limit to avoid memory issues
        # For seamless, we don't use max_length parameter as it conflicts
        suggested_max_length = None
    else:
        # Helsinki-NLP and other models typically handle 512 tokens well
        max_input_tokens = 350
        suggested_max_length = max(100, min(400, int(input_length * 1.5)))

    # Language code mappings
    seamless_lang_mapping = {
        "en": "eng", "es": "spa", "fr": "fra", "de": "deu", "it": "ita",
        "pt": "por", "ru": "rus", "zh": "cmn", "ja": "jpn", "ko": "kor",
        "ar": "arb", "hi": "hin", "tr": "tur", "pl": "pol", "nl": "nld",
        "he": "heb", "sv": "swe", "da": "dan", "no": "nor", "fi": "fin"
    }

    standard_lang_mapping = {
        "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it",
        "pt": "pt", "ru": "ru", "zh": "zh", "ja": "ja", "ko": "ko",
        "ar": "ar", "hi": "hi", "tr": "tr", "pl": "pl", "nl": "nl",
        "he": "he", "sv": "sv", "da": "da", "no": "no", "fi": "fi"
    }

    def _attempt_translation(text_to_translate, attempt_name=""):
        """Helper function to attempt translation with proper error handling"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if is_seamless:
                # Seamless M4T requires both src_lang and tgt_lang
                if source_lang:
                    mapped_source = seamless_lang_mapping.get(source_lang[:2].lower(), "eng")
                else:
                    mapped_source = "eng"

                # For Seamless M4T pipeline, use generation parameters
                input_tokens = len(text_to_translate.split())

                logger.info(f"Text: {text_to_translate}, Seamless translation: {mapped_source} -> eng, input_tokens: {input_tokens}")

                # For Hugging Face pipeline, pass generation parameters correctly
                result = translator(
                    text_to_translate,
                    src_lang=mapped_source,
                    tgt_lang="eng",
                    **{"max_length": 1000, "do_sample": False, "num_beams": 1}
                )
            else:
                # For other models, use standard translation parameters
                # Calculate appropriate max_length based on input
                input_tokens = len(text_to_translate.split())
                dynamic_max_length = max(50, min(300, input_tokens * 2))

                translation_params = {
                    "max_length": dynamic_max_length,
                    "do_sample": False,
                    "num_beams": 1,
                    "early_stopping": True
                }

                if source_lang:
                    mapped_source = standard_lang_mapping.get(source_lang[:2].lower(), source_lang)
                    if model_name and "helsinki" in model_name.lower():
                        result = translator(text_to_translate, **translation_params)
                    else:
                        result = translator(
                            text_to_translate,
                            src_lang=mapped_source,
                            tgt_lang="en",  # Always translate to English
                            **translation_params
                        )
                else:
                    result = translator(text_to_translate, **translation_params)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"✓ Translation successful {attempt_name}: {len(text_to_translate.split())} words")
            return result

        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.warning(f"Translation attempt failed {attempt_name}: {e}")
            return None

    # Strategy 1: Try with full text first (if not extremely long)
    words = text.split()
    if len(words) <= max_input_tokens:
        logger.info(f"Attempting translation with full text ({len(words)} words)")
        result = _attempt_translation(text, "(full text)")
        if result is not None:
            return result

    # Strategy 2: Smart truncation to sentence boundaries
    if len(words) > max_input_tokens:
        logger.info(f"Text too long ({len(words)} words), attempting smart truncation to {max_input_tokens} words")

        # Try to truncate at sentence boundaries
        sentences = text.split('. ')
        truncated_sentences = []
        word_count = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            if word_count + len(sentence_words) <= max_input_tokens:
                truncated_sentences.append(sentence)
                word_count += len(sentence_words)
            else:
                break

        if truncated_sentences:
            smart_truncated = '. '.join(truncated_sentences)
            if not smart_truncated.endswith('.'):
                smart_truncated += '.'

            logger.info(f"Smart truncation: {len(words)} -> {len(smart_truncated.split())} words")
            result = _attempt_translation(smart_truncated, "(smart truncated)")
            if result is not None:
                return result

    # Strategy 3: Hard truncation as fallback
    logger.warning(f"Falling back to hard truncation at {max_input_tokens} words")
    hard_truncated = " ".join(words[:max_input_tokens])
    result = _attempt_translation(hard_truncated, "(hard truncated)")
    if result is not None:
        return result

    # Strategy 4: Ultra-conservative fallback
    logger.warning("Trying ultra-conservative approach with 150 words")
    ultra_conservative = " ".join(words[:150])
    result = _attempt_translation(ultra_conservative, "(ultra-conservative)")
    if result is not None:
        return result

    # Strategy 5: Last resort with minimal parameters
    try:
        logger.warning("Last resort: minimal parameters with 80 words")
        minimal_text = " ".join(words[:80])

        if is_seamless:
            fallback_source = seamless_lang_mapping.get(source_lang[:2].lower() if source_lang else "en", "eng")
            fallback_target = seamless_lang_mapping.get(target_lang[:3].lower(), "eng")
            result = translator(
                minimal_text,
                src_lang=fallback_source,
                tgt_lang=fallback_target,
                max_new_tokens=40,
                do_sample=False
            )
        else:
            result = translator(minimal_text, max_length=100)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    except Exception as final_e:
        logger.error("❌ All translation strategies failed: %s", final_e)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None


def _embed_sync(embedder, texts: List[str], batch_size: int, normalize: bool):
    """Synchronous embedding function."""

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Log input information

    # Validate inputs
    if not texts:
        logger.warning("⚠️ Empty texts list provided to embedder")
        result = torch.empty((0, 384), dtype=torch.float32).numpy()  # all-MiniLM-L6-v2 has 384 dimensions
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
        return result

    try:
        vecs = embedder.encode(
            non_empty_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        ).astype("float32")

        # Log output information

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

    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        # Return zero embeddings as fallback
        vecs = torch.zeros((len(texts), 384), dtype=torch.float32).numpy()  # all-MiniLM-L6-v2 has 384 dimensions
        logger.warning(f"⚠️ Returning zero embeddings with shape: {vecs.shape}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vecs
