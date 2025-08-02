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
    from .clustering import build_clustering_graph, split_large_communities
    from src.models import ClusterGroup
    import itertools
    import numpy as np
    from networkx.algorithms.community import louvain_communities

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        G, sims, why, topics, entities = build_clustering_graph(texts, nlp, embedder, classifier, show_bar=debug, config=config)
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

def _translate_sync(translator, text: str, source_lang: Optional[str] = None, target_lang: str = "eng"):
    """Synchronous translation function for thread execution"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        max_text_length = 512
        if len(text) > max_text_length:
            text = text[:max_text_length]
        
        lang_mapping = {
            "en": "eng", "es": "spa", "fr": "fra", "de": "deu", "it": "ita",
            "pt": "por", "ru": "rus", "zh": "cmn", "ja": "jpn", "ko": "kor",
            "ar": "arb", "hi": "hin", "tr": "tur", "pl": "pol", "nl": "nld",
            "he": "heb"
        }
        mapped_target = lang_mapping.get(target_lang[:3], "eng")
        
        if source_lang:
            mapped_source = lang_mapping.get(source_lang[:2], source_lang)
            result = translator(text, src_lang=mapped_source, tgt_lang=mapped_target)
        else:
            result = translator(text, tgt_lang=mapped_target)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    except Exception as e:
        logger.error("❌ Translation error: %s", e)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            return translator(text, max_new_tokens=128)
        except Exception as fallback_e:
            logger.error("❌ Fallback translation also failed: %s", fallback_e)
            return None

def _embed_sync(embedder, texts: List[str], batch_size: int, normalize: bool):
    """Synchronous embedding function."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    vecs = embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    ).astype("float32")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vecs
