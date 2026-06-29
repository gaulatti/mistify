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
        "af": "afr", "am": "amh", "ar": "arb", "arz": "arz", "as": "asm",
        "az": "azj", "be": "bel", "bg": "bul", "bn": "ben", "bs": "bos",
        "ca": "cat", "ceb": "ceb", "cs": "ces", "cy": "cym", "da": "dan",
        "de": "deu", "el": "ell", "en": "eng", "es": "spa", "et": "est",
        "eu": "eus", "fa": "pes", "ff": "fuv", "fr": "fra", "ga": "gle",
        "gl": "glg", "gu": "guj", "he": "heb", "hi": "hin", "hr": "hrv",
        "hu": "hun", "hy": "hye", "id": "ind", "ig": "ibo", "is": "isl",
        "it": "ita", "ja": "jpn", "jv": "jav", "ka": "kat", "kk": "kaz",
        "km": "khm", "kn": "kan", "ko": "kor", "ky": "kir", "lo": "lao",
        "lt": "lit", "lv": "lvs", "mk": "mkd", "ml": "mal", "mn": "khk",
        "mr": "mar", "ms": "zlm", "mt": "mlt", "my": "mya", "ne": "npi",
        "nl": "nld", "nn": "nno", "no": "nob", "ny": "nya", "or": "ory",
        "pa": "pan", "pl": "pol", "ps": "pbt", "pt": "por", "ro": "ron",
        "ru": "rus", "sd": "snd", "sk": "slk", "sl": "slv", "sn": "sna",
        "so": "som", "sr": "srp", "sv": "swe", "sw": "swh", "ta": "tam",
        "te": "tel", "tg": "tgk", "th": "tha", "tl": "tgl", "tr": "tur",
        "uk": "ukr", "ur": "urd", "uz": "uzn", "vi": "vie", "yo": "yor",
        "zh": "cmn", "zu": "zul",
        "afr": "afr", "amh": "amh", "arb": "arb", "ary": "ary", "asm": "asm",
        "azj": "azj", "bel": "bel", "ben": "ben", "bos": "bos", "bul": "bul",
        "cat": "cat", "ces": "ces", "ckb": "ckb", "cmn": "cmn", "cmn-hant": "cmn_Hant",
        "cym": "cym", "deu": "deu", "ell": "ell", "eng": "eng", "est": "est",
        "eus": "eus", "fuv": "fuv", "gaz": "gaz", "gle": "gle", "glg": "glg",
        "guj": "guj", "heb": "heb", "hin": "hin", "hrv": "hrv", "hun": "hun",
        "hye": "hye", "ibo": "ibo", "ind": "ind", "isl": "isl", "jav": "jav",
        "jpn": "jpn", "kan": "kan", "kat": "kat", "kaz": "kaz", "khk": "khk",
        "khm": "khm", "kir": "kir", "lao": "lao", "lug": "lug", "luo": "luo",
        "mai": "mai", "mal": "mal", "mar": "mar", "mkd": "mkd", "mlt": "mlt",
        "mni": "mni", "mya": "mya", "nld": "nld", "nno": "nno", "nob": "nob",
        "npi": "npi", "ory": "ory", "pan": "pan", "pbt": "pbt", "pes": "pes",
        "pol": "pol", "por": "por", "ron": "ron", "rus": "rus", "sat": "sat",
        "slk": "slk", "slv": "slv", "sna": "sna", "snd": "snd", "som": "som",
        "spa": "spa", "srp": "srp", "swe": "swe", "swh": "swh", "tam": "tam",
        "tel": "tel", "tgk": "tgk", "tgl": "tgl", "ukr": "ukr", "urd": "urd",
        "uzn": "uzn", "vie": "vie", "yor": "yor", "yue": "yue", "zlm": "zlm",
        "zul": "zul"
    }

    standard_lang_mapping = {
        "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it",
        "pt": "pt", "ru": "ru", "zh": "zh", "ja": "ja", "ko": "ko",
        "ar": "ar", "hi": "hi", "tr": "tr", "pl": "pl", "nl": "nl",
        "he": "he", "sv": "sv", "da": "da", "no": "no", "fi": "fi",
        "fa": "fa", "ur": "ur", "id": "id", "vi": "vi", "th": "th",
        "uk": "uk", "cs": "cs", "ro": "ro", "hu": "hu", "el": "el",
        "bg": "bg", "sr": "sr", "hr": "hr", "sk": "sk", "sl": "sl",
        "et": "et", "lv": "lv", "lt": "lt"
    }

    lang_aliases = {
        "jp": "ja", "jpn": "ja", "japanese": "ja",
        "fas": "fa", "per": "fa", "pes": "fa", "farsi": "fa", "persian": "fa",
        "iw": "he", "heb": "he", "hebrew": "he",
        "zh-cn": "zh", "zh-tw": "cmn-hant", "cmn": "zh", "chi": "zh", "zho": "zh", "chinese": "zh",
        "eng": "en", "english": "en",
        "spa": "es", "spanish": "es",
        "fra": "fr", "fre": "fr", "french": "fr",
        "deu": "de", "ger": "de", "german": "de",
        "arb": "ar", "ara": "ar", "arabic": "ar",
        "ast": "es", "lim": "nl", "vec": "it",
    }

    def _clean_language_code(lang: Optional[str]) -> Optional[str]:
        if not lang:
            return None

        cleaned = str(lang).strip().lower()
        if not cleaned:
            return None

        cleaned = cleaned.replace("__label__", "")
        cleaned = cleaned.split(".", 1)[0]
        cleaned = cleaned.replace("_", "-")

        if cleaned in lang_aliases:
            return lang_aliases[cleaned]

        base = cleaned.split("-", 1)[0]
        return lang_aliases.get(base, base)

    def _map_language(lang: Optional[str], mapping: Dict[str, str], default: str) -> str:
        normalized = _clean_language_code(lang)
        if not normalized:
            return default
        return mapping.get(normalized, default)

    def _seamless_generation_kwargs(text_to_translate: str) -> Dict:
        word_tokens = len(text_to_translate.split())
        char_tokens = max(1, len(text_to_translate) // 2)
        approx_tokens = max(word_tokens, char_tokens)
        return {
            "max_new_tokens": max(96, min(384, int(approx_tokens * 2.0))),
            "num_beams": 4,
            "do_sample": False,
            "early_stopping": True,
            "length_penalty": 1.0,
        }

    def _attempt_translation(text_to_translate, attempt_name=""):
        """Helper function to attempt translation with proper error handling"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if is_seamless:
                # Seamless M4T requires both src_lang and tgt_lang
                mapped_source = _map_language(source_lang, seamless_lang_mapping, "eng")
                mapped_target = _map_language(target_lang, seamless_lang_mapping, "eng")

                # For Seamless M4T pipeline, use generation parameters
                input_tokens = len(text_to_translate.split())

                logger.info(
                    "Seamless translation: %s -> %s, input_words=%d, chars=%d",
                    mapped_source,
                    mapped_target,
                    input_tokens,
                    len(text_to_translate),
                )

                # Seamless M4T v2 proper usage with processor and model.generate()
                try:
                    # Prefer direct model usage when available for better control
                    # New HF Seamless pipelines expose .model and either .tokenizer or .processor
                    model = getattr(translator, 'model', None)
                    processor = getattr(translator, 'tokenizer', None) or getattr(translator, 'processor', None)

                    if model is not None and processor is not None:
                        # Prepare inputs
                        text_inputs = processor(
                            text=text_to_translate,
                            src_lang=mapped_source,
                            return_tensors="pt"
                        )
                        # Move all tensors to model device (avoids CPU/GPU mismatch)
                        model_device = next(model.parameters()).device
                        logger.debug(f"Seamless translation: moving inputs to model device {model_device}")
                        for k, v in text_inputs.items():
                            if isinstance(v, torch.Tensor):
                                text_inputs[k] = v.to(model_device)

                        generation_kwargs = _seamless_generation_kwargs(text_to_translate)
                        logger.debug("Seamless generation parameters: %s", generation_kwargs)
                        generated_tokens = model.generate(
                            **text_inputs,
                            tgt_lang=mapped_target,
                            **generation_kwargs,
                        )
                        try:
                            gen_device = generated_tokens.device
                        except Exception:
                            gen_device = 'unknown'
                        logger.debug(f"Generated tokens device: {gen_device}")

                        # Decode full sequence (previous code only decoded first token!)
                        if hasattr(processor, 'batch_decode'):
                            decoded = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                        else:  # fallback
                            decoded = processor.decode(generated_tokens[0], skip_special_tokens=True)

                        result = [{"translation_text": decoded}]
                    else:
                        # Fallback: use pipeline call directly
                        result = translator(
                            text_to_translate,
                            src_lang=mapped_source,
                            tgt_lang=mapped_target,
                            **_seamless_generation_kwargs(text_to_translate),
                        )
                except Exception as e1:
                    logger.warning(f"Direct model approach failed: {e1}")
                    try:
                        # Second attempt: pipeline interface (might differ after HF updates)
                        result = translator(text_to_translate, src_lang=mapped_source, tgt_lang=mapped_target)
                    except Exception as e2:
                        logger.warning(f"Pipeline with languages failed: {e2}")
                        raise e2
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
                    mapped_source = _map_language(source_lang, standard_lang_mapping, source_lang)
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
            fallback_source = _map_language(source_lang, seamless_lang_mapping, "eng")
            result = translator(
                minimal_text,
                src_lang=fallback_source,
                tgt_lang="eng"
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



