#!/usr/bin/env python3
"""
Model verification script - checks that all models are available locally
Run this after building the Docker image to verify models are cached
"""

import os
import pathlib
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model-verifier")

# HuggingFace cache location - use same location as download_models.py
HF_CACHE = pathlib.Path.home() / ".hf_models"
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_HUB_OFFLINE"] = "1"  # Force offline mode
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def verify_fasttext_model():
    """Verify FastText model is available"""
    model_path = "lid.176.bin"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"✓ FastText model found: {model_path} ({size_mb:.1f} MB)")
        return True
    else:
        logger.error(f"❌ FastText model not found: {model_path}")
        return False

def verify_spacy_model():
    """Verify SpaCy model is available"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=("tagger", "parser", "lemmatizer"))
        test_doc = nlp("Apple Inc. is located in California.")
        entities = [ent.text for ent in test_doc.ents]
        logger.info(f"✓ SpaCy model loaded successfully. Test entities: {entities}")
        return True
    except Exception as e:
        logger.error(f"❌ SpaCy model verification failed: {e}")
        return False

def verify_sentence_transformer():
    """Verify Sentence Transformer model is cached"""
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            cache_folder=str(HF_CACHE),
            local_files_only=True  # Force offline mode
        )
        test_embedding = embedder.encode(["test sentence"], convert_to_tensor=False)
        logger.info(f"✓ Sentence Transformer model loaded from cache. Embedding shape: {test_embedding.shape}")
        return True
    except Exception as e:
        logger.error(f"❌ Sentence Transformer verification failed: {e}")
        return False

def verify_classification_model():
    """Verify BART classification model is cached"""
    try:
        from transformers import pipeline
        classifier = pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-3",
            device=-1,  # CPU only for verification
            local_files_only=True,  # Force offline mode
            model_kwargs={'cache_dir': str(HF_CACHE)}
        )
        test_result = classifier("This is a test sentence", ["test", "example"])
        logger.info(f"✓ Classification model loaded from cache. Test result: {test_result['labels'][0]}")
        return True
    except Exception as e:
        logger.error(f"❌ Classification model verification failed: {e}")
        return False

def verify_translation_model():
    """Verify translation model is cached"""
    try:
        from transformers import pipeline
        # Try primary required model first: Helsinki-NLP/opus-mt-mul-en
        try:
            translator = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-mul-en",
                device=-1,  # CPU only for verification
                local_files_only=True,  # Force offline mode
                model_kwargs={'cache_dir': str(HF_CACHE)}
            )
            logger.info("✓ Helsinki-NLP/opus-mt-mul-en translation model loaded from cache")
            return True
        except Exception:
            # Try Seamless M4T v2 as fallback
            translator = pipeline(
                "translation",
                model="facebook/seamless-m4t-v2-large",
                device=-1,
                local_files_only=True,
                model_kwargs={'cache_dir': str(HF_CACHE)}
            )
            logger.info("✓ Seamless M4T v2 translation model loaded from cache")
            return True
    except Exception as e:
        logger.error(f"❌ Translation model verification failed: {e}")
        return False

def check_cache_size():
    """Check total cache size"""
    if HF_CACHE.exists():
        total_size = sum(f.stat().st_size for f in HF_CACHE.rglob('*') if f.is_file())
        size_gb = total_size / (1024 * 1024 * 1024)
        logger.info(f"📊 Total HuggingFace cache size: {size_gb:.2f} GB")
        return size_gb
    else:
        logger.warning("⚠️ HuggingFace cache directory not found")
        return 0

def main():
    logger.info("🔍 Verifying all models are cached and available offline...")
    logger.info("=" * 60)
    
    # Run all verifications
    checks = [
        ("FastText Language Detection", verify_fasttext_model),
        ("SpaCy NLP", verify_spacy_model),
        ("Sentence Transformer", verify_sentence_transformer),
        ("Classification Model", verify_classification_model),
        ("Translation Model", verify_translation_model),
    ]
    
    results = []
    for name, check_func in checks:
        logger.info(f"\n🔍 Checking {name}...")
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            logger.error(f"❌ {name} check failed with exception: {e}")
            results.append(False)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 VERIFICATION SUMMARY")
    logger.info("=" * 60)
    
    for i, (name, _) in enumerate(checks):
        status = "✓ PASS" if results[i] else "❌ FAIL"
        logger.info(f"{status} {name}")
    
    cache_size = check_cache_size()
    passed = sum(results)
    total = len(results)
    
    logger.info(f"\n📊 Results: {passed}/{total} models verified")
    logger.info(f"💾 Cache size: {cache_size:.2f} GB")
    
    if passed == total:
        logger.info("🎉 All models are cached and ready for offline operation!")
        return 0
    else:
        logger.error(f"❌ {total - passed} models failed verification")
        logger.error("Docker build may not have completed successfully")
        return 1

if __name__ == "__main__":
    sys.exit(main())
