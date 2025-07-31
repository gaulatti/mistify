#!/usr/bin/env python3
"""
Pre-download models for the unified text analysis API
This script downloads all required models during Docker build to avoid runtime downloads
"""

import os
import torch
import pathlib
from transformers import pipeline
import logging

# Set environment variables for thread safety
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "RAYON_NUM_THREADS": "1"
})

# Configure PyTorch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model-downloader")

# HuggingFace cache configuration
# Use HF_HOME environment variable if set, otherwise default to ~/.hf_models
HF_CACHE = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".hf_models"))
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE)

def download_translation_models():
    """Download translation models - ensuring Helsinki-NLP/opus-mt-mul-en is available"""
    logger.info("📥 Pre-downloading required translation model...")
    
    # Download the required Helsinki-NLP model first
    try:
        helsinki_translator = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-mul-en",
            device=-1,
            cache_dir=str(HF_CACHE)
        )
        logger.info("✅ Helsinki-NLP/opus-mt-mul-en model downloaded successfully")
        translation_success = True
    except Exception as e:
        logger.error(f"❌ Required Helsinki-NLP model download failed: {e}")
        translation_success = False
    
    # Try to also download Seamless M4T v2 as backup (optional)
    try:
        logger.info("📥 Pre-downloading additional Seamless M4T v2 model...")
        seamless_translator = pipeline(
            "translation",
            model="facebook/seamless-m4t-v2-large",
            device=-1,  # Use CPU for download
            torch_dtype=torch.float32,
            trust_remote_code=True,
            cache_dir=str(HF_CACHE)
        )
        logger.info("✅ Seamless M4T v2 model downloaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Optional Seamless M4T v2 model failed (not critical): {e}")
    
    return translation_success

def download_classification_model():
    """Download classification model - ensuring valhalla/distilbart-mnli-12-3 is available"""
    logger.info("📥 Pre-downloading required classification model...")
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-3",
            device=-1,
            hypothesis_template="This post is {}.",
            cache_dir=str(HF_CACHE)
        )
        logger.info("✅ valhalla/distilbart-mnli-12-3 classification model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Required classification model download failed: {e}")
        return False

def download_clustering_models():
    """Download clustering models (Sentence Transformers and SpaCy)"""
    logger.info("📥 Pre-downloading clustering models...")
    
    # Download Sentence Transformer model
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("📥 Downloading Sentence Transformer embedder...")
        embedder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=torch.device("cpu"),  # Use CPU for download
            cache_folder=str(HF_CACHE)
        )
        # Test the model with a simple encoding
        test_embedding = embedder.encode(["test sentence"], convert_to_tensor=False)
        logger.info("✅ Sentence Transformer model downloaded and tested successfully")
        st_success = True
    except Exception as e:
        logger.error(f"❌ Sentence Transformer download failed: {e}")
        st_success = False
    
    # Download and verify SpaCy model
    try:
        import spacy
        logger.info("📥 Loading SpaCy English model...")
        nlp = spacy.load("en_core_web_sm", disable=("tagger", "parser", "lemmatizer"))
        # Test the model with a simple text
        test_doc = nlp("Apple Inc. is located in California.")
        test_entities = [ent.text for ent in test_doc.ents]
        logger.info(f"✅ SpaCy model loaded and tested successfully. Found entities: {test_entities}")
        spacy_success = True
    except Exception as e:
        logger.error(f"❌ SpaCy model loading failed: {e}")
        logger.error("Make sure 'python -m spacy download en_core_web_sm' was run during build")
        spacy_success = False
    
    return st_success and spacy_success

if __name__ == "__main__":
    logger.info("🚀 Starting model download process...")
    
    # Download models
    translation_success = download_translation_models()
    classification_success = download_classification_model()
    clustering_success = download_clustering_models()
    
    # Report results
    total_models = 3
    successful_models = sum([translation_success, classification_success, clustering_success])
    
    logger.info(f"📊 Download Summary: {successful_models}/{total_models} model groups downloaded")
    
    if successful_models == total_models:
        logger.info("🎉 All models downloaded successfully!")
    elif successful_models > 0:
        logger.info("⚠️ Some models downloaded successfully")
        if not translation_success:
            logger.warning("  - Translation models failed")
        if not classification_success:
            logger.warning("  - Classification model failed")
        if not clustering_success:
            logger.warning("  - Clustering models failed")
    else:
        logger.error("❌ All model downloads failed")
        exit(1)
    
    logger.info("✅ Model pre-download completed!")
