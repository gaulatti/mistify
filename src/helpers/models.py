# service/src/helpers/models.py

import logging
import os
import urllib.request
import torch
import fasttext
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def download_fasttext_model(model_path: str, model_url: str) -> bool:
    """Download FastText language detection model if not present"""
    try:
        import urllib.request
        logger.info("🌍 Downloading FastText language detection model...")
        urllib.request.urlretrieve(model_url, model_path)
        logger.info("✓ FastText model downloaded successfully")
        return True
    except Exception as e:
        logger.error("❌ Failed to download FastText model: %s", e)
        return False


def initialize_models(config):
    """Initialize all models"""
    logger.info("🔧 Initializing models...")

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        device_id = 0
        logger.info("🔧 Using device: CUDA GPU")
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        device = "mps"
        device_id = 0
        logger.info("🔧 Using device: MPS (Apple Silicon GPU)")
    else:
        device = "cpu"
        device_id = -1
        logger.info("🔧 Using device: CPU")

    # FastText
    fasttext_model = None
    try:
        if not os.path.exists(config["FASTTEXT_MODEL_PATH"]):
            logger.info("FastText model not found, downloading...")
            if not download_fasttext_model(config["FASTTEXT_MODEL_PATH"], config["FASTTEXT_MODEL_URL"]):
                raise Exception("Failed to download FastText model")
        logger.info("🔧 Loading FastText model from: %s", config["FASTTEXT_MODEL_PATH"])
        fasttext_model = fasttext.load_model(config["FASTTEXT_MODEL_PATH"])
        fasttext_model.predict("Hello world", k=1)
        logger.info("✓ FastText model loaded and tested successfully")
    except Exception as e:
        logger.error("❌ Failed to load FastText model: %s", e)

    # Classifier
    classifier = None
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-3",
            device=device_id,
            hypothesis_template="This post is {}.",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            model_kwargs={"low_cpu_mem_usage": True} if device == "cuda" else {},
            cache_dir=str(config["HF_CACHE"])
        )
        logger.info("✓ BART classification model loaded successfully")
    except Exception as e:
        logger.error("❌ Failed to load BART classification model: %s", e)

    # Translator
    translator = None
    translator_model_name = None
    try:
        logger.info("🔧 Loading Seamless M4T v2 translation model...")
        # Remove cache_dir to avoid model_kwargs issues with Seamless M4T
        translator = pipeline(
            "translation",
            model="facebook/seamless-m4t-v2-large",
            device=device_id,
            trust_remote_code=True
        )
        translator_model_name = "seamless-m4t-v2"
        logger.info("✓ Seamless M4T v2 translation model loaded successfully")
    except Exception as e:
        logger.error("❌ Failed to load Seamless M4T v2 model: %s", e)
        logger.warning("Translation functionality will use fallback model")
        try:
            logger.info("🔧 Trying fallback translation model...")
            translator = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-mul-en",
                device=device_id
            )
            translator_model_name = "helsinki-nlp"
            logger.info("✓ Fallback translation model (Helsinki-NLP) loaded successfully")
        except Exception as fallback_e:
            logger.error("❌ Fallback translation model also failed: %s", fallback_e)
            translator_model_name = "none"

    # Embedder
    embedder = None
    try:
        torch_device = torch.device(device)
        embedder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=torch_device,
            cache_folder=str(config["HF_CACHE"])
        )
        logger.info("✓ Sentence transformer embedder loaded successfully")
    except Exception as e:
        logger.error("❌ Failed to load sentence transformer: %s", e)

    # SpaCy
    nlp = None
    try:
        nlp = spacy.load("en_core_web_sm", disable=("tagger", "parser", "lemmatizer"))
        logger.info("✓ SpaCy model loaded for entity extraction")
    except Exception as e:
        logger.error("❌ Failed to load SpaCy model: %s", e)
        logger.info("Please install SpaCy English model: python -m spacy download en_core_web_sm")

    # Ensure translator_model_name is always set
    if translator_model_name is None:
        translator_model_name = "none"

    return fasttext_model, classifier, translator, embedder, nlp, translator_model_name
