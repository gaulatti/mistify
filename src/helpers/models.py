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
        # Configure Seamless M4T v2 with better parameters
        translator = pipeline(
            "translation",
            model="facebook/seamless-m4t-v2-large",
            device=device_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            model_kwargs={
                "low_cpu_mem_usage": True,
                "use_safetensors": True,
            } if device == "cuda" else {},
            cache_dir=str(config["HF_CACHE"])
        )
        translator_model_name = "seamless-m4t-v2"
        # Force move model to target device (sometimes pipeline keeps parts on CPU)
        try:
            if hasattr(translator, 'model'):
                model_device = torch.device(device if device != 'cuda' else f'cuda:{device_id}')
                translator.model.to(model_device)
                # Log a sample parameter device
                sample_param_device = next(translator.model.parameters()).device
                logger.info(f"✓ Seamless M4T model placed on device: {sample_param_device}")
        except Exception as move_e:
            logger.warning(f"⚠️ Could not explicitly move Seamless model to device: {move_e}")
        logger.info("✓ Seamless M4T v2 translation model loaded successfully")
    except Exception as e:
        logger.error("❌ Failed to load Seamless M4T v2 model: %s", e)
        logger.warning("Translation functionality will use fallback model")
        try:
            logger.info("🔧 Trying fallback translation model...")
            translator = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-mul-en",
                device=device_id,
                cache_dir=str(config["HF_CACHE"])
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

    # Text Generator (Flan-T5 with graceful fallback)
    text_generator = None
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        requested_model_name = os.getenv("TEXT_GENERATION_MODEL", "google/flan-t5-xl")
        attempted_models = [requested_model_name]

        # If user explicitly sets a smaller model we only try that
        fallback_chain = [] if requested_model_name != "google/flan-t5-xl" else [
            "google/flan-t5-large",
            "google/flan-t5-base"
        ]

        for model_name in [requested_model_name] + fallback_chain:
            try:
                logger.info(f"🔧 Loading text generation model: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(config["HF_CACHE"]),
                    use_fast=True
                )
                # Use accelerate device_map to spread layers if needed
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None,
                    low_cpu_mem_usage=True,
                    cache_dir=str(config["HF_CACHE"])
                )
                # Build pipeline around pre-loaded model (avoids double init)
                text_generator = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=device_id if device == "cuda" else -1
                )
                logger.info(f"✓ Text generation model loaded: {model_name}")
                break
            except RuntimeError as re:
                if "out of memory" in str(re).lower():
                    logger.warning(f"⚠️ CUDA OOM loading {model_name}. Trying next fallback (if any)...")
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    continue  # try next model in chain
                else:
                    logger.error(f"❌ RuntimeError loading {model_name}: {re}")
                    continue
            except Exception as ge:
                logger.error(f"❌ General error loading {model_name}: {ge}")
                continue

        if text_generator is None:
            logger.error("❌ All text generation model load attempts failed: %s", attempted_models)
        else:
            # Attach chosen model name for introspection
            try:
                text_generator.chosen_model_name = text_generator.model.config.name_or_path  # type: ignore
            except Exception:
                pass
    except Exception as e:
        logger.error("❌ Failed during text generation model initialization logic: %s", e)

    # Ensure translator_model_name is always set
    if translator_model_name is None:
        translator_model_name = "none"

    return fasttext_model, classifier, translator, embedder, nlp, translator_model_name, text_generator
