# server.py — Mistify API
# =====================================
# Language detection, content classification, translation, text clustering,
# and batch sentence embeddings generation in one service.

import os

# ---- HARD CAP hidden thread pools for PyTorch compatibility -------------------
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "RAYON_NUM_THREADS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
    "CUDA_LAUNCH_BLOCKING": "0",
})

# ---- Imports -------------------------------------------------------------------
import asyncio
import logging
import psutil
import torch
import warnings
import pathlib
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request
from types import SimpleNamespace

from src.helpers.models import initialize_models
from src.endpoints import (
    language, classification, translation, embedding, clustering, analysis, generation
)

# Suppress transformers deprecation warnings
warnings.filterwarnings("ignore", message="transformers.deepspeed module is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.deepspeed")
warnings.filterwarnings("ignore", message="You must either specify a `tgt_lang`")
warnings.filterwarnings("ignore", message="Both `max_new_tokens`.*and `max_length`.*seem to have been set")
warnings.filterwarnings("ignore", message="Your input_length.*is bigger than.*max_length")
warnings.filterwarnings("ignore", message="Setting `pad_token_id`.*to `eos_token_id`")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# ---- Logging Setup -------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mistify")

# ---- PyTorch Configuration -----------------------------------------------------
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ---- FastAPI App ---------------------------------------------------------------
app = FastAPI(
    title="Mistify",
    description="Language detection, content classification, translation, sentence embeddings, and entity-aware clustering",
    version="1.3.0"
)

# ---- App State -----------------------------------------------------------------
app_state = SimpleNamespace()

# ---- Configuration -------------------------------------------------------------
app_state.config = {
    "FASTTEXT_MODEL_PATH": os.getenv("FASTTEXT_MODEL_PATH", "lid.176.bin"),
    "FASTTEXT_MODEL_URL": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
    "DEFAULT_CLASSIFICATION_LABELS": [
        "breaking news", "newsworthy factual", "humor / satire", "non-newsworthy rant"
    ],
    "MIN_SCORE": float(os.getenv("MIN_SCORE", "0.30")),
    "MIN_MARGIN": float(os.getenv("MIN_MARGIN", "0.10")),
    "POOL_WORKERS": int(os.getenv("POOL_WORKERS", "4")),
    "TIMEOUT": int(os.getenv("TIMEOUT", "10")),
    "HF_CACHE": pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".hf_models")),
}
app_state.config["HF_CACHE"].mkdir(parents=True, exist_ok=True)

# ---- Global Variables ----------------------------------------------------------
(
    app_state.fasttext_model,
    app_state.classifier,
    app_state.translator,
    app_state.embedder,
    app_state.nlp,
    app_state.translator_model_name,
    app_state.text_generator,
) = initialize_models(app_state.config)
app_state.classification_lock = asyncio.Lock()
app_state.translation_lock = asyncio.Lock()
app_state.clustering_lock = asyncio.Lock()
app_state.thread_pool = ThreadPoolExecutor(max_workers=app_state.config["POOL_WORKERS"])


# ---- Middleware to pass state --------------------------------------------------
@app.middleware("http")
async def add_state_to_request(request: Request, call_next):
    request.state.app_state = app_state
    response = await call_next(request)
    return response


# ---- API Routers ---------------------------------------------------------------
app.include_router(language.router)
app.include_router(classification.router)
app.include_router(translation.router)
app.include_router(embedding.router)
app.include_router(clustering.router)
app.include_router(analysis.router)
app.include_router(generation.router)


# ---- System Endpoints ----------------------------------------------------------
@app.get("/health")
def health():
    """Health check endpoint with system information"""
    process = psutil.Process()
    return {
        "status": "healthy",
        "models": {
            "fasttext_loaded": app_state.fasttext_model is not None,
            "classifier_loaded": app_state.classifier is not None,
            "translator_loaded": app_state.translator is not None,
            "embedder_loaded": app_state.embedder is not None,
            "nlp_loaded": app_state.nlp is not None,
            "text_generator_loaded": app_state.text_generator is not None,
        },
        "system": {
            "threads": process.num_threads(),
            "memory_mb": round(process.memory_info().rss / 2 ** 20, 1),
            "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        }
    }


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "service": "Unified Text Analysis API",
        "version": "1.3.0",
        "capabilities": ["language_detection", "content_classification", "translation", "text_clustering", "text_generation"],
        "endpoints": {
            "language_detection": "/detect",
            "content_classification": "/classify",
            "translation": "/translate",
            "embeddings": "/embed",
            "text_clustering": "/cluster",
            "unified_analysis": "/analyze",
            "text_generation": "/generate/text",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
