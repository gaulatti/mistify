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
import json
import psutil
import torch
import warnings
import pathlib
import time
from redis.asyncio import Redis
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from types import SimpleNamespace
from dotenv import load_dotenv

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.helpers.models import initialize_models
from src.endpoints import (
    language, classification, translation, embedding, clustering, analysis
)
from src.grpc.server import start_grpc_server
from src.operations.queue import OperationQueue
from src.operations.worker import OperationWorker
from src import metrics

load_dotenv()

# Suppress transformers deprecation warnings
warnings.filterwarnings("ignore", message="transformers.deepspeed module is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.deepspeed")
warnings.filterwarnings("ignore", message="You must either specify a `tgt_lang`")
warnings.filterwarnings("ignore", message="Both `max_new_tokens`.*and `max_length`.*seem to have been set")
warnings.filterwarnings("ignore", message="Your input_length.*is bigger than.*max_length")
warnings.filterwarnings("ignore", message="Setting `pad_token_id`.*to `eos_token_id`")
warnings.filterwarnings("ignore", message="`resume_download` is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# ---- Logging Setup -------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mistify")

# ---- PyTorch Configuration -----------------------------------------------------
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await start_processing_loop()
    try:
        yield
    finally:
        await stop_processing_loop()


# ---- FastAPI App ---------------------------------------------------------------
app = FastAPI(
    title="Mistify",
    description="Language detection, content classification, translation, sentence embeddings, and entity-aware clustering",
    version="1.3.0",
    lifespan=lifespan,
)

# ---- App State -----------------------------------------------------------------
app_state = SimpleNamespace()

# ---- Configuration -------------------------------------------------------------
app_state.config = {
    "FASTTEXT_MODEL_PATH": os.getenv("FASTTEXT_MODEL_PATH", "lid.176.bin"),
    "FASTTEXT_MODEL_URL": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
    "DEFAULT_CLASSIFICATION_LABELS": [],
    "MIN_SCORE": float(os.getenv("MIN_SCORE", "0.30")),
    "MIN_MARGIN": float(os.getenv("MIN_MARGIN", "0.10")),
    "POOL_WORKERS": int(os.getenv("POOL_WORKERS", "4")),
    "TIMEOUT": int(os.getenv("TIMEOUT", "10")),
    "PROCESSING_TRANSLATE_TO_ENGLISH": os.getenv("PROCESSING_TRANSLATE_TO_ENGLISH", "true").lower() in {"1", "true", "yes"},
    "MONITOR_GRPC_CALLBACK_TARGET": os.getenv("MONITOR_GRPC_CALLBACK_TARGET", "localhost:50055"),
    "HTTP_PORT": int(os.getenv("HTTP_PORT", "8000")),
    "VALKEY_HOST": os.getenv("VALKEY_HOST", "host.docker.internal"),
    "VALKEY_PORT": int(os.getenv("VALKEY_PORT", "6379")),
    "HF_CACHE": pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".hf_models")),
    # Clustering thresholds (tune for recall vs precision)
    "CLUSTERING_SIM_ENTITY": float(os.getenv("CLUSTERING_SIM_ENTITY", "0.35")),
    "CLUSTERING_SIM_GLOBAL": float(os.getenv("CLUSTERING_SIM_GLOBAL", "0.50")),
    "CLUSTERING_AVG_SIM_MIN": float(os.getenv("CLUSTERING_AVG_SIM_MIN", "0.40")),
    "CLUSTERING_PRE_FILTER_MIN_SIM": float(os.getenv("CLUSTERING_PRE_FILTER_MIN_SIM", "0.25")),
    "CLUSTERING_MAX_CANDIDATES": int(os.getenv("CLUSTERING_MAX_CANDIDATES", "30")),
}
app_state.config["HF_CACHE"].mkdir(parents=True, exist_ok=True)

# ---- Global Variables ----------------------------------------------------------
load_models_on_startup = os.getenv("LOAD_MODELS_ON_STARTUP", "true").strip().lower() in {"1", "true", "yes"}
if load_models_on_startup:
    (
        app_state.fasttext_model,
        app_state.classifier,
        app_state.translator,
        app_state.embedder,
        app_state.nlp,
        app_state.translator_model_name,
    ) = initialize_models(app_state.config)
else:
    logger.warning("⏭️ Skipping eager model initialization (LOAD_MODELS_ON_STARTUP=false)")
    app_state.fasttext_model = None
    app_state.classifier = None
    app_state.translator = None
    app_state.embedder = None
    app_state.nlp = None
    app_state.translator_model_name = "none"
app_state.classification_lock = asyncio.Lock()
app_state.translation_lock = asyncio.Lock()
app_state.clustering_lock = asyncio.Lock()
app_state.embedding_lock = asyncio.Lock()
app_state.editorial_lock = asyncio.Lock()
app_state.thread_pool = ThreadPoolExecutor(max_workers=app_state.config["POOL_WORKERS"])


def _create_redis_client(config):
    return Redis(
        host=config["VALKEY_HOST"],
        port=config["VALKEY_PORT"],
        decode_responses=True,
    )


app_state.redis_client = _create_redis_client(app_state.config)
app_state.operation_queue = OperationQueue(app_state.redis_client)
app_state.operation_worker = OperationWorker(app_state.operation_queue, app_state)
app_state.operation_worker_task = None
app_state.grpc_server = None


# ---- Middleware to pass state --------------------------------------------------
@app.middleware("http")
async def add_state_to_request(request: Request, call_next):
    request.state.app_state = app_state
    response = await call_next(request)
    return response


# ---- Prometheus Metrics Middleware --------------------------------------------
@app.middleware("http")
async def prometheus_http_metrics(request: Request, call_next):
    method = request.method
    route_label = metrics.route_label_from_request_scope(request.scope)

    start = time.perf_counter()
    metrics.HTTP_INPROGRESS.inc()
    try:
        response = await call_next(request)
        status_code = str(response.status_code)
        return response
    except HTTPException as e:
        # Expected HTTP errors (4xx/5xx raised intentionally by handlers).
        status_code = str(e.status_code)
        raise
    except Exception as e:
        # Unexpected exception. FastAPI will turn this into a 500.
        metrics.HTTP_EXCEPTIONS_TOTAL.labels(method=method, route=route_label, exception_type=type(e).__name__).inc()
        status_code = "500"
        raise
    finally:
        duration = time.perf_counter() - start
        metrics.HTTP_INPROGRESS.dec()
        metrics.HTTP_REQUESTS_TOTAL.labels(method=method, route=route_label, status_code=status_code).inc()
        metrics.HTTP_REQUEST_DURATION_SECONDS.labels(method=method, route=route_label, status_code=status_code).observe(duration)


# ---- API Routers ---------------------------------------------------------------
app.include_router(language.router)
app.include_router(classification.router)
app.include_router(translation.router)
app.include_router(embedding.router)
app.include_router(clustering.router)


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
        },
        "system": {
            "threads": process.num_threads(),
            "memory_mb": round(process.memory_info().rss / 2 ** 20, 1),
            "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        }
    }


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus scrape endpoint."""
    # Keep gauges fresh on each scrape.
    metrics.update_runtime_metrics(app_state)
    payload = generate_latest()  # default registry
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "service": "Unified Text Analysis API",
        "version": "1.3.0",
        "capabilities": ["language_detection", "content_classification", "translation", "text_clustering"],
        "operation_queue": {
            "queue_name": app_state.operation_queue.queue_name,
        },
        "endpoints": {
            "language_detection": "/detect",
            "content_classification": "/classify",
            "translation": "/translate",
            "embeddings": "/embed",
            "text_clustering": "/cluster",
            "health": "/health"
        }
    }


async def start_processing_loop():
    if app_state.grpc_server is None:
        app_state.grpc_server = await start_grpc_server(app_state.operation_queue)

    if app_state.operation_worker_task is None or app_state.operation_worker_task.done():
        app_state.operation_worker_task = asyncio.create_task(app_state.operation_worker.run_forever())


async def stop_processing_loop():
    if app_state.grpc_server is not None:
        await app_state.grpc_server.stop(grace=5)

    operation_task = app_state.operation_worker_task
    if operation_task and not operation_task.done():
        operation_task.cancel()
        try:
            await operation_task
        except asyncio.CancelledError:
            pass
    await app_state.redis_client.aclose()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=app_state.config["HTTP_PORT"])
