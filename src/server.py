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
import httpx
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from types import SimpleNamespace
from pydantic import ValidationError

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.helpers.models import initialize_models
from src.endpoints import (
    language, classification, translation, embedding, clustering, analysis, generation
)
from src.models import UnifiedAnalysisRequest
from src import metrics

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
    "DEFAULT_CLASSIFICATION_LABELS": [],
    "MIN_SCORE": float(os.getenv("MIN_SCORE", "0.30")),
    "MIN_MARGIN": float(os.getenv("MIN_MARGIN", "0.10")),
    "POOL_WORKERS": int(os.getenv("POOL_WORKERS", "4")),
    "TIMEOUT": int(os.getenv("TIMEOUT", "10")),
    "PROCESSING_LOOP_ENABLED": os.getenv("PROCESSING_LOOP_ENABLED", "true").lower() in {"1", "true", "yes"},
    "PROCESSING_TRANSLATE_TO_ENGLISH": os.getenv("PROCESSING_TRANSLATE_TO_ENGLISH", "true").lower() in {"1", "true", "yes"},
    "PROCESSING_FETCH_URL": os.getenv("PROCESSING_FETCH_URL", "https://api.cronkite.fifthbell.com/processing/fetch"),
    "PROCESSING_POST_URL": os.getenv("PROCESSING_POST_URL", "https://n8n.gaulatti.com/webhook/727a2ba4-a274-462c-91cc-8d2abc7bb81e"),
    "PROCESSING_IDLE_SLEEP_SECONDS": float(os.getenv("PROCESSING_IDLE_SLEEP_SECONDS", "5")),
    "PROCESSING_HTTP_TIMEOUT_SECONDS": float(os.getenv("PROCESSING_HTTP_TIMEOUT_SECONDS", "30")),
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
app_state.processing_loop_task = None


def _extract_items_and_queue(payload):
    """Extract (items, queue_size) from supported fetch payload shapes."""
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            items = first.get("items") or []
            queue_size = first.get("queueSize")
            return items if isinstance(items, list) else [], queue_size

    if isinstance(payload, dict):
        items = payload.get("items") or []
        queue_size = payload.get("queueSize")
        return items if isinstance(items, list) else [], queue_size

    return [], None


def _merge_processed_items(raw_items, analysis_resp):
    """Merge analysis output back into original item payloads."""
    processed = []
    results = analysis_resp.results if analysis_resp else []

    for idx, raw in enumerate(raw_items):
        base = dict(raw) if isinstance(raw, dict) else {"raw": raw}
        if idx >= len(results):
            processed.append(base)
            continue

        res = results[idx]

        base["content"] = res.content

        if res.language_detection:
            lang_dump = res.language_detection.model_dump()
            base["language_detection"] = lang_dump
            langs = lang_dump.get("languages") or []
            if langs and not base.get("lang"):
                base["lang"] = langs[0]

        if res.content_classification:
            cls_dump = res.content_classification.model_dump()
            base["content_classification"] = cls_dump
            labels = (cls_dump.get("full_result") or {}).get("labels") or []
            if labels:
                base["classification_labels"] = labels

        if res.translation:
            base["translation"] = res.translation.model_dump()

        if res.newsworthiness is not None:
            base["newsworthiness"] = res.newsworthiness

        if res.urgency is not None:
            base["urgency"] = res.urgency

        if res.timings:
            base["analysis_timings"] = res.timings.model_dump()

        processed.append(base)

    return processed


async def _processing_loop():
    """Continuously POST-fetch items, process, POST-forward results, and repeat."""
    fetch_url = app_state.config["PROCESSING_FETCH_URL"]
    post_url = app_state.config["PROCESSING_POST_URL"]
    idle_sleep = app_state.config["PROCESSING_IDLE_SLEEP_SECONDS"]
    timeout = app_state.config["PROCESSING_HTTP_TIMEOUT_SECONDS"]

    logger.info("🔁 Processing loop started (POST fetch -> analyze -> POST forward)")
    logger.info("   fetch_url=%s", fetch_url)
    logger.info("   post_url=%s", post_url)

    fake_request = SimpleNamespace(state=SimpleNamespace(app_state=app_state))

    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            try:
                fetch_resp = await client.post(fetch_url, json={})
                fetch_resp.raise_for_status()

                try:
                    fetched_payload = fetch_resp.json()
                except json.JSONDecodeError:
                    logger.warning("⚠️ Fetch response is not valid JSON; sleeping %.1fs", idle_sleep)
                    await asyncio.sleep(idle_sleep)
                    continue

                items, queue_size = _extract_items_and_queue(fetched_payload)

                if not items:
                    logger.info("📭 Queue empty (queueSize=%s). Sleeping %.1fs", queue_size, idle_sleep)
                    await asyncio.sleep(idle_sleep)
                    continue

                logger.info("📥 Fetched %d items (queueSize=%s)", len(items), queue_size)

                try:
                    req = UnifiedAnalysisRequest(
                        items=items,
                        translate_to_english=app_state.config["PROCESSING_TRANSLATE_TO_ENGLISH"]
                    )
                except ValidationError as ve:
                    logger.warning("⚠️ Batch validation failed; attempting per-item filtering: %s", ve)
                    valid_items = []
                    for it in items:
                        try:
                            UnifiedAnalysisRequest(items=[it])
                            valid_items.append(it)
                        except ValidationError:
                            continue

                    if not valid_items:
                        logger.warning("⚠️ No valid items in fetched batch; sleeping %.1fs", idle_sleep)
                        await asyncio.sleep(idle_sleep)
                        continue

                    logger.info("🧹 Filtered batch to %d valid items (from %d)", len(valid_items), len(items))
                    req = UnifiedAnalysisRequest(
                        items=valid_items,
                        translate_to_english=app_state.config["PROCESSING_TRANSLATE_TO_ENGLISH"]
                    )
                    items = valid_items

                analysis_resp = await analysis.unified_analysis(req, fake_request)
                processed_items = _merge_processed_items(items, analysis_resp)

                forward_resp = await client.post(post_url, json=processed_items)
                forward_resp.raise_for_status()

                logger.info("📤 Forwarded %d processed items", len(processed_items))

                # Continue immediately while queue has data.
                await asyncio.sleep(0)

            except asyncio.CancelledError:
                logger.info("🛑 Processing loop cancelled")
                raise
            except Exception as e:
                logger.error("❌ Processing loop error: %s", e)
                await asyncio.sleep(idle_sleep)


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
        "capabilities": ["language_detection", "content_classification", "translation", "text_clustering", "text_generation"],
        "background_processing_loop": {
            "enabled": app_state.config["PROCESSING_LOOP_ENABLED"],
            "fetch_url": app_state.config["PROCESSING_FETCH_URL"],
            "post_url": app_state.config["PROCESSING_POST_URL"],
        },
        "endpoints": {
            "language_detection": "/detect",
            "content_classification": "/classify",
            "translation": "/translate",
            "embeddings": "/embed",
            "text_clustering": "/cluster",
            "text_generation": "/generate/text",
            "health": "/health"
        }
    }


@app.on_event("startup")
async def start_processing_loop():
    if not app_state.config.get("PROCESSING_LOOP_ENABLED"):
        logger.info("⏸️ Processing loop disabled by config")
        return

    if app_state.processing_loop_task is None or app_state.processing_loop_task.done():
        app_state.processing_loop_task = asyncio.create_task(_processing_loop())


@app.on_event("shutdown")
async def stop_processing_loop():
    task = app_state.processing_loop_task
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
