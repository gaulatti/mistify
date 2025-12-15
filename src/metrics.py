"""Prometheus metrics for Mistify.

Design goals:
- Low-cardinality labels (route templates, not raw paths)
- Useful out-of-the-box HTTP request metrics
- Optional operation-level metrics for model-heavy endpoints

All metrics are registered in the default Prometheus registry used by
`prometheus_client.generate_latest()`.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Optional

import psutil
import torch
from prometheus_client import Counter, Gauge, Histogram, Info

# ---- Build / service identity -------------------------------------------------

BUILD_INFO = Info(
    "mistify_build",
    "Mistify build and runtime info",
)

# Populate basic info eagerly (safe to call multiple times; it overwrites labels).
BUILD_INFO.info(
    {
        "service": "mistify",
    }
)

# ---- HTTP (generic) -----------------------------------------------------------

HTTP_REQUESTS_TOTAL = Counter(
    "mistify_http_requests_total",
    "Total HTTP requests received",
    labelnames=("method", "route", "status_code"),
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "mistify_http_request_duration_seconds",
    "HTTP request latency in seconds",
    labelnames=("method", "route", "status_code"),
    # FastAPI endpoints can be heavy; include a few long buckets.
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        20.0,
        40.0,
        80.0,
    ),
)

HTTP_INPROGRESS = Gauge(
    "mistify_http_inprogress_requests",
    "Number of HTTP requests currently being processed",
)

HTTP_EXCEPTIONS_TOTAL = Counter(
    "mistify_http_exceptions_total",
    "Unhandled exceptions raised while processing HTTP requests",
    labelnames=("method", "route", "exception_type"),
)

# ---- Model / operation-level metrics -----------------------------------------

MODEL_AVAILABLE = Gauge(
    "mistify_model_available",
    "Whether a model/component is available (1) or not (0)",
    labelnames=("model",),
)

MODEL_OPERATION_TOTAL = Counter(
    "mistify_model_operation_total",
    "Total model/operation executions",
    labelnames=("operation", "outcome"),
)

MODEL_OPERATION_DURATION_SECONDS = Histogram(
    "mistify_model_operation_duration_seconds",
    "Duration of model/operation executions in seconds",
    labelnames=("operation", "outcome"),
    buckets=(
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        20.0,
        40.0,
        80.0,
    ),
)

# ---- System gauges (updated at scrape-time) ----------------------------------

PROCESS_RSS_BYTES = Gauge(
    "mistify_process_resident_memory_bytes",
    "Resident set size (RSS) of the current process",
)

PROCESS_THREADS = Gauge(
    "mistify_process_threads",
    "Number of threads in the current process",
)

TORCH_DEVICE_INFO = Gauge(
    "mistify_torch_device_available",
    "Whether a torch device backend is available (1) or not (0)",
    labelnames=("device",),
)


def update_runtime_metrics(app_state: Optional[object] = None) -> None:
    """Update gauges that should reflect current runtime state.

    Called from the /metrics endpoint so it stays fresh without a background loop.
    """

    p = psutil.Process()
    PROCESS_RSS_BYTES.set(p.memory_info().rss)
    PROCESS_THREADS.set(p.num_threads())

    TORCH_DEVICE_INFO.labels(device="cuda").set(1.0 if torch.cuda.is_available() else 0.0)
    TORCH_DEVICE_INFO.labels(device="mps").set(1.0 if torch.backends.mps.is_available() else 0.0)
    TORCH_DEVICE_INFO.labels(device="cpu").set(1.0)

    if app_state is not None:
        # Keep model names stable to avoid label cardinality explosions.
        MODEL_AVAILABLE.labels(model="fasttext").set(1.0 if getattr(app_state, "fasttext_model", None) is not None else 0.0)
        MODEL_AVAILABLE.labels(model="classifier").set(1.0 if getattr(app_state, "classifier", None) is not None else 0.0)
        MODEL_AVAILABLE.labels(model="translator").set(1.0 if getattr(app_state, "translator", None) is not None else 0.0)
        MODEL_AVAILABLE.labels(model="embedder").set(1.0 if getattr(app_state, "embedder", None) is not None else 0.0)
        MODEL_AVAILABLE.labels(model="nlp").set(1.0 if getattr(app_state, "nlp", None) is not None else 0.0)
        MODEL_AVAILABLE.labels(model="text_generator").set(1.0 if getattr(app_state, "text_generator", None) is not None else 0.0)


def route_label_from_request_scope(scope: dict) -> str:
    """Derive a low-cardinality route label from an ASGI scope.

    Prefers templated route paths (e.g. "/items/{id}") over raw URLs.
    """

    route = scope.get("route")
    if route is not None:
        path = getattr(route, "path", None)
        if isinstance(path, str) and path:
            return path
    # Fallbacks: keep cardinality low.
    raw_path = scope.get("path")
    if raw_path == "/metrics":
        return "/metrics"
    if raw_path == "/health":
        return "/health"
    return "unmatched"


@contextmanager
def record_operation(operation: str, *, outcome: str = "success"):
    """Context manager to record an operation duration + outcome.

    Use for model-heavy segments inside endpoints.

    If an exception escapes, outcome is recorded as "error".
    """

    start = time.perf_counter()
    final_outcome = outcome
    try:
        yield
    except Exception:
        final_outcome = "error"
        raise
    finally:
        duration = time.perf_counter() - start
        MODEL_OPERATION_TOTAL.labels(operation=operation, outcome=final_outcome).inc()
        MODEL_OPERATION_DURATION_SECONDS.labels(operation=operation, outcome=final_outcome).observe(duration)
