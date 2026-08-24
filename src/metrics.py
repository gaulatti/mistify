"""Prometheus metrics for Mistify.

Design goals:
- Low-cardinality labels (route templates, not raw paths)
- Useful out-of-the-box HTTP request metrics
- Optional operation-level metrics for model-heavy endpoints

All metrics are registered in the default Prometheus registry used by
`prometheus_client.generate_latest()`.
"""

from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager
from functools import wraps
from typing import Optional

import psutil
import torch
from prometheus_client import Counter, Gauge, Histogram, Info

from src.version import VERSION

OPERATIONS = frozenset(
    {
        "analyze",
        "analyze_posts",
        "classify",
        "classify_content",
        "cluster",
        "cluster_post",
        "detect_language",
        "embed",
        "embed_text",
        "generate_scout_queries",
        "language_detect",
        "rank_scout_candidates",
        "translate",
        "translate_text",
    }
)
RPC_METHODS = frozenset(
    {
        "AnalyzePosts",
        "ClassifyContent",
        "ClusterPost",
        "DetectLanguage",
        "EmbedText",
        "GenerateScoutQueries",
        "RankScoutCandidates",
        "TranslateText",
    }
)

# ---- Build / service identity -------------------------------------------------

BUILD_INFO = Info(
    "mistify_build",
    "Mistify build and runtime info",
)

# Populate basic info eagerly (safe to call multiple times; it overwrites labels).
BUILD_INFO.info(
    {
        "service": "mistify",
        "version": VERSION,
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
    labelnames=("method", "route", "failure_type"),
)

# ---- gRPC --------------------------------------------------------------------

RPC_REQUESTS_TOTAL = Counter(
    "mistify_grpc_requests_total",
    "Total gRPC requests received",
    labelnames=("method", "outcome"),
)

RPC_REQUEST_DURATION_SECONDS = Histogram(
    "mistify_grpc_request_duration_seconds",
    "gRPC request latency in seconds",
    labelnames=("method", "outcome"),
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
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

MODEL_OPERATION_PHASE_DURATION_SECONDS = Histogram(
    "mistify_model_operation_phase_duration_seconds",
    "Duration of individual phases inside a model/operation execution in seconds",
    labelnames=("operation", "phase"),
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

ANALYZE_STEP_DURATION_SECONDS = Histogram(
    "mistify_analyze_step_duration_seconds",
    "Duration of individual /analyze stages in seconds",
    labelnames=("step",),
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

ANALYZE_BATCH_ITEMS = Histogram(
    "mistify_analyze_batch_items",
    "Number of items submitted in each /analyze request",
    buckets=(1, 2, 5, 10, 20, 50, 100, 250, 500, 1000),
)

# ---- Posts/Items processing metrics ------------------------------------------

POSTS_PROCESSED_TOTAL = Counter(
    "mistify_posts_processed_total",
    "Total number of posts/items processed",
    labelnames=("endpoint",),
)

POSTS_BATCH_SIZE = Histogram(
    "mistify_posts_batch_size",
    "Number of posts/items in each request",
    labelnames=("endpoint",),
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
)

# ---- Retry/Failure metrics ---------------------------------------------------

OPERATION_RETRIES_TOTAL = Counter(
    "mistify_operation_retries_total",
    "Total number of operation retries",
    labelnames=("operation",),
)

OPERATION_FAILURES_TOTAL = Counter(
    "mistify_operation_failures_total",
    "Total number of operation failures",
    labelnames=("operation", "failure_type"),
)

OPERATION_JOBS_TOTAL = Counter(
    "mistify_operation_jobs_total",
    "Total asynchronous operation jobs completed",
    labelnames=("operation", "outcome"),
)

OPERATION_JOB_DURATION_SECONDS = Histogram(
    "mistify_operation_job_duration_seconds",
    "Duration of asynchronous operation jobs in seconds",
    labelnames=("operation", "outcome"),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80, 160),
)

OPERATION_QUEUE_EVENTS_TOTAL = Counter(
    "mistify_operation_queue_events_total",
    "Total operation queue transitions",
    labelnames=("operation", "outcome"),
)

OPERATION_QUEUE_DEPTH = Gauge(
    "mistify_operation_queue_depth",
    "Number of operations currently stored in the Redis queue",
)

CALLBACK_REQUESTS_TOTAL = Counter(
    "mistify_callback_requests_total",
    "Total callback delivery attempts",
    labelnames=("channel", "outcome"),
)

CALLBACK_REQUEST_DURATION_SECONDS = Histogram(
    "mistify_callback_request_duration_seconds",
    "Callback delivery attempt duration in seconds",
    labelnames=("channel", "outcome"),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30),
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

GPU_MEMORY_ALLOCATED_BYTES = Gauge(
    "mistify_gpu_memory_allocated_bytes",
    "Total GPU memory currently allocated by PyTorch in bytes",
)

GPU_MEMORY_RESERVED_BYTES = Gauge(
    "mistify_gpu_memory_reserved_bytes",
    "Total GPU memory currently reserved by PyTorch in bytes",
)

GPU_UTILIZATION_PERCENT = Gauge(
    "mistify_gpu_utilization_percent",
    "Maximum GPU utilization percentage (if available via nvidia-smi)",
)


def update_runtime_metrics(app_state: Optional[object] = None) -> None:
    """Update gauges that should reflect current runtime state.

    Called from the /metrics endpoint so it stays fresh without a background loop.
    """

    p = psutil.Process()
    PROCESS_RSS_BYTES.set(p.memory_info().rss)
    PROCESS_THREADS.set(p.num_threads())

    TORCH_DEVICE_INFO.labels(device="cuda").set(
        1.0 if torch.cuda.is_available() else 0.0
    )
    TORCH_DEVICE_INFO.labels(device="mps").set(
        1.0 if torch.backends.mps.is_available() else 0.0
    )
    TORCH_DEVICE_INFO.labels(device="cpu").set(1.0)

    # Update GPU metrics if CUDA is available
    if torch.cuda.is_available():
        try:
            device_count = torch.cuda.device_count()
            allocated_total = 0
            reserved_total = 0
            utilization = []
            for device_id in range(device_count):
                # Get memory stats
                allocated = torch.cuda.memory_allocated(device_id)
                reserved = torch.cuda.memory_reserved(device_id)
                
                allocated_total += allocated
                reserved_total += reserved
                
                # Try to get GPU utilization using nvidia-smi (optional)
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu",
                            "--format=csv,noheader,nounits",
                            "-i",
                            str(device_id),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=1
                    )
                    if result.returncode == 0:
                        util = float(result.stdout.strip())
                        utilization.append(util)
                except Exception:
                    # nvidia-smi not available or failed, skip utilization metric
                    pass
        except Exception:
            # GPU metrics collection failed, continue with other metrics
            pass
        else:
            GPU_MEMORY_ALLOCATED_BYTES.set(allocated_total)
            GPU_MEMORY_RESERVED_BYTES.set(reserved_total)
            if utilization:
                GPU_UTILIZATION_PERCENT.set(max(utilization))

    if app_state is not None:
        # Keep model names stable to avoid label cardinality explosions.
        for metric_name, attribute in (
            ("fasttext", "fasttext_model"),
            ("classifier", "classifier"),
            ("translator", "translator"),
            ("embedder", "embedder"),
            ("nlp", "nlp"),
        ):
            available = getattr(app_state, attribute, None) is not None
            MODEL_AVAILABLE.labels(model=metric_name).set(1.0 if available else 0.0)


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


def require_metrics_token(value: Optional[str]) -> str:
    """Require an explicit secret so metrics can never start unguarded."""

    if not value:
        raise RuntimeError("METRICS_BEARER_TOKEN is required")
    return value


def operation_label(value: str) -> str:
    """Normalize operation values before they reach metric labels."""

    return value if value in OPERATIONS else "unknown"


def rpc_method_label(value: str) -> str:
    """Normalize gRPC method values before they reach metric labels."""

    return value if value in RPC_METHODS else "unknown"


def record_rpc(method: str):
    """Record a bounded result and duration for an async gRPC method."""

    method_label = rpc_method_label(method)

    def decorator(function):
        @wraps(function)
        async def instrumented(*args, **kwargs):
            start = time.perf_counter()
            outcome = "success"
            try:
                return await function(*args, **kwargs)
            except Exception:
                outcome = "error"
                raise
            finally:
                duration = time.perf_counter() - start
                RPC_REQUESTS_TOTAL.labels(method=method_label, outcome=outcome).inc()
                RPC_REQUEST_DURATION_SECONDS.labels(
                    method=method_label, outcome=outcome
                ).observe(duration)

        return instrumented

    return decorator


@contextmanager
def record_job(operation: str):
    """Record completion and duration for an asynchronous operation job."""

    operation_name = operation_label(operation)
    start = time.perf_counter()
    outcome = "success"
    try:
        yield
    except Exception:
        outcome = "error"
        raise
    finally:
        duration = time.perf_counter() - start
        OPERATION_JOBS_TOTAL.labels(operation=operation_name, outcome=outcome).inc()
        OPERATION_JOB_DURATION_SECONDS.labels(
            operation=operation_name, outcome=outcome
        ).observe(duration)


def record_queue_event(operation: str, outcome: str) -> None:
    """Record a controlled queue transition."""

    controlled_outcome = (
        outcome
        if outcome in {"enqueued", "dequeued", "duplicate", "error"}
        else "error"
    )
    OPERATION_QUEUE_EVENTS_TOTAL.labels(
        operation=operation_label(operation), outcome=controlled_outcome
    ).inc()


def record_queue_depth(depth: int) -> None:
    """Record the authoritative Redis queue depth observed at scrape time."""

    OPERATION_QUEUE_DEPTH.set(max(0, depth))


def record_callback(channel: str, outcome: str, duration_seconds: float) -> None:
    """Record one callback dependency attempt without target/error labels."""

    channel_label = channel if channel in {"grpc", "http"} else "unknown"
    outcome_label = outcome if outcome in {"success", "error"} else "error"
    CALLBACK_REQUESTS_TOTAL.labels(
        channel=channel_label, outcome=outcome_label
    ).inc()
    CALLBACK_REQUEST_DURATION_SECONDS.labels(
        channel=channel_label, outcome=outcome_label
    ).observe(duration_seconds)


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
        operation_name = operation_label(operation)
        MODEL_OPERATION_TOTAL.labels(
            operation=operation_name, outcome=final_outcome
        ).inc()
        MODEL_OPERATION_DURATION_SECONDS.labels(
            operation=operation_name, outcome=final_outcome
        ).observe(duration)
