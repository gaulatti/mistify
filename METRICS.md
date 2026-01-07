# Prometheus Metrics Documentation

This document describes all the Prometheus metrics exposed by the Mistify service on the `/metrics` endpoint.

## Overview

The Mistify service exposes comprehensive metrics for monitoring:
- HTTP request performance and errors
- Posts/items processing throughput
- Model operation performance
- GPU usage and anomalies
- System resource utilization
- Failures and timeouts

## Metric Categories

### 1. HTTP Metrics

These metrics track all HTTP requests to the service:

- **`mistify_http_requests_total`** (Counter)
  - Description: Total HTTP requests received
  - Labels: `method`, `route`, `status_code`
  - Use case: Calculate requests per minute (RPM)

- **`mistify_http_request_duration_seconds`** (Histogram)
  - Description: HTTP request latency in seconds
  - Labels: `method`, `route`, `status_code`
  - Buckets: 0.005 to 80 seconds
  - Use case: Monitor request latency and performance

- **`mistify_http_inprogress_requests`** (Gauge)
  - Description: Number of HTTP requests currently being processed
  - Use case: Monitor concurrent request load

- **`mistify_http_exceptions_total`** (Counter)
  - Description: Unhandled exceptions during request processing
  - Labels: `method`, `route`, `exception_type`
  - Use case: Track unexpected errors

### 2. Posts/Items Processing Metrics

These metrics track the volume of posts and items being processed:

- **`mistify_posts_processed_total`** (Counter)
  - Description: Total number of posts/items processed
  - Labels: `endpoint`
  - Use case: **Calculate posts analyzed per minute** by endpoint
  - Endpoints tracked: `analyze`, `cluster`, `classify`, `translate`, `detect`, `embed`, `generate_text`

- **`mistify_posts_batch_size`** (Histogram)
  - Description: Number of posts/items in each request
  - Labels: `endpoint`
  - Buckets: 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000
  - Use case: **Track how many posts come in every request**

### 3. Model Operation Metrics

These metrics track individual model operations:

- **`mistify_model_operation_total`** (Counter)
  - Description: Total model/operation executions
  - Labels: `operation`, `outcome`
  - Operations: `analyze`, `cluster`, `classify`, `translate`, `language_detect`, `embed`, `generate_text`
  - Outcomes: `success`, `error`, `timeout`
  - Use case: Track operation success/failure rates

- **`mistify_model_operation_duration_seconds`** (Histogram)
  - Description: Duration of model/operation executions in seconds
  - Labels: `operation`, `outcome`
  - Buckets: 0.001 to 80 seconds
  - Use case: Monitor operation performance

- **`mistify_model_available`** (Gauge)
  - Description: Whether a model/component is available (1) or not (0)
  - Labels: `model`
  - Models: `fasttext`, `classifier`, `translator`, `embedder`, `nlp`, `text_generator`
  - Use case: Monitor model availability

### 4. Failure and Retry Metrics

These metrics track failures and retries:

- **`mistify_operation_failures_total`** (Counter)
  - Description: Total number of operation failures
  - Labels: `operation`, `failure_type`
  - Failure types: `timeout`, `exception`, `invalid_result`, `null_result`, `parse_error`
  - Use case: **Track failures/retries** by operation and failure type

- **`mistify_operation_retries_total`** (Counter)
  - Description: Total number of operation retries
  - Labels: `operation`
  - Use case: Monitor retry patterns (reserved for future use)

### 5. GPU Usage Metrics

These metrics track GPU utilization and memory:

- **`mistify_gpu_memory_allocated_bytes`** (Gauge)
  - Description: GPU memory currently allocated by PyTorch in bytes
  - Labels: `device_id`
  - Use case: **Monitor GPU memory usage**

- **`mistify_gpu_memory_reserved_bytes`** (Gauge)
  - Description: GPU memory currently reserved by PyTorch in bytes
  - Labels: `device_id`
  - Use case: **Monitor GPU memory reservation**

- **`mistify_gpu_utilization_percent`** (Gauge)
  - Description: GPU utilization percentage (if nvidia-smi is available)
  - Labels: `device_id`
  - Use case: **Detect anomalies with GPU usage**

### 6. System Metrics

These metrics track system resource utilization:

- **`mistify_process_resident_memory_bytes`** (Gauge)
  - Description: Resident set size (RSS) of the current process
  - Use case: Monitor memory usage

- **`mistify_process_threads`** (Gauge)
  - Description: Number of threads in the current process
  - Use case: Monitor thread pool usage

- **`mistify_torch_device_available`** (Gauge)
  - Description: Whether a torch device backend is available (1) or not (0)
  - Labels: `device`
  - Devices: `cuda`, `mps`, `cpu`
  - Use case: Monitor available compute devices

- **`mistify_build_info`** (Info)
  - Description: Mistify build and runtime info
  - Labels: `service`
  - Use case: Service identification

## Example Prometheus Queries

### Requests Per Minute (RPM)
```promql
# Total requests per minute across all endpoints
rate(mistify_http_requests_total[1m]) * 60

# Requests per minute by endpoint
rate(mistify_http_requests_total[1m]) * 60 by (route)
```

### Posts Analyzed Per Minute
```promql
# Posts analyzed per minute across all endpoints
rate(mistify_posts_processed_total[1m]) * 60

# Posts analyzed per minute by endpoint
rate(mistify_posts_processed_total[1m]) * 60 by (endpoint)
```

### Average Posts Per Request
```promql
# Average batch size by endpoint
rate(mistify_posts_batch_size_sum[5m]) / rate(mistify_posts_batch_size_count[5m]) by (endpoint)
```

### Failure Rate
```promql
# Failure rate by operation
rate(mistify_operation_failures_total[5m]) by (operation, failure_type)

# Overall error rate
rate(mistify_http_requests_total{status_code=~"5.."}[5m])
```

### GPU Metrics
```promql
# GPU memory usage
mistify_gpu_memory_allocated_bytes

# GPU utilization
mistify_gpu_utilization_percent

# GPU memory anomaly detection (sudden changes)
deriv(mistify_gpu_memory_allocated_bytes[5m])
```

### Request Latency
```promql
# P95 latency by endpoint
histogram_quantile(0.95, rate(mistify_http_request_duration_seconds_bucket[5m])) by (route)

# P99 latency
histogram_quantile(0.99, rate(mistify_http_request_duration_seconds_bucket[5m])) by (route)
```

## Grafana Dashboard Recommendations

Create dashboards with the following panels:

1. **Traffic Overview**
   - Requests per minute (total and by endpoint)
   - Posts analyzed per minute
   - Average posts per request

2. **Performance**
   - Request latency percentiles (P50, P95, P99)
   - Operation duration by type
   - In-progress requests

3. **Errors & Failures**
   - HTTP error rate by status code
   - Operation failures by type
   - Exception counts

4. **Resource Usage**
   - GPU memory allocated/reserved
   - GPU utilization percentage
   - Process memory (RSS)
   - Thread count

5. **Model Health**
   - Model availability status
   - Operation success/failure rates
   - Timeout occurrences

## Alert Recommendations

Set up alerts for:

1. **High error rate**: `rate(mistify_http_requests_total{status_code=~"5.."}[5m]) > 0.05`
2. **High latency**: `histogram_quantile(0.95, rate(mistify_http_request_duration_seconds_bucket[5m])) > 10`
3. **GPU memory high**: `mistify_gpu_memory_allocated_bytes / mistify_gpu_memory_reserved_bytes > 0.9`
4. **Model unavailable**: `mistify_model_available == 0`
5. **High failure rate**: `rate(mistify_operation_failures_total[5m]) > 0.1`
