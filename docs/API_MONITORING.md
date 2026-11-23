# API monitoring handoff

The API exposes Prometheus text format at GET /metrics on port 8000.
No extra monitoring process is needed inside the API container.

| Metric | Labels | Meaning |
| --- | --- | --- |
| trajectory_api_requests_total | method, route, status | Completed requests, including upload and inference errors |
| trajectory_api_request_duration_seconds | method, route | Request duration histogram |
| trajectory_predictions_total | class | Successful predictions by selected class |
| trajectory_prediction_confidence | class | Top-class softmax score histogram |
| trajectory_model_ready | none | Loaded model: 1; unavailable: 0 |

Readiness/liveness probes and scrapes are excluded from request metrics.
Unknown paths share `unmatched`; unrecognized methods share `OTHER`.
No filenames, ride IDs, query strings, request IDs or model versions are labels.
Rejected images and failed inference do not count as successful predictions.

Each API process has its own registry. The current single-worker deployment is
supported. For multiple replicas, scrape each pod rather than a load-balanced
Service and aggregate in Prometheus. Do not enable multiple Uvicorn workers
without adding Prometheus multiprocess support.

## Member 5 integration

For Compose, place Prometheus on the API's network and add a scrape job:

```yaml
scrape_configs:
  - job_name: trajectory-api
    scrape_interval: 15s
    metrics_path: /metrics
    static_configs:
      - targets: ["api:8000"]
```

For Kubernetes, the API pod template includes prometheus.io/scrape, port and path
annotations. Configure Prometheus pod discovery to honor them in namespace
trajectory-mlops, or configure an explicit pod scrape. Annotations alone do not
install Prometheus or create a dashboard. There is no ServiceMonitor dependency.

Suggested Grafana panels:

```promql
sum(rate(trajectory_api_requests_total{route="/predict"}[5m]))
sum(rate(trajectory_api_requests_total{route="/predict",status=~"5.."}[5m]))
histogram_quantile(0.95, sum by (le) (rate(trajectory_api_request_duration_seconds_bucket{route="/predict"}[5m])))
sum by (class) (rate(trajectory_predictions_total[5m]))
sum by (class) (rate(trajectory_prediction_confidence_sum[5m])) / clamp_min(sum by (class) (rate(trajectory_prediction_confidence_count[5m])), 0.000001)
min(trajectory_model_ready)
```

Latency is in seconds and includes image decoding and inference. The confidence
panel shows uncalibrated model scores; it cannot measure live accuracy or drift
without ground-truth labels. Compare class volumes over time before interpreting
score changes.

Generate a valid prediction and an invalid upload, then check /metrics.
The integration tests verify success/error counts, histograms, readiness,
bounded labels and isolation between application instances.
