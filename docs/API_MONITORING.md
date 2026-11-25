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

## Prometheus and Grafana (Compose)

`docker compose up -d` starts Prometheus and Grafana next to the API.
Set `GRAFANA_ADMIN_PASSWORD` in `.env` first (see `.env.example`).

| Service | URL | Configuration |
| --- | --- | --- |
| Prometheus | http://localhost:9090 | `infra/prometheus/prometheus.yml` scrapes `api:8000/metrics` every 15 s (job `trajectory-api`), 7-day retention |
| Grafana | http://localhost:3000 (user `admin`) | Datasource and dashboard provisioned from `infra/grafana`; the "Trajectory API · monitoring" dashboard (folder MLOps) is the home page |

The dashboard has two rows: service health (scrape status, model loaded,
`/predict` throughput, 5xx share, p50/p95 latency) and prediction distribution
(predictions per class, class share over 30 minutes, mean top-class score).
It is read-only in the UI: edit `infra/grafana/dashboards/trajectory-api.json` instead.

Alert rules live in `infra/prometheus/alerts.yml` and show under Prometheus > Alerts.
There is no Alertmanager, so nothing is sent.

| Alert | Fires when | Severity |
| --- | --- | --- |
| ApiDown | Prometheus cannot scrape the API for 1 min | page |
| ModelNotReady | `trajectory_model_ready == 0` for 2 min | page |
| PredictErrorRateHigh | over 5% of `/predict` requests return 5xx for 2 min | page |
| PredictLatencyP95High | `/predict` p95 above 250 ms for 5 min | ticket |
| RecklessShareHigh | over 10% Reckless predictions in 30 min (at least 20 predictions; test baseline 2.2%) | ticket |

CI checks the configuration with `promtool check config`, and
`tests/unit/test_monitoring_config.py` fails if the dashboard or alerts query a
metric the API does not expose.

Kubernetes: the API pod template includes prometheus.io/scrape, port and path
annotations, but the cluster deployment does not install Prometheus or Grafana.
Monitoring runs in the Compose stack only.

Latency is in seconds and includes image decoding and inference. The confidence
panel shows uncalibrated model scores; it cannot measure live accuracy or drift
without ground-truth labels. Compare class volumes over time before interpreting
score changes.

Generate a valid prediction and an invalid upload, then check /metrics.
The integration tests verify success/error counts, histograms, readiness,
bounded labels and isolation between application instances.
