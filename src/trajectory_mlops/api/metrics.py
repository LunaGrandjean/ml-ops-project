"""Bounded-cardinality Prometheus metrics, isolated for each application."""

from time import perf_counter

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from trajectory_mlops.config import CLASSES


class ApiMetrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "trajectory_api_requests_total",
            "Completed API requests.",
            ("method", "route", "status"),
            registry=self.registry,
        )
        self.duration = Histogram(
            "trajectory_api_request_duration_seconds",
            "API response duration in seconds.",
            ("method", "route"),
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
            registry=self.registry,
        )
        self.predictions = Counter(
            "trajectory_predictions_total",
            "Successful predictions by selected class.",
            ("class",),
            registry=self.registry,
        )
        self.confidence = Histogram(
            "trajectory_prediction_confidence",
            "Uncalibrated top-class softmax score.",
            ("class",),
            buckets=(0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0),
            registry=self.registry,
        )
        self.ready = Gauge(
            "trajectory_model_ready",
            "One when a model is loaded, otherwise zero.",
            registry=self.registry,
        )
        for label in CLASSES:
            self.predictions.labels(label)
            self.confidence.labels(label)
        self.ready.set(0)

    def prediction(self, label: str, confidence: float):
        self.predictions.labels(label).inc()
        self.confidence.labels(label).observe(confidence)

    def render(self) -> bytes:
        return generate_latest(self.registry)


class MetricsMiddleware:
    def __init__(self, app, metrics: ApiMetrics):
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] in {
            "/metrics",
            "/health/live",
            "/health/ready",
        }:
            await self.app(scope, receive, send)
            return
        start, status = perf_counter(), 500

        async def observe_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, observe_send)
        finally:
            # Never label by raw path, filename, request IDs, version or query string.
            route = getattr(scope.get("route"), "path", "unmatched")
            method = scope["method"]
            if method not in {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}:
                method = "OTHER"
            self.metrics.requests.labels(method, route, str(status)).inc()
            self.metrics.duration.labels(method, route).observe(perf_counter() - start)
