"""The Grafana dashboard and Prometheus alerts must only query metrics the API exposes."""

import json
import re
from pathlib import Path

import yaml

from trajectory_mlops.api.metrics import ApiMetrics

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = ROOT / "infra/grafana/dashboards/trajectory-api.json"
ALERTS = ROOT / "infra/prometheus/alerts.yml"
SCRAPE = ROOT / "infra/prometheus/prometheus.yml"
DATASOURCE = ROOT / "infra/grafana/provisioning/datasources/prometheus.yml"
METRIC = re.compile(r"\btrajectory_[a-z_]+")


SUFFIXES = {"counter": ("_total",), "histogram": ("_bucket", "_sum", "_count"), "gauge": ("",)}


def exposed_sample_names() -> set[str]:
    # Built from declared metric types: labelled metrics have no samples before the first request.
    families = ApiMetrics().registry.collect()
    return {family.name + suffix for family in families for suffix in SUFFIXES[family.type]}


def dashboard_exprs() -> list[str]:
    panels = json.loads(DASHBOARD.read_text())["panels"]
    return [t["expr"] for p in panels for t in p.get("targets", [])]


def alert_exprs() -> list[str]:
    groups = yaml.safe_load(ALERTS.read_text())["groups"]
    return [rule["expr"] for group in groups for rule in group["rules"]]


def test_dashboard_and_alerts_only_use_exposed_metrics():
    exposed = exposed_sample_names()
    used = {name for expr in dashboard_exprs() + alert_exprs() for name in METRIC.findall(expr)}
    assert used, "No trajectory_* metric found in the monitoring config."
    assert used <= exposed, f"Unknown metrics: {sorted(used - exposed)}"


def test_scrape_job_matches_queries_and_datasource_uid():
    job = yaml.safe_load(SCRAPE.read_text())["scrape_configs"][0]["job_name"]
    for expr in dashboard_exprs() + alert_exprs():
        if "up{" in expr:
            assert f'job="{job}"' in expr
    uid = yaml.safe_load(DATASOURCE.read_text())["datasources"][0]["uid"]
    panels = json.loads(DASHBOARD.read_text())["panels"]
    assert {p["datasource"]["uid"] for p in panels if "datasource" in p} == {uid}
