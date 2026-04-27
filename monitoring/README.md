# Monitoring Runbook (Prometheus + Grafana)

This folder contains the full observability stack for the CAFA-5 MLOps services.
It is designed to be reproducible and version-controlled (no click-only drift).

## Scope

- Metrics collection with Prometheus.
- Dashboard provisioning with Grafana.
- Basic operational alerting.
- Dashboard JSON versioning workflow in git.

## Files In This Folder

```text
monitoring/
├── prometheus.yml
├── alerts.yml
└── grafana/
    ├── dashboards/
    │   ├── cafa5-service-health.json
    │   └── cafa5-domain-pipelines.json
    └── provisioning/
        ├── datasources/
        │   └── prometheus.yml
        └── dashboards/
            └── providers.yml
```

## What Is Implemented

### Prometheus

- Scrapes only Prometheus-compatible `/metrics` endpoints for service availability:
  - `prometheus`
  - `embedding_api_metrics`
  - `go_prediction_api_metrics`
  - `trainer_api_metrics`
- Loads alert rules from `alerts.yml`.

Why this matters:
- JSON health endpoints are not Prometheus exposition format.
- Using `/metrics` jobs for `up` avoids scrape-format false alerts.

### Grafana Provisioning

- Prometheus datasource is auto-provisioned:
  - UID: `prometheus`
  - URL: `http://prometheus:9090`
- Dashboards are file-provisioned from `/etc/grafana/dashboards`.
- Dashboard provider auto-refreshes periodically.

### Dashboards

- `cafa5-service-health.json`
  - target up/down
  - request rate
  - 5xx ratio
  - p95 latency
  - in-flight requests

- `cafa5-domain-pipelines.json`
  - embedding queue/outcomes/duration/sequence-length signals
  - training queue/failure reasons/duration by mode
  - inference latency by model version / input validation failures / top_k usage

### Alerts (minimal and intentional)

Current minimal alert set:

- `Cafa5ServiceMetricsTargetDown`
  - Trigger: metrics scrape target down for >2m.
  - Goal: detect service unavailability.

- `Cafa5HighHttp5xxRatio`
  - Trigger: sustained high 5xx ratio with minimum traffic.
  - Goal: detect user-visible API quality regressions.

- `Cafa5EmbeddingQueueBacklogHigh`
  - Trigger: queued embedding jobs above threshold for >10m.
  - Goal: detect pipeline saturation.

## Startup And Verification

## 1) Start monitoring profile

From repo root:

```bash
make monitoring-up
```

Equivalent:

```bash
docker compose --profile monitoring up -d
```

## 2) Check containers

```bash
docker compose ps
```

Expected:
- `prometheus` running
- `grafana` running

## 3) Verify Prometheus is ready

```bash
curl -s http://127.0.0.1:9090/-/ready
```

Expected: `Prometheus is Ready.`

## 4) Verify active targets

Open Prometheus targets page:
- <http://127.0.0.1:9090/targets>

Or query:

```bash
curl -s "http://127.0.0.1:9090/api/v1/query?query=up"
```

Expected:
- `up=1` for `prometheus`, `embedding_api_metrics`, `go_prediction_api_metrics`
- `trainer_api_metrics` is `1` only when training profile service is running.

## 5) Verify rules and active alerts

```bash
curl -s http://127.0.0.1:9090/api/v1/rules
curl -s http://127.0.0.1:9090/api/v1/alerts
```

## 6) Open Grafana

- URL: <http://127.0.0.1:3000>
- Default credentials are set via `docker-compose.yml`.
- Confirm:
  - Prometheus datasource exists and is healthy.
  - `CAFA5 Service Health` dashboard loads.
  - `CAFA5 Domain Pipelines` dashboard loads.

## Reload Behavior

If you edit monitoring files:

- Prometheus config/rules:
  - soft reload:
    ```bash
    curl -X POST http://127.0.0.1:9090/-/reload
    ```
  - if needed, restart Prometheus container:
    ```bash
    docker compose restart prometheus
    ```

- Grafana provisioning/dashboard JSON:
  - provider auto-refresh is enabled.
  - if changes do not appear quickly, restart Grafana:
    ```bash
    docker compose restart grafana
    ```

## Dashboard Export And Versioning Workflow (Milestone 6 hardening)

Use this workflow to keep dashboards merge-friendly and reproducible:

1. **Edit source of truth in repo**
   - Update JSON in `monitoring/grafana/dashboards/*.json`.
   - Prefer editing files directly instead of ad-hoc UI edits.

2. **If UI edits were made, export and normalize**
   - Export dashboard JSON from Grafana UI.
   - Replace file in `monitoring/grafana/dashboards/`.
   - Keep stable fields:
     - set `"id": null`
     - keep fixed `"uid"` per dashboard
     - increment `"version"` only when needed

3. **Validate dashboard JSON**
   - Open in Grafana and ensure no panel query errors.
   - Confirm variables resolve and panels render with data.

4. **Commit with intent**
   - Commit dashboard changes with a clear message:
     - what signal changed
     - why threshold/query/panel was updated

5. **Review checklist before merge**
   - Queries align with available labels.
   - No accidental high-cardinality labels introduced.
   - Panel titles/units are explicit.
   - Alerts still align with dashboard logic.

## Operational Troubleshooting

## Prometheus target is down

Symptoms:
- `up == 0`
- `Cafa5ServiceMetricsTargetDown` firing

Checks:

```bash
docker compose ps
docker compose logs --tail=200 embedding-api go-prediction-api trainer-api prometheus
```

Also verify endpoint from Prometheus container network perspective:
- service name and port are correct (`<service>:8000`).
- `/metrics` endpoint responds and is not protected by gateway auth internally.

## Grafana dashboard shows no data

Checks:
- Datasource UID matches dashboard datasource UID (`prometheus`).
- Time range is not too narrow.
- Prometheus query works directly in Prometheus UI.
- Target labels in panel query match actual labels.

## Alerts never fire or always fire

Checks:
- Rule expression in Prometheus graph first.
- Ensure `for:` duration is appropriate.
- Validate denominator guards (`clamp_min`) for ratio expressions.
- Confirm traffic floor conditions to avoid low-traffic noise.

## High-cardinality warning (important for route labels)

Problem:
- Dynamic path segments (job IDs, UUIDs) can explode time-series cardinality.

Current mitigation:
- `embedding-api` normalizes route labels in middleware (`_route_label`) before metric labeling.

Recommendation:
- Keep route labels templated/static across services.
- Avoid raw identifiers in labels.

## Suggested Routine During Model/Service Changes

When rolling new model versions or changing inference behavior:

1. Confirm inference `model_version` appears in metrics.
2. Compare p95 latency by `model_version`.
3. Check validation failure reasons for schema drift.
4. Watch embedding queue depth for upstream pressure.
5. Confirm 5xx ratio remains under threshold.

## Useful Queries

Service availability:

```promql
up{job=~"prometheus|embedding_api_metrics|go_prediction_api_metrics|trainer_api_metrics"}
```

HTTP 5xx ratio by service:

```promql
sum by (service) (rate(cafa5_http_requests_total{status_code=~"5.."}[5m]))
/
clamp_min(sum by (service) (rate(cafa5_http_requests_total[5m])), 0.001)
```

Embedding queue depth:

```promql
cafa5_embedding_queue_jobs{status="queued"}
```

Inference p95 latency by model version:

```promql
histogram_quantile(
  0.95,
  sum by (le, model_version) (rate(cafa5_inference_duration_seconds_bucket[5m]))
)
```

## Stop Monitoring Stack

```bash
make monitoring-down
```

