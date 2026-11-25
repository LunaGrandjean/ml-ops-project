# Team, responsibilities and schedule

Group members (sent to the teachers on 20/11/2026):

| Name | Role | Main contributions (Git history up to 25/11/2026) |
| --- | --- | --- |
| Viviam Baiz | Member 1 – Data Engineer | DVC data versioning on MinIO, ride-to-image converter, split by ride, EDA, training-only augmentation, data tests, experiments 1–3 |
| Hoang Ngoc-Tran | Member 2 – ML Engineer | Architecture justification, hyperparameter tuning, evaluation, model handoff bundle |
| Luna Grandjean | Member 3 – MLOps Engineer (+ backend) | Champion/challenger promotion, retraining workflow, release of promoted models, FastAPI serving with confidence scores, Streamlit app, API metrics |
| Dhia Ben Cheikh | Member 5 – DevOps Engineer (monitoring) | Prometheus scrape config, Grafana dashboard and alert rules |
| Malek Saoudi | To confirm | To complete |

Cross-review: each member reviews the pull requests of another member
(1 → 2 → 3 → 4 → 5 → 1).

Every member contributes to the README and can explain the whole pipeline.
Tests are shared: each member writes the tests for their own part.

| Period | Checkable goal |
| --- | --- |
| 12–15 November | Validate the data and start Compose; green CI |
| 16–19 November | First MLflow/S3 training; API and web app connected to the model |
| Before 20 November | Send the group names (done) |
| 20–22 November | Docker Hub images; services on Kubernetes; first e2e tests |
| 23–24 November | Stabilise results, screenshots and presentation rehearsal |
| 25 November before 20:00 | Final tag, check from a fresh clone and send the submission |
| 26 November | 10 min presentation + 10 min questions |

Minimum backlog:
- [x] Confirm the data structure, labels and ride independence (split by ride).
- [x] Put the data in a versioned remote (DVC on MinIO).
- [x] Measure the majority and random baselines; improve the CNN.
- [x] Test promotion, served version and rollback.
- [ ] Validate Compose, Docker Hub images and Kubernetes.
- [x] Test a prediction from the web app down to the real model.
- [x] Complete results and evidence (screenshots) in the README.

Bonus after the minimum path: retraining then automatic redeployment after validation
(done), monitoring (done in Compose: API metrics, Prometheus alerts, Grafana dashboard), feature store (not done).
