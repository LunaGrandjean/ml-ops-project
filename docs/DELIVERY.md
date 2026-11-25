# Submission checklist

Deadline: **25 November 2026, 20:00 (Europe/Paris)**.
Presentation: **26 November**, 10 min + 10 min of questions.
Group names sent to the teachers on **20 November** (done).

## Repository and evidence

- [x] Five names and contributions in TEAM.md.
- [x] Code for preprocessing, training, evaluation, API and web app.
- [x] Working DAG in dvc.yaml, visible with `dvc dag`.
- [x] Green GitHub Actions CI; pull request history with team reviews.
- [x] Versioned dataset with documented shared access, without unauthorised publication.
- [x] MLflow runs, metrics, parameters, versions and model stored on MinIO.
- [x] Real baselines, accuracy and macro-F1 in the README.
- [ ] Three Docker Hub links (api, ui, tracking) in the README.
- [ ] API, web app, MLflow and MinIO running on local Kubernetes.
- [ ] Swagger and a real prediction tested end to end.
- [ ] Screenshots added to docs/screenshots and linked from the README.
- [ ] Instructions checked from a fresh clone.
- [ ] Final tag and exact submitted commit recorded.

## Email to send as a group

Recipients: celia.doclot@hotmail.fr and matthieu.decibeins@protonmail.com
Subject containing: **[ALBERT][MLOPS]**

Include:
- GitHub repository link and final tag/commit;
- list of the five members;
- Docker Hub image URLs;
- access URLs if deployed remotely (bonus), with access details if needed.

This document is a checklist; no email is sent automatically.

## Suggested 10-minute plan

1. Problem, data and the five classes: 1 min.
2. Architecture and DAG: 2 min.
3. Tracking, baselines and results: 2 min.
4. Demo: image → web app → API → versioned model on Kubernetes: 3 min.
5. CI/CD, collaboration and limitations: 2 min.

Prepare answers on data leakage, metric choice, versioning, model storage, secrets,
promotion and rollback.
