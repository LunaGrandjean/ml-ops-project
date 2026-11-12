# Équipe, responsabilités et calendrier

Remplir les cinq noms puis envoyer la composition du groupe avant le **20/11/2026**.

| Membre | Responsabilité principale | Livrable et revue croisée |
| --- | --- | --- |
| 1 — À renseigner | Données / extraction Inside | Contrat, version DVC, split sans fuite ; revue membre 2 |
| 2 — À renseigner | Modèle PyTorch / évaluation | Baselines, CNN amélioré, rapport ; revue membre 3 |
| 3 — À renseigner | MLflow / MinIO / pipeline | Runs, modèles S3, DAG reproductible ; revue membre 4 |
| 4 — À renseigner | API / interface / tests | Swagger, prédictions et tests HTTP ; revue membre 5 |
| 5 — À renseigner | Docker / K8s / CI/CD | Images Docker Hub et démonstration cluster ; revue membre 1 |

Chaque membre contribue au README et sait expliquer le parcours complet.
Répartir les issues ; éviter de réserver tous les tests à une seule personne.

| Période | Objectif vérifiable |
| --- | --- |
| 12–15 novembre | Valider les données et démarrer Compose ; CI verte |
| 16–19 novembre | Premier entraînement MLflow/S3 ; API et UI reliées au modèle |
| Avant le 20 novembre | Transmettre les noms du groupe |
| 20–22 novembre | Images Docker Hub ; services sur Kubernetes ; premiers tests e2e |
| 23–24 novembre | Stabiliser les résultats, screenshots et répétition de présentation |
| 25 novembre avant 20 h | Tag final, vérification depuis clone et envoi du rendu |
| 26 novembre | Présentation 10 min + questions 10 min |

Backlog minimum :
- [ ] Confirmer structure/labels et indépendance des trajectoires Inside.
- [ ] Mettre les données dans un remote partagé et versionné.
- [ ] Mesurer majorité et hasard ; améliorer le CNN si nécessaire.
- [ ] Tester promotion, version servie et rollback.
- [ ] Valider Compose, images Docker Hub et Kubernetes.
- [ ] Tester une prédiction depuis l'interface jusqu'au modèle réel.
- [ ] Compléter résultats et preuves dans le README.

Bonus seulement après le parcours minimum : réentraînement puis redéploiement
automatique après validation, monitoring, feature store.
