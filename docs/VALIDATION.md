# Validation du socle

Vérifications effectuées à la création du projet :
- **17 tests réussis**, un test e2e de service déployé exclu de cette exécution.
- Tests unitaires : prétraitement, split déterministe/disjoint, doublons, quality gate.
- Tests HTTP : santé, contrat de prédiction, formats invalides et taille maximale.
- Intégration ML : entraînement sur 60 images synthétiques, tracking MLflow SQLite
  temporaire, évaluation, registre puis prédiction avec le modèle rechargé.
- Ruff : lint et format conformes.
- `uv.lock` vérifié ; distributions Python wheel et source construites.
- `docker compose config --quiet` valide.
- Tests de cohérence du DAG, des dépendances Compose et du rendu des manifests K8s.

Environnement de vérification : Python 3.12, dépendances de uv.lock, CPU.
Des avertissements de dépréciation proviennent de TorchScript et de dépendances ;
ils n'empêchent pas les tests de passer. Une migration vers torch.export pourra
être étudiée après stabilisation du prototype.

**Aucun résultat sur les données Inside n'est établi.** Le dataset synthétique sert
uniquement à vérifier les branchements du pipeline et du service.

Docker Desktop n'étant pas démarré sur la machine de création, les builds des
images, les échanges réels MLflow/MinIO et le cluster Kubernetes restent à vérifier.
Le backend MLflow temporaire du test ne valide pas le stockage S3.

À vérifier sur l'environnement de groupe :
- build et démarrage Docker Compose ;
- stockage des artefacts dans MinIO et persistance après redémarrage ;
- accès aux données partagées ;
- CI GitHub, publication Docker Hub et déploiement Kubernetes ;
- prédiction réelle depuis Streamlit et Swagger ;
- baselines et métriques du dataset réel ;
- promotion, version servie et rollback.

Ajouter les screenshots et résultats réels au README avant le rendu.
