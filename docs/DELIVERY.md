# Checklist de rendu

Date limite : **25 novembre 2026, 20 h (Europe/Paris)**.
Présentation : **26 novembre**, 10 min puis 10 min de questions.
Transmettre les noms des cinq membres avant le **20 novembre**.

## Dépôt et preuves

- [ ] Cinq noms et contributions complétés dans TEAM.md.
- [ ] Code preprocessing, entraînement, évaluation, API et interface.
- [ ] DAG fonctionnel visible dans dvc.yaml et via dvc dag.
- [ ] CI GitHub Actions verte ; historique de PR et revues d'équipe.
- [ ] Dataset versionné et accès partagé documenté, sans publication non autorisée.
- [ ] Runs MLflow, métriques, paramètres, versions et modèle stocké sur MinIO.
- [ ] Baselines, accuracy et macro-F1 réelles dans le README.
- [ ] Trois liens Docker Hub (api, ui, tracking).
- [ ] API, UI, MLflow et MinIO sur Kubernetes local.
- [ ] Swagger et une véritable prédiction testés de bout en bout.
- [ ] Captures ajoutées à docs/screenshots et reliées dans le README.
- [ ] Instructions validées depuis un nouveau clone.
- [ ] Tag final et commit exact du rendu consignés.

## Email à envoyer par le groupe

Destinataires : celia.doclot@hotmail.fr et matthieu.decibeins@protonmail.com
Objet contenant : **[ALBERT][MLOPS]**

Inclure :
- lien du dépôt GitHub, tag/commit final ;
- liste des cinq membres ;
- URLs des images Docker Hub ;
- URLs d'accès si déploiement distant (bonus), modalités d'accès si nécessaire.

Ce document est une checklist ; aucun email n'est envoyé automatiquement.

## Proposition pour les 10 minutes

1. Problème, données et cinq classes : 1 min.
2. Architecture et DAG : 2 min.
3. Tracking, baselines et résultats : 2 min.
4. Démo image → UI → API → modèle versionné sur Kubernetes : 3 min.
5. CI/CD, collaboration et limites : 2 min.

Préparer les réponses sur les fuites de données, le choix des métriques, le versionnement,
le stockage des modèles, les secrets, la promotion et le rollback.
