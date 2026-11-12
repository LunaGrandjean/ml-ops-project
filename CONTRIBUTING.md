# Travailler à cinq

1. Chaque tâche possède une issue, un responsable et des critères de fin.
2. Partir de `main` à jour ; créer une branche `feat/...`, `fix/...` ou `docs/...`.
3. Faire des commits courts et descriptifs. Aucun push direct sur main après bootstrap.
4. Ouvrir une PR ; demander la revue d'un autre membre.
5. Fusionner après CI verte. Résoudre les conflits dans la branche de travail.
6. Mettre à jour la documentation et conserver les run IDs des expériences retenues.

Protéger main dans GitHub : une approbation et les checks CI obligatoires.
Remplir TEAM.md avec les cinq noms réels et partager les responsabilités.

Ne pas committer les secrets, images Inside, poids ou caches. Les métadonnées DVC
(`.dvc`, `dvc.lock`) et `uv.lock` sont versionnées. Les notebooks éventuels
servent à explorer ; déplacer le code réutilisable dans `src/`.

Lors d'une modification de dépendances : `uv lock`, vérifier les tests et
committer ensemble `pyproject.toml` et `uv.lock`.
