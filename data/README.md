# Données Inside

Les données réelles ne sont pas fournies dans ce squelette. Télécharger l'archive
depuis Inside selon les droits du groupe, puis l'extraire dans `data/raw/`.
Le premier responsable data doit confirmer le format réel avant l'entraînement.

Contrat attendu (noms exacts, espaces et majuscules inclus) :

```text
raw/
├── Safe/*.png
├── Safe Overall/*.png
├── Bad Weather/*.png
├── Tandem/*.png
└── Reckless/*.png
```

PNG et JPEG acceptés. Au moins 10 images par classe pour le split initial 70/15/15.
`prepare.py` vérifie les images, refuse les doublons binaires, produit un manifeste
avec SHA-256 et fige les splits de manière déterministe. Il ne supprime aucune donnée.

**À confirmer avant toute mesure** : une image correspond-elle à une trajectoire
indépendante ? Si plusieurs images partagent un trajet, conducteur ou session,
ajouter cet identifiant au manifeste et effectuer un split par groupe. Le split
actuel par image ne protège pas de cette fuite. Conserver le jeu de test final
hors des choix d'hyperparamètres et des expérimentations répétées.

Git ignore les images. DVC peut les versionner via un stockage partagé autorisé.
Ne jamais utiliser un chemin `C:/Users/...` comme remote pour tout le groupe.
Ne pas publier les données Inside sur GitHub sans autorisation.

