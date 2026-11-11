# Kit d'intégration — SafeAI Common V1.0

**Ce ZIP est un complément à appliquer sur le projet Common V1.0 existant ; ce n'est PAS le backend complet.** Aucun module d'acquisition, d'alignement, de Whisper ou de Community-1 n'a été remplacé dans le kit.

## Fichiers modifiés

- `src/safeai/ui/app.py` : routes `/common` et `/api/capabilities`, microphone désactivable dans Docker.
- `src/safeai/ui/static/index.html` et `styles.css` : navigation vers la nouvelle vue, sans détruire l'ancien diagnostic.
- `pyproject.toml` : métadonnées V1.0 et inclusion des exemples JSON dans le paquet Python.
- `README.md` : documentation V1 avec les chemins vers le guide équipe.

## Fichiers ajoutés

- `src/safeai/ui/static/common.html`, `common.js`, `common.css` et `static/examples/*.json`.
- `docs/DEMARRAGE_EQUIPE.md`, `docs/FRONTEND_COMMON_V1.md`, `docs/DOCKER_EQUIPE.md`, `docs/ARCHITECTURE_COMMON_V1.md`, `docs/CHECKLIST_AVANT_PUSH.md`.
- `Dockerfile`, `compose.yaml`, `.dockerignore`, `tests/ui/test_common_dashboard.py`, `scripts/verify_common_pair.py`.

## Mode opératoire (important)

1. **Faire une copie de sauvegarde ou une branche locale non publiée** de ton projet. Le ZIP contient un `app.py` créé à partir de la version d'interface qui avait été partagée au début de la V0 ; comparer avant de remplacer si l'interface de ton PC a évolué depuis.
2. Copier le contenu du kit à la racine du dépôt Common V1.0 actuel. Pour `app.py` et `pyproject.toml`, examiner les différences avant remplacement.
3. Exécuter `python -m pytest tests -q`, puis `python -m pytest tests/ui/test_common_dashboard.py -q` et ouvrir `/common`.
4. Vérifier le Docker Compose sur ta propre machine avec `docker compose config` et `docker compose up --build demo`.
5. Ne publier dans Git qu'après avoir contrôlé l'intégralité des fichiers indexés et obtenu l'accord de l'équipe.

## Résultats de tests obtenus pendant la préparation

- **5 tests de l'interface** réussis dans une copie reconstruite du candidat antérieur.
- **136 tests réussis, 4 ignorés** sur cette copie reconstruite V0 enrichie de l'interface (Silero et les corpus de recette non disponibles dans cet environnement).
- **174** tests (gel backend) puis **177** (backend + UI `/common`) sur la V1 Windows de référence ; **non reproduits** automatiquement sur chaque machine.
- Les images Docker et le frontend dans un navigateur graphique **n'ont pas été exécutés** ici, faute de moteur Docker et de navigateur de test disponible. La configuration Docker a été contrôlée statiquement.

## Préserver la distinction

Cette vue de démonstration **ne lance pas de modèles depuis le navigateur**. Pour un R06 réel, le backend hors ligne tourne via CLI et l'équipe ouvre ensuite les deux JSON localement. Ceci est volontaire : autonomie du frontend, confidentialité et démo rapide.
