# Revue avant le premier push GitHub

Kit documentation / frontend / Docker fusionné dans le dépôt Common V1.0. Les modules communs gelés (acquisition, Whisper, Community-1, alignement, pipeline) n'ont pas été réécrits.

## État local (11 novembre 2026)

| Vérification | Résultat |
| --- | --- |
| Fusion `app.py` | Greffe `/common` + `SAFEAI_DISABLE_MICROPHONE` sur l'UI V1 existante |
| `pyproject.toml` | `1.0.0`, deps V1 conservées, `safeai-pipeline` conservé, extras Docker + package-data exemples |
| `pytest tests/ui/test_common_dashboard.py` | **3/3** réussis |
| `pytest tests -q` (`.venv` avec corpus local) | **177** réussis, 0 ignoré (174 = gel backend ; +3 UI) |
| Copie = fichiers indexés seulement | **173** réussis, **4** ignorés (`test_french_conversations` sans `data/conversations_fr/`) |
| `docker compose config` | OK ; port hôte **8766** |
| UI native `/common` | **OK** — exemples completed / empty / R06 réel |
| R06 dans `/common` | `Partielle`, 2 locuteurs, 10 énoncés, 4 unknown, 4 diagnostics non transcrite, 1 intersection / 2 tours overlap |
| `docker compose build demo` | **OK** — image légère `[ui]` sans ffmpeg/torch CUDA |
| `docker compose up -d demo` | **OK** — `http://127.0.0.1:8766/common` (micro désactivé) |
| Profil `models` / offline | Non validé — optionnel |

## État Git — 11 novembre 2026

- Branche : `main`
- Commit initial : `bf72f48`
- Commit de finalisation : `f129976`
- Fichiers enregistrés : **146**
- Copie propre : **173** passed, **4** skipped
- Tests locaux avec corpus : **177** passed
- Working tree : propre
- Remote : `https://github.com/AyaQuarine/SafeAI.git`
- Push : effectué (`main` suit `origin/main`)

Common V1.0 : fonctionnel hors ligne.  
Docker demo : validé.  
Docker offline/models : non validé (optionnel).

## Avant publication

- [x] Comparer `app.py` / `pyproject.toml` avant fusion (fait).
- [x] `pytest tests -q` sur la version exacte livrable.
- [x] `pytest tests/ui/test_common_dashboard.py -q`.
- [x] Ouvrir `/common` en natif (uvicorn) et charger les exemples.
- [x] `docker compose build demo` + `up -d demo` → `/common` sur `8766`.
- [ ] **Optionnel** : `docker compose --profile models build offline`.
- [x] Confirmer R01 / R02 / R06 déjà validés (R06 reste `partial`).
- [x] Premier commit local `bf72f48` + contrôles d'index.
- [ ] Accorder le partage des captures / transcriptions réelles du PDF.
- [ ] Ne pas présenter les exemples `/static/examples` comme des résultats réels.
- [x] Dépôt GitHub privé créé et `main` poussé (`origin`).

## Contenu à transmettre

Inclure : `src/`, `tests/`, `docs/`, `configs/`, `data/interface_commune/`, `Dockerfile`, `compose.yaml`, `.dockerignore`, `pyproject.toml`, `README.md`, `scripts/verify_common_pair.py`.

Exclure : `data/local_dev/*` (WAV), `.cache/`, `.venv*`, `.env*`, modèles, jetons.
