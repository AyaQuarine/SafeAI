# SafeAI Common V1.0 — Offline Functional Pipeline

Recette hors ligne R01 / R02 / R06 — 10 novembre 2026.

Environnement de recette : `.venv-diarization` (Whisper + Community-1 + Silero). Whisper local `.cache/faster-whisper-small`. Aucun jeton Hugging Face dans le dépôt.

```powershell
$env:PYTHONPATH = "src"
.\.venv-diarization\Scripts\python.exe -m safeai.pipeline inspect `
  --file data/local_dev/r06_deux_locuteurs.wav `
  --out data/local_dev/common_r06.json `
  --whisper-model .cache/faster-whisper-small `
  --session-id session_common_r06
```

Chaque `--out` écrit aussi `*_diarization.json` (même `session_id`).

## Décision de clôture

| Jalon | Décision |
| --- | --- |
| Acquisition, Whisper, Community-1 | Intégrés |
| AlignmentEngine | Fonctionnel |
| CommonPipeline hors ligne | Fonctionnel |
| COMMON-016…020 | Acceptés |
| R01 | Accepté avec réserve ASR |
| R02 | Conforme sur cet essai |
| R06 | Accepté en `partial` documenté |
| Gel Common V1.0 | **Autorisé** après protection silence non vérifié |

Pas de développement fonctionnel commun supplémentaire avant livraison GitHub. WER / DER / latence / continu = jalons ultérieurs.

## Synthèse recette

| Scénario | Fichier | Statut | Détail |
| --- | --- | --- | --- |
| R01 | `r01_officiel.wav` | `completed` | 2 énoncés, 1 locuteur, `session_common_r01` |
| R02 | `r02_silence.wav` | `empty` | Hallucination Amara dans `diagnostics`, `session_common_r02` |
| R06 | `r06_deux_locuteurs.wav` | `partial` | 10 énoncés, 2 locuteurs, 9 tours / 2 overlaps, `session_common_r06` |

Suite automatisée :

| Moment | Résultat |
| --- | --- |
| Gel backend Common V1.0 (COMMON-016…020 + silence non vérifié) | **174** tests réussis, **0** ignoré |
| Version enrichie frontend / Docker (page `/common` + 3 tests UI) | **177** tests réussis, **0** ignoré |

Les 174 correspondent au backend gelé ; les 177 incluent les tests de démonstration UI ajoutés ensuite.

## Protections retenues

| Ticket | Rôle |
| --- | --- |
| COMMON-016 | Silence confirmé vs hallucination ASR ; `None` ≠ silence |
| COMMON-017 | Filtrage temporel mot à mot (support Silero / Community-1, seuil 25 %) |
| COMMON-018 | Export diarisation indépendant du texte (overlaps conservés) |
| COMMON-019 | Token non lexical trop long (défaut > 2 s) écarté |
| COMMON-020 | Parole Community-1 non transcrite → diagnostics |
| Gel | ASR vide + diarisation vide **sans** `silence_confirmed` → `failed` (« Silence non vérifié ») |

## R06 — justification de `partial`

- Textes suspects écartés (Il, en, 1,, Sous, -titres).
- Token `?` 24,64–39,50 s écarté (COMMON-019).
- Parole Community-1 non transcrite signalée (25,55–29,04 ; 30,07–34,73 ; 32,72–33,04 ; 35,00–36,03).
- Chevauchement acoustique ~321 ms conservé dans `common_r06_diarization.json` (`overlap_count = 2` = deux tours chevauchants, **un** intervalle d'intersection).

## Frontend et Docker (kit équipe)

- Page `/common` : visualisation des JSON aligné + diarisation (exemples fictifs inclus). **Aucune** inférence depuis le navigateur.
- `Dockerfile` + `compose.yaml` : service `demo` (UI) ; profil `models` / service `offline` optionnel.
- Checklist : `docs/CHECKLIST_AVANT_PUSH.md`.

## Livraison GitHub (préparation)

Inclure : code `src/`, tests, docs, fixtures fictives `data/interface_commune/` et `src/safeai/ui/static/examples/`, configs, `pyproject.toml`, Docker, `.gitignore`.

Exclure (déjà couvert par `.gitignore`) : WAV réels (`data/local_dev/*`), `.cache/`, `.venv*`, `.env*`, `outputs/`, modèles ONNX, jetons HF.
