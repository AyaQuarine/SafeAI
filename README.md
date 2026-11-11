# SafeAI Common V1.0 — Offline Functional Pipeline

Chaîne commune **locale** : un WAV → acquisition (Silero + qualité) → Whisper → Community-1 → alignement → JSON pour les équipes texte et voix.

Livraison fonctionnelle hors ligne validée sur R01 / R02 / R06. Ce n'est **pas** encore la transcription continue, ni une validation WER / DER / latence.

SafeAI est un système **pédagogique, non médical, non certifié**. Il n'est pas autorisé à prendre automatiquement des décisions d'urgence.

## Périmètre V1.0

| Composant | Statut |
| --- | --- |
| Acquisition 16 kHz mono + Silero + qualité | Intégré |
| Whisper (faster-whisper, hors ligne) | Intégré |
| Community-1 (pyannote, `.venv-diarization`) | Intégré |
| `AlignmentEngine` + `AlignedSession` | Fonctionnel |
| `CommonPipeline` + CLI `safeai.pipeline` | Fonctionnel |
| Sortie diarisation indépendante (`*_diarization.json`) | COMMON-018 |
| Cohérence acoustique / filtrage temporel / diagnostics | COMMON-016…020 |

R01 accepté avec réserve ASR · R02 conforme (`empty`) · R06 accepté en `partial` documenté. Détail : `docs/common_v1_hors_ligne.md`.

## Guides pour l'équipe

| Document | Contenu |
| --- | --- |
| [`docs/DEMARRAGE_EQUIPE.md`](docs/DEMARRAGE_EQUIPE.md) | Installation, CLI, sorties, limites |
| [`docs/ARCHITECTURE_COMMON_V1.md`](docs/ARCHITECTURE_COMMON_V1.md) | Circulation des données |
| [`docs/FRONTEND_COMMON_V1.md`](docs/FRONTEND_COMMON_V1.md) | Page `/common` (visualisation JSON seulement) |
| [`docs/DOCKER_EQUIPE.md`](docs/DOCKER_EQUIPE.md) | Démo légère et profil modèles |
| [`docs/CHECKLIST_AVANT_PUSH.md`](docs/CHECKLIST_AVANT_PUSH.md) | Sécurité et validation avant publication |
| [`docs/interface_commune.md`](docs/interface_commune.md) | Contrats texte / voix |

Démonstration UI (après analyse CLI) :

```powershell
python -m safeai.acquisition ui
# puis http://127.0.0.1:8765/common
```

Docker — démonstration légère (pas d'inférence navigateur) :

```powershell
docker compose up --build demo
# http://127.0.0.1:8765/common
```

Profil modèles (CPU, HF, plus lourd) : `docker compose --profile models build offline` — à valider localement ; micro Windows hors Docker.

## Module 01 — Acquisition audio et contrôle de qualité

Premier pilier : flux audio normalisé et horodaté, détection de parole, qualité technique, incidents. Cette brique **n'interprète pas** le contenu et n'identifie pas les locuteurs.

## Ce que fait ce module

| Question | Réponse du module |
| --- | --- |
| Est-ce que le microphone fournit des données ? | `stream_status` |
| Est-ce que le micro est coupé, si l'état est accessible ? | `mute_status` (`UNKNOWN` sinon, jamais inventé) |
| Est-ce que quelqu'un parle ? | VAD + `SpeechSegment` |
| Le signal est-il exploitable ? | `quality_status`, RMS, clipping, bruit |
| Quand cela se passe-t-il ? | Horodatages relatifs à la session + origine UTC |

Ce que ce module **ne fait pas** : Whisper, diarisation, analyse acoustique avancée, classification textuelle, fusion / risque.

## Architecture

Six composants dans **la même application Python**, reliés par des files d'événements bornées en mémoire — pas de microservices, pas de broker réseau.

```
AudioSource → AudioNormalizer → QualityMonitor
                              → VADDetector → AudioSegmenter
                                           → AcquisitionPublisher
```

Deux flux complémentaires sont publiés :

- **flux continu** : tous les `AudioChunk`, y compris les silences (nécessaire au scénario SC11) ;
- **flux de parole** : intervalles VAD avec une marge temporelle, pour la future transcription.

Format interne : **16 kHz, mono, float32, amplitude nominale [-1, +1]**. Un fichier stéréo est ramené en mono par **moyenne des canaux**. Ce downmix est adapté à un mixage déjà stéréo, pas à deux micros distincts que la diarisation voudrait conserver.

## Installation

Python 3.11 ou plus récent (validé sur 3.12).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Les données locales autorisées se placent dans `data/local_dev/` (ignoré par Git). Aucun enregistrement personnel ne doit être commité.

Les extraits de **conversations françaises réelles** (réunions multi-locuteurs SUMM-RE, dialogues téléphoniques Accueil_UBS, licences CC BY-SA) se recréent hors Git :

```powershell
python scripts\fetch_french_conversations.py
```

Le manifeste et les annotations sont dans `data/conversations_fr/`. Aucun corpus anglais n'est inclus : priorité MVP contractuelle. Les mesures VAD et le premier WER sont dans `docs/pilier1_mesure.md`.

Transcription locale (modèle `small`, CPU, INT8) :

```powershell
python scripts\test_whisper.py --model .cache\faster-whisper-small
```

## Utilisation

Le moteur fonctionne **sans interface graphique**.

```powershell
# Fichier WAV
python -m safeai.acquisition inspect --file data/local_dev\test.wav

# Périphériques
python -m safeai.acquisition devices

# Microphone (consentement obligatoire)
python -m safeai.acquisition inspect --device 0 --consent --duration 10

# Reecoute locale des SpeechSegment (validation d'alignement)
python -m safeai.acquisition export-segments --file data/local_dev\prise.wav --out data/local_dev\segments

# Scénario synthétique de diagnostic
python -m safeai.acquisition demo

# Console de diagnostic locale
python -m safeai.acquisition ui
```

L'interface s'ouvre sur `http://127.0.0.1:8765`. Elle permet de charger un fichier, de lancer une démonstration synthétique, ou de capturer un microphone après consentement. Le moteur derrière cette console est le même que celui que Whisper et la diarisation réutiliseront.

Le drapeau `recording_consent` est une commande interne. Il **ne remplace pas** la preuve d'accord écrit prévue par le guide.

## Interface commune (analyses textuelle et vocale)

Échantillon fictif pour démarrer sans modèles : `data/interface_commune/session_demo.json` (`provenance: synthetic_fixture`). Silence légitime : `session_empty.json`. Mode d'emploi : `docs/interface_commune.md`.

Pipeline hors ligne (orchestrateur injectable) — produit `AlignedSession` **et** `*_diarization.json` :

```powershell
# Environnement avec Community-1 + Whisper
.\.venv-diarization\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m safeai.pipeline inspect `
  --file data/local_dev\r06_deux_locuteurs.wav `
  --out data/local_dev\common_r06.json `
  --whisper-model .cache\faster-whisper-small `
  --session-id session_common_r06
```

Community-1 nécessite `.venv-diarization` (ou `pyannote` installé) et un accès Hugging Face local déjà autorisé. Aucun jeton ne doit figurer dans Git (`.env`, caches et modèles exclus).

## Contrats de sortie

Trois sorties distinctes, sans échantillons bruts dans les journaux :

1. `AudioChunk` — bloc normalisé en mémoire
2. `SpeechSegment` — intervalle de parole + probabilité VAD moyenne
3. `AudioQualityEvent` — `stream_status`, `mute_status`, `speech_activity`, `quality_status` et métriques

API du service :

```python
from safeai.acquisition import AcquisitionService, SourceConfig

service = AcquisitionService()
service.start(SourceConfig(source_type="file", file_path="data/local_dev/test.wav"))
audio_q = service.subscribe_audio()
speech_q = service.subscribe_speech_segments()
quality_q = service.subscribe_quality_events()
status = service.get_status()
service.stop()
```

## Tests

```powershell
$env:PYTHONPATH = "src"
# Suite complète — environnement de développement
.\.venv\Scripts\python.exe -m pytest tests -q

# Même suite dans l'environnement de recette Community-1
.\.venv-diarization\Scripts\python.exe -m pytest tests -q
```

Recette locale V1.0 (10 novembre 2026) : **174** tests réussis dans `.venv` et dans `.venv-diarization` (0 ignoré sur cette machine). Un environnement d'audit sans Silero peut ignorer ou faire échouer le test `LocalAcquisition` réel ; ce n'est pas un défaut du pipeline simulé.

Couverture principale : acquisition (16 kHz, silence, clipping, SC11/SC22…), transcription, diarisation, alignement, cohérence acoustique COMMON-016…020.

Le VAD de production est **Silero** (ONNX, inférence locale). Si `vad.backend: silero` est demandé et que le modèle est indisponible, l'application **échoue explicitement**. Le détecteur énergétique n'est utilisé que lorsque `vad.backend: energy` ou `--energy-vad` est demandé.

`mute_status` reste `UNKNOWN` tant que Windows n'expose pas un état de coupe fiable via sounddevice. Un silence n'est jamais interprété comme une confirmation de micro coupé.

## Confidentialité

- Aucun flux brut n'est envoyé vers un service distant.
- Les échantillons ne sont pas sérialisés dans les logs ni dans la sortie JSON.
- `data/local_dev/` et les fichiers audio sont exclus de Git.
- L'interface de diagnostic reste sur `127.0.0.1`.

## Configuration

Voir `configs/acquisition.yaml` : durée des fenêtres (1–2 s), seuil VAD, marges, files bornées, seuils de qualité. Ces valeurs sont une **calibration**, pas un entraînement.
