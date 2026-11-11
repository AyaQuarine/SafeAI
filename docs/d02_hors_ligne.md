# D2 — Première diarisation hors ligne, 9 novembre 2026

Les contrats D1 ont été ajustés, puis l'adaptateur Community-1 a été branché. Les erreurs de chargement, d'inférence et de fichier sont distinctes. Un WAV 48 kHz stéréo est ramené à 16 kHz mono avant l'inférence. La révision `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` est transmise à `Pipeline.from_pretrained()`. Aucun jeton Hugging Face n'a été écrit dans le dépôt. Le 9 novembre, le login Hugging Face a été fait localement (compte `Ayaqr1`, jeton OAuth dans le cache utilisateur, hors dépôt). Les conditions Community-1 ont été acceptées. R01, R02 et R06 ont produit des inférences réelles. Ce n'est pas une acceptation D2 signée, ni un DER.

## Contrats, après correction

`duration_s` est la durée du fichier traité. `offset_s` est le début de cet extrait dans la session. `start_s` et `end_s` sont des temps de session. Un fichier de 10 s commençant à 120 s accepte un tour 122–125 s.

`overlap` est vrai seulement lorsque deux identifiants distincts et connus se recouvrent. Deux tours `speaker_00` qui s'intersectent, ou un tour `unknown` qui s'intersecte avec une voix connue, ne sont pas un chevauchement vocal confirmé.

## Environnement isolé

Les 132 tests du projet principal restent dans `.venv`, sans `pyannote`. Community-1 est installé dans `.venv-diarization/` (gitignored).

| Composant | Version |
| --- | --- |
| Python | 3.12.0 |
| torch | 2.15.0+cpu, CUDA absent |
| pyannote.audio | 4.0.7 |
| huggingface_hub | 2.4.0 |
| FFmpeg | absent du PATH |

L'audio SafeAI est lu par `soundfile` puis passé en mémoire. FFmpeg n'est pas requis pour ces WAV 16 kHz.

`num_speakers` n'est jamais imposé. Les bornes `min_speakers` / `max_speakers` restent vides. La sortie utilisée est `speaker_diarization`, pas la variante exclusive.

## Authentification Hugging Face

Login OAuth appareil dans `.venv-diarization` : compte `Ayaqr1`. Le jeton est dans le cache Hugging Face local, pas dans Git, pas dans `HF_TOKEN` de session, pas dans ce rapport. Les conditions d'accès de `pyannote/speaker-diarization-community-1` (CC-BY-4.0, gated auto) ont été acceptées. `list_repo_files` a renvoyé 10 fichiers, dont `embedding/pytorch_model.bin`.

## R01 — inférence réelle

`python -m safeai.diarization inspect --file data/local_dev/r01_officiel.wav`, dans `.venv-diarization`, code de sortie 0.

| Champ | Valeur |
| --- | --- |
| Durée | 14.76 s |
| Statut | completed |
| Locuteurs estimés | 1 |
| Tours | 1, `speaker_00` 4.92–8.23 s, overlap faux |
| Traitement | 43.08 s, RTF 2.92 |
| Modèle | `pyannote/speaker-diarization-community-1` révision `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` |
| Sorties | `data/local_dev/d02_r01.json`, `data/local_dev/d02_r01.rttm` (pas d'audio brut) |

Ce tour est compatible avec la parole entendue vers 4.8–8.4 s.

## R02 — silence, aucun locuteur inventé

`inspect --file data/local_dev/r02_silence.wav`, code de sortie 0.

| Champ | Valeur |
| --- | --- |
| Durée | 19.53 s |
| Statut | empty |
| Locuteurs estimés | 0 |
| Tours | 0 |
| Traitement | 11.03 s, RTF 0.57 |
| Sorties | `data/local_dev/d02_r02.json`, `data/local_dev/d02_r02.rttm` |

Community-1 n'a pas fabriqué de `speaker_00` sur le silence.

## R06 — deux sources

`inspect --file data/local_dev/r06_deux_locuteurs.wav`, code de sortie 0.

| Champ | Valeur |
| --- | --- |
| Durée | 39.82 s |
| Statut | completed |
| Locuteurs estimés | 2 |
| Tours | 9 |
| Chevauchements | 2 (`speaker_00` 30.07–34.73 et `speaker_01` 32.72–33.04) |
| Traitement | 55.47 s, RTF 1.39 |
| Sorties | `data/local_dev/d02_r06.json`, `data/local_dev/d02_r06.rttm` |

Les deux identifiants sont `speaker_00` et `speaker_01`. L'overlap détecté tombe dans la fenêtre 29.77–36.50 s déjà confirmée à l'oreille sur R06. Ce n'est pas un DER. D3/D4 ne commencent pas ici.
