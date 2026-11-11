# Recette — Pilier 1 Acquisition

Date : 2026-11-01  
Configuration : `configs/acquisition.yaml` (16 kHz, mono, fenêtres 1 s, VAD Silero seuil 0,5)

SafeAI reste un système pédagogique, non médical, non certifié.

## 1. Corrections P0 / P1

| ID | Correction | Statut |
| --- | --- | --- |
| P0 | Tampon horodaté : un `SpeechSegment` ouvert et fermé dans le même bloc conserve ses échantillons | Fait |
| P0 | `samples[0]` correspond à `start_s` dans la source ; `end_s - start_s = len(samples) / 16000` | Fait |
| P0 | `SpeechSegment.validate()` contrôle durée, mono, float32, finitude | Fait |
| P1 | `speech_present_in_window` ≠ `speech_active_at_end` | Fait |
| P1 | `backend=silero` échoue explicitement s'il est indisponible | Fait |
| P1 | Pertes `Publisher` dans `AcquisitionStatus`, incidents et console | Fait |
| P1 | Trou d'une seconde : perte explicite, pas des zéros relus comme silence | Fait |
| P1 | Silence en fin de fenêtre de parole compté dans `silence_duration_s` | Fait |

## 2. Tests automatiques

```
python -m pytest -q
```

Résultat du 9 novembre 2026 : **132 passed** dans `.venv`. Community-1 reste isolé dans `.venv-diarization`. Aucun seuil d'acquisition n'a été baissé. Aucun jeton Hugging Face n'est stocké.

Les 8 tests de `tests/test_acquisition_qa.py` passent sans affaiblissement des assertions.

## 3. Silero — vérification d'exécution

```
python -c "from safeai.acquisition.vad import build_vad; print(build_vad().name)"
```

Résultat : `silero`

`inspect` sans `--energy-vad` affiche `VAD backend : silero`.

`inspect_demo.wav` est un signal **synthétique**. Silero n'y détecte pas de parole (0 segment) ; le VAD énergétique en détecte une. C'est le comportement attendu : ce fichier ne doit pas servir à évaluer Silero.

Comparaison utile déjà disponible : extraits **français multi-locuteurs** dans `data/conversations_fr/` (SUMM-RE + Accueil_UBS, CC BY-SA). Relancer `python scripts/fetch_french_conversations.py` si l'audio est absent. Sur `fr_reunion_mix_00_30s.wav`, Silero produit 5 segments, horodatage 0 erreur. Les prises micro personnelles R01–R08 restent nécessaires pour la recette salle.

## 4. Microphone et limites matérielles

`sounddevice` liste les périphériques Windows, mais n'expose pas un état mute fiable. `default_mute_probe` retourne donc `UNKNOWN`.

| Réalité | Prédiction actuelle | Lecture |
| --- | --- | --- |
| Silence, micro ouvert | `NON_SPEECH` + `mute_status=UNKNOWN` | Correct, pas de coupure inventée |
| Micro coupé, état non observable | `UNKNOWN` | Limite matérielle, à discuter avec le PO |
| Micro coupé, état injecté (test) | `MUTED` | Couvert par les tests T06 / SC14 |

Une absence de parole n'est jamais convertie en coupure matérielle.

## 5. Phase B — essais réels (à faire avec consentement)

Les enregistrements personnels restent dans `data/local_dev/` (Git ignore).

Commandes, avec enregistrement explicite. `inspect --device` seul n'écrit pas de WAV.

```powershell
python -m safeai.acquisition devices
python -m safeai.acquisition inspect --device <id> --consent --duration 15 --record data/local_dev/r01.wav
python -m safeai.acquisition export-segments --file data/local_dev/r01.wav --out data/local_dev/segments
```

`--record` écrit le flux 16 kHz au fur et à mesure, seulement après que le microphone a démarré. Si le périphérique refuse l'accès, aucun WAV n'est créé. Le fichier reste dans `data/local_dev/`, hors git. Le consentement des personnes présentes est obligatoire.

Le rapport distingue `chunks_published` (toute la session) et `chunks_retained` (historique borné en mémoire). `timestamp_errors` cumule les anomalies au fil du traitement, y compris celles qui ont quitté l'historique.

À l'arrêt, le tampon microphone est vidé : les blocs complets encore en file et le reste plus court qu'une seconde sont publiés et écrits, sans ajout d'échantillons. Une capture simulée de 2,5 s produit un WAV de 2,5 s. Les prises R01 du 2 novembre et R01 bis du 6 novembre ont été faites avant cette correction : elles s'arrêtent sur la dernière seconde incomplète.

Le diagnostic R01 bis, phrase courte, est dans `docs/r01_bis.md`. Il ne remplace pas le R01 contractuel.

| Scénario | Attendu | Observé | Décision |
| --- | --- | --- | --- |
| R01 Parole claire, 0–3 s silence, 3–8 s phrase, 8–15 s silence | Son présent, horodatages alignés sur la phrase | 2026-11-06, micro 1. 14,76 s, 15 blocs dont un reste de 0,76 s. Parole 4,84–8,44 s, phrase complète à l'écoute. Écart de 0,44 s en fin de phrase. | Accepté, écart mineur |
| R02 Silence 20 s, micro ouvert | `NON_SPEECH`, pas d'incident de coupure inventé | 2026-11-06, micro 1. 19,53 s, 20 blocs dont un reste de 0,53 s. 0 segment de parole, 0 erreur, 0 perte, signal ACTIVE. Crête 0,009. | Accepté |
| R03 Bruit de fond et parole | Qualité dégradée visible, parole encore détectée | 2026-11-06, reprise, micro 1. 29,78 s, 30 blocs, reste 0,78 s. Paroles 5,87–9,33 s et 16,49–20,12 s. Ventilateur RMS environ 0,02, classé silence. Bruit estimé 0,014, seuil 0,04 non franchi, qualité OK. | Accepté, réserve |
| R04 Voix forte | Saturation mesurée, pas un incident fictif | 2026-11-06, micro 1. 19,81 s, 20 blocs, reste 0,81 s. Paroles 4,39–7,54 s (crête 0,16) et 10,92–15,32 s (crête 0,21). Clipping 0. | Accepté, réserve |
| R05 Mute matériel Windows | Limite réelle documentée. `UNKNOWN` est acceptable si le mute n'est pas observable | 2026-11-08, micro 1. Mute Windows 10,00–20,03 s : signal à zéro, flux ACTIVE, pas d'incident. Voix 5,7–9,1 s puis 21,2–26,0 s. 29,42 s, reste 0,42 s. | Accepté, limite documentée |
| R06 Deux locuteurs, successifs puis simultanés | Segments de parole, pas d'identité inventée | 2026-11-08, reprise, micro 1. 39,82 s, 40 blocs, reste 0,82 s. Toi 5,83–9,85 s. Vidéo 13,1–18,9 s puis 25,4–29,2 s. Ensemble un seul segment 29,77–36,50 s. Écoute : deux voix en même temps. Deuxième voix = haut-parleur. | Accepté, réserve |
| R07 Capture 3 à 5 min | Pertes, mémoire et horodatages relevés | 2026-11-08, micro 1. 299,56 s, 300 blocs, 0 erreur, 0 perte. Rejeu intégral identique au direct, 45 segments. 124–132 s : probabilité continue max 0,27 ; après remise à zéro de l'état, 0,99. Whisper y lit trois phrases. | Accepté, réserve |
| R08 Micro interrompu | Incident explicite | 2026-11-08, micro 1. Endpoint Intel passé à Error à 10,06 s puis remis OK. Flux resté ACTIVE, 29,80 s, 0 incident. Parole encore captée à 24–28 s. Preuve logicielle séparée : flux injecté coupé après 2 s, incident UNAVAILABLE, pas de silence ajouté. | Non concluant |

Ces huit lignes ne sont pas une validation. Elles seront remplies après la prise, avec le résultat observé et les anomalies. Une limite matérielle documentée peut être acceptée avec réserve. Le rapport de clôture ACQ-01 à ACQ-13 attend ces preuves. Décision du pilier 1 : non prononcée.

`export-segments` écrit un WAV par `SpeechSegment` pour réécoute : l'intervalle audio doit correspondre à la phrase, pas à un décalage.

Pour un fichier annoté manuellement, `safeai.acquisition.eval.frame_metrics` calcule précision / rappel / F1 sur des frames de 32 ms.

Objectifs internes à confirmer après mesure : rappel ≥ 90 % sur prises claires ; peu de faux segments sur silences ; exactitude ≥ 90 % silence / micro coupé / bruit lorsque l'état mute est observable.

## 6. Latence mesurée (traitement local, pas bout-en-bout salle)

Le service distingue maintenant :

- `processing_latency_s` — traitement d'un bloc
- `vad_processing_s` — inférence VAD
- `capture_to_publish_s` — réception du bloc jusqu'à publication
- `publisher_drops` / `samples_dropped`

La latence capture réelle → abonné en salle n'est pas mesurée par R07. R07 vérifie la tenue d'une session de cinq minutes.

## 7. Décision

| Niveau | État |
| --- | --- |
| Architecture | Gelée hors corrections révélées par R01–R08 |
| Recette microphone R01–R08 | R01 accepté, écart mineur. R02 accepté. R03 accepté, réserve sur le seuil de bruit. R04 accepté, réserve d'intensité. R05 accepté, mute non observable. R06 accepté, réserve haut-parleur. R07 accepté, réserve sur l'état Silero (124–132 s). R08 matériel non concluant ; interruption simulée validée. |
| Clôture du cahier n° 01 | En clôture documentaire, avec réserves. L'architecture d'acquisition est gelée hors correction d'anomalie. La matrice ACQ-01 à ACQ-13 n'est pas encore signée. |
| Cahier n° 02 | Diarisation autorisée. Le VAD ne filtre pas l'audio d'entrée. |
