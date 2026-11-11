# Diagnostic vocal — microphone Intel, 6 novembre 2026

Prise directe devant le microphone, sans lecture par les haut-parleurs. Le décompte 0–4 s / 4–10 s / 10–18 s était affiché à l’écran pendant la capture SafeAI.

Ce document ne prononce pas l’acceptation du cahier des charges n° 01. R02–R08 restent non réalisés.

## Étape A — Environnement

```
python -m pytest -q
```

**95 passed** en 21,6 s. Aucun échec.

```
python -m safeai.acquisition devices
```

L’identifiant **1** est toujours le réseau Intel :

| Id | Entrée |
| --- | --- |
| 1 | Microphone Array (Intel Smart Sound), 4 canaux, 44 100 Hz |

Le consentement microphone Windows de la session est `Allow`.

## Étape B — Microphone, hors SafeAI

La prise indépendante n’a pas été faite dans l’application Enregistreur de sons. Elle a été faite par PortAudio sur le même identifiant 1, 8 s, 44 100 Hz, 4 canaux, fichier `data/local_dev/r01_micro_windows.wav`.

| Seconde | Canal 0–1, crête | Canaux 2 et 3 | Moyenne des 4 canaux, crête |
| --- | --- | --- | --- |
| 0–5 | ≤ 0,009 | ~0 | ≤ 0,004 |
| 5–6 | 0,32 | ~0 | 0,16 |
| 6–7 | 0,27 | ~0 | 0,13 |
| 7–8 | 0,14 | ~0 | 0,07 |

Les canaux 0 et 1 portent le signal, et ils sont pratiquement identiques. Les canaux 2 et 3 restent au silence numérique. La moyenne des quatre canaux divise donc l’amplitude utile par environ deux. Une crête de 0,32 sur les canaux actifs est un niveau de voix normale, sans amplification extrême. L’intelligibilité des mots n’a pas été jugée à l’oreille dans cette session.

## Étape C — Capture SafeAI

```
python -m safeai.acquisition inspect --device 1 --consent --duration 18 --record data/local_dev/r01_voix_directe.wav
```

```
Source             : microphone:1
Input rate         : 44100 Hz
Output rate        : 16000 Hz
Channels           : 4 -> 1 (average)
Duration           : 17.00 s
Chunks published   : 17
Chunks retained    : 17
Timestamp errors   : 0
Speech alignment   : 0
Signal status      : ACTIVE
VAD backend        : silero
Speech segments    : 2
Publisher drops    : 0

 0.0-3.5    Silence    OK           silence
 3.5-11.1   Parole     OK           ok
11.1-14.8   Silence    OK           ok
14.8-15.7   Parole     OK           silence
15.7-17.0   Silence    OK           ok
Enregistrement      : data\local_dev\r01_voix_directe.wav
```

Le fichier fait 17,00 s et 272 000 échantillons à 16 kHz, crête 0,241. La seconde demandée en plus n’est pas dans le WAV : le bloc d’une seconde encore ouvert à l’arrêt n’est pas émis. Les 17 blocs publiés sont tous conservés. Aucune perte, aucune erreur d’horodatage, signal `ACTIVE`.

Niveau par seconde sur le WAV déjà ramené en mono :

| Seconde | RMS | Crête | Lecture |
| --- | --- | --- | --- |
| 0–3 | ≤ 0,0007 | ≤ 0,004 | silence |
| 3–11 | 0,016 à 0,052 | 0,08 à 0,24 | voix |
| 11–14 | ≤ 0,0002 | ≤ 0,005 | silence |
| 14–16 | 0,006 puis 0,017 | 0,07 | son court |
| 16–17 | 0,0001 | 0,0003 | silence |

## Étape D — Segments

Le WAV a été ouvert avec le lecteur par défaut. Segments exportés dans `data/local_dev/r01_voix_segments` :

| Fichier | Début | Fin | Durée |
| --- | --- | --- | --- |
| `segment_000_3.47-11.13.wav` | 3,466 s | 11,126 s | 7,660 s |
| `segment_001_14.83-15.73.wav` | 14,826 s | 15,734 s | 0,908 s |

Silero, alignement 0. Le segment principal encadre la fenêtre 4–10 s : il commence 0,5 s avant et se termine 1,1 s après. L’énergie confirme que ce segment porte la voix, et que le second segment correspond à un son réel pendant le silence demandé, pas à un faux positif sur le bruit de fond.

## Décision de ce diagnostic

| Point | Résultat |
| --- | --- |
| Tests | 95 réussis |
| Identifiant du micro Intel | 1, inchangé |
| Voix recueillie hors SafeAI | Oui, canaux 0 et 1, crête 0,32 |
| Parole SafeAI autour de 4–10 s | Oui, 3,47–11,13 s |
| Horodatage et pertes | 0 erreur, 0 perte |
| Silence final | Un son d’environ 0,9 s vers 14,8 s |
| Durée enregistrée | 17,00 s pour 18 s demandées |

**À analyser.** La détection de la voix directe est au rendez-vous, avec une marge de VAD compatible avec la fenêtre demandée. Deux réserves empêchent de cocher le scénario comme accepté : le second segment pendant le silence de fin, et la seconde manquante en queue de fichier. Les mots de la phrase restent à confirmer à l’écoute de `data/local_dev/r01_voix_directe.wav` et de `segment_000_3.47-11.13.wav`.
