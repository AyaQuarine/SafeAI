# R04 — Voix forte, 6 novembre 2026

Microphone 1, 20 s demandées. Phrase normale, puis la même phrase plus fort, sans crier. Aucune modification du code. Le clipping volontaire reste à faire sur un signal synthétique.

Fichier hors git : `data/local_dev/r04_voix_forte.wav`

316 947 échantillons, 19,809 s, 16 kHz mono. SHA-256 `d3979c8a51c35ed048fe9e0ec578162653f9bdf3617161e5830a50377d2a2ef3`.

Segments :

- `data/local_dev/r04_segments/segment_000_4.39-7.54.wav`
- `data/local_dev/r04_segments/segment_001_10.92-15.32.wav`

## Console

```
Duration           : 19.81 s
Chunks published   : 20
Chunks retained    : 20
Timestamp errors   : 0
Speech alignment   : 0
Signal status      : ACTIVE
VAD backend        : silero
Speech segments    : 2
Publisher drops    : 0

 0.0-4.4    Silence
 4.4-7.5    Parole
 7.5-10.9   Silence
10.9-15.3   Parole
15.3-19.8   Silence
```

Aucun incident. Dernier bloc : 12 947 échantillons, soit 0,809 s.

## Intensité

| Phrase | Intervalle | RMS | Crête | Clipping |
| --- | --- | --- | --- | --- |
| Normale | 4,39–7,54 s | 0,022 à 0,028 | 0,157 | 0 |
| Plus forte | 10,92–15,32 s | 0,022 à 0,036 | 0,207 | 0 |

La seconde crête est supérieure. Le seuil `peak_intense` est 0,85 et le clipping commence à 0,99 : les deux phrases restent `OK`, motif `ok`. Aucune saturation n'est inventée.

Whisper small :

- normale : « Bonjour, ceci est un test de Microphone Safe IA. »
- plus forte : « Bonjour, ceci est un test de microphone. »

## Décision

**Accepté, avec réserve.** Les deux phrases sont détectées, la seconde est plus forte, les compteurs sont à zéro et le dernier bloc est conservé. L'écart d'intensité reste modéré, et le cas saturé n'est pas couvert par cette voix. R05 n'est pas lancé.
