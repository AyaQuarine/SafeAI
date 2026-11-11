# R02 — Silence, 6 novembre 2026

Microphone 1 ouvert, 20 s demandées, personne ne parle. Aucune modification du code d'acquisition. Fichier hors git : `data/local_dev/r02_silence.wav`.

## Console

```
Source             : microphone:1
Input rate         : 44100 Hz
Output rate        : 16000 Hz
Channels           : 4 -> 1 (average)
Duration           : 19.53 s
Chunks published   : 20
Chunks retained    : 20
Timestamp errors   : 0
Speech alignment   : 0
Signal status      : ACTIVE
VAD backend        : silero
Speech segments    : 0
Publisher drops    : 0

 0.0-19.5   Silence    OK    silence
```

Aucun incident. Le rapport console n'affiche pas le mute. La sonde utilisée sur ce microphone retourne `UNKNOWN`, parce que l'état mute Windows n'est pas observable.

## Durée et niveau

312 413 échantillons, 16 kHz, mono. Durée 19,526 s. Dix-neuf blocs de 16 000 échantillons, puis un dernier bloc de 8 413 échantillons, soit 0,526 s. Il est dans le WAV.

Crête sur toute la prise : 0,0089. Le plancher reste autour de 0,0001 RMS. Quelques secondes montent un peu (crête 0,006 à 0,009 vers 7–8 s, 11–12 s et 15–16 s), très en dessous de la voix du R01, dont la crête était 0,094. Silero ne produit aucun segment. Il n'y a donc pas d'événement à qualifier de faux positif.

## Décision

| Indicateur | Observé |
| --- | --- |
| Speech segments | 0 |
| Timestamp errors | 0 |
| Speech alignment | 0 |
| Publisher drops | 0 |
| Signal status | ACTIVE |
| Mute | UNKNOWN, non affiché par la console |
| Durée | 19,53 s, dernier bloc conservé |
| Contenu | Bruit de fond faible, sans parole |

**Accepté.** Le silence réel n'est pas converti en parole ni en coupure. R03 n'est pas lancé.
