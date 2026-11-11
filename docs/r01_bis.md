# R01 bis — phrase courte, 6 novembre 2026

Diagnostic simplifié, pas le R01 contractuel (15 s, parole entre 3 et 8 s). Micro 1, réseau Intel. Phrase demandée : « Bonjour, ceci est un test du microphone SafeAI. » Fenêtre affichée : silence 0–4 s, parole 4–9 s, silence 9–20 s. Pas de diffusion par les haut-parleurs.

Fichiers, hors git :

- `data/local_dev/r01_bis.wav`
- `data/local_dev/r01_bis_segments/segment_000_5.16-8.63.wav`

## Console

```
Source             : microphone:1
Input rate         : 44100 Hz
Output rate        : 16000 Hz
Channels           : 4 -> 1 (average)
Duration           : 19.00 s
Chunks published   : 19
Chunks retained    : 19
Timestamp errors   : 0
Speech alignment   : 0
Signal status      : ACTIVE
VAD backend        : silero
Speech segments    : 1
Publisher drops    : 0

 0.0-5.2    Silence
 5.2-8.6    Parole
 8.6-19.0   Silence
```

Le WAV fait 19,00 s, 304 000 échantillons, 16 kHz, mono, crête 0,071. L'écart avec 20 s est la dernière seconde incomplète, non écrite par le code en place au moment de cette prise.

## Niveau

| Seconde | RMS | Crête |
| --- | --- | --- |
| 0–5 | ≤ 0,0011 | ≤ 0,009 |
| 5–9 | 0,009 à 0,014 | 0,050 à 0,071 |
| 9–19 | ≤ 0,0002 | ≤ 0,0016 |

Un seul segment exporté : 5,162–8,630 s, 3,468 s. Il est à l'intérieur de 4–9 s. Le silence après 9 s ne contient pas de seconde détection. L'énergie de 9–19 s reste au plancher.

## Phrase

Whisper small, sur le segment seul, rend : « Bonjour, ceci est un test de Microphone Safe Air. » La phrase demandée est reconnaissable. « SafeAI » est entendu « Safe Air », et « du » est rendu « de ». Ce relevé est une transcription automatique, pas une écoute humaine. Le WAV complet et le segment ont été ouverts dans le lecteur.

## Décision

| Critère | Observé |
| --- | --- |
| Voix | Présente de 5 à 9 s, phrase reconnaissable par la transcription |
| Silero | 5,16–8,63 s, dans la fenêtre 4–9 s |
| Silences | Aucune autre parole détectée |
| Horodatages | 0 |
| Pertes | 0 |
| Format | 16 kHz, mono |
| Durée | 19,00 s pour 20 s demandées, dernière seconde incomplète non écrite |

**À analyser** tant que l'écoute humaine des deux fichiers n'a pas confirmé l'intelligibilité. Rien, dans cette prise, ne demande de changer Silero ni de reconstruire l'acquisition. Le R01 contractuel n'est pas refait ici.

## Fin de capture, corrigée ensuite

Le pipeline publie maintenant, à l'arrêt, les blocs complets encore en attente et le reste plus court, avec sa durée réelle, sans compléter par des zéros. Le test simulé de 2,5 s écrit 40 000 échantillons, soit 2,5 s, en trois blocs (1 s, 1 s, 0,5 s). Suite complète : **96 passed**. Cette prise R01 bis n'inclut pas encore cette correction.
