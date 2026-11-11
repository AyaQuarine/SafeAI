# R01 officiel — 6 novembre 2026

Prise avec le code qui conserve le dernier bloc partiel. Micro 1, Silero, phrase « Bonjour, ceci est un test du microphone SafeAI. » Consigne affichée : silence 0–3 s, phrase à partir de la 4e seconde, silence 8–15 s. Pas de diffusion par les haut-parleurs.

Fichiers, hors git :

- `data/local_dev/r01_officiel.wav`
- `data/local_dev/r01_officiel_segments/segment_000_4.84-8.44.wav`

## Console

```
Source             : microphone:1
Input rate         : 44100 Hz
Output rate        : 16000 Hz
Channels           : 4 -> 1 (average)
Duration           : 14.76 s
Chunks published   : 15
Chunks retained    : 15
Timestamp errors   : 0
Speech alignment   : 0
Signal status      : ACTIVE
VAD backend        : silero
Speech segments    : 1
Publisher drops    : 0

 0.0-4.8    Silence
 4.8-8.4    Parole
 8.4-14.8   Silence
```

Aucun incident affiché.

## Durée et dernier bloc

236 165 échantillons, 16 kHz, mono, crête 0,094. Durée 14,760 s.

Les quatorze premières secondes sont des blocs de 16 000 échantillons. Le dernier bloc en contient 12 165, soit 0,760 s. Il est dans le WAV. La prise de 15 s faite avant la correction s'arrêtait à 14,00 s pile. L'écart entre 14,76 s et 15 s est le décalage entre l'horloge demandée et les échantillons réellement arrivés avant l'arrêt, pas une seconde entière abandonnée.

## Niveau et segment

| Seconde | Échantillons | RMS | Crête |
| --- | --- | --- | --- |
| 0–4 | 16 000 | ≤ 0,0006 | ≤ 0,003 |
| 4–5 | 16 000 | 0,0010 | 0,019 |
| 5–8 | 16 000 | 0,011 à 0,017 | 0,049 à 0,094 |
| 8–9 | 16 000 | 0,0045 | 0,036 |
| 9–14,76 | 16 000, puis 12 165 | ≤ 0,0001 | ≤ 0,0004 |

Un segment : 4,842–8,438 s, 3,596 s, alignement 0. Il commence après la 4e seconde, comme demandé, et dépasse 8 s de 0,44 s. L'énergie de cette fin est encore au-dessus du silence, puis le plancher reprend dès 9 s. Pas de seconde détection.

## Phrase

Whisper small, sur le segment seul : « Bonjour, ceci est un test du Microphone Safe Ear. » La phrase est reconnaissable. « SafeAI » sort en « Safe Ear ». Transcription automatique, pas une écoute. Le WAV complet et le segment ont été ouverts dans le lecteur.

## Décision

| Critère | Observé |
| --- | --- |
| Format | 16 kHz, mono |
| Durée | 14,76 s, dernier bloc de 0,76 s conservé |
| Silero | 4,84–8,44 s, une seule plage |
| Silence | Aucune parole détectée hors de cette plage |
| Horodatages | 0 |
| Speech alignment | 0 |
| Publisher drops | 0 |
| Incidents | Aucun |
| Intelligibilité à l'oreille | Non confirmée dans cette session |

**Accepté, avec écart mineur.** Écoute du 6 novembre 2026 : la phrase « Bonjour, ceci est un test du microphone Safe IA. » est complète et intelligible dans le WAV et dans le segment, sans syllabe coupée. Le segment correspond à son intervalle dans le WAV. La fin à 8,44 s est la fin réelle de la phrase, pas une erreur de Silero. La prononciation de « SafeAI » n'est pas un défaut d'acquisition. Aucune modification de code.
