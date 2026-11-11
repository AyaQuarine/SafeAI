# R03 — Reprise, 6 novembre 2026

Deuxième prise, microphone 1, 30 s. Les deux WAV locaux sont distincts. Aucune modification du code.

| Fichier | Échantillons | Durée | SHA-256 | Écrit |
| --- | --- | --- | --- | --- |
| `data/local_dev/r03_bruit.wav` | 476 451 | 29,778 s | `06cf4e05d6ba26e95b5ae45482f5d7bdf7f19258c46feeae7ecd572ec26bc513` | 14:30 |
| `data/local_dev/r03_bruit_prise1.wav` | 476 863 | 29,804 s | `7a5bda447d6b289c9b7ec7de55508de28949bc20bb6a7c1b957f91b4cf84d90c` | 14:25 |

La reprise est `r03_bruit.wav`. La première prise, sans bruit mesurable, est `r03_bruit_prise1.wav`.

Segments de la reprise, hors git :

- `data/local_dev/r03_segments/segment_000_5.87-9.33.wav`
- `data/local_dev/r03_segments/segment_001_16.49-20.12.wav`

Repères prévus : silence 0–5 s, phrase au calme 5–10 s, ventilateur 10–15 s, même phrase avec ventilateur 15–20 s, ventilateur seul 20–25 s, silence 25–30 s.

## Console

```
Duration           : 29.78 s
Chunks published   : 30
Chunks retained    : 30
Timestamp errors   : 0
Speech alignment   : 0
Signal status      : ACTIVE
VAD backend        : silero
Speech segments    : 2
Publisher drops    : 0

 0.0-5.9    Silence    OK    silence
 5.9-9.3    Parole     OK    silence
 9.3-16.5   Silence    OK    ok
16.5-20.1  Parole     OK    ok
20.1-29.8  Silence    OK    ok
```

Aucun incident. 476 451 échantillons, 16 kHz mono. Dernier bloc : 12 451 échantillons, soit 0,778 s.

## Niveau

| Période observée | RMS | Lecture |
| --- | --- | --- |
| 0–6 s | ≤ 0,0006 | silence initial |
| 6–9 s | 0,015 à 0,021 | phrase au calme, crête 0,12 |
| 11–16 s | 0,007 à 0,029 | ventilateur, crête 0,12, classé silence |
| 16–20 s | 0,023 à 0,030 | phrase avec bruit, crête 0,24 |
| 21–26 s | environ 0,004 | bruit résiduel, classé silence |
| 27–29,8 s | ≤ 0,0001 | retour au silence |

L'estimation de bruit du `QualityMonitor` monte de 0,0004 à 0,0138. Le seuil `noise_degraded` est 0,04 : il n'est pas franchi. Les 30 événements restent `OK`, clipping 0. Le ventilateur est donc visible dans les mesures, sans atteindre le statut `DEGRADED`.

Silero ne crée aucun segment pendant le ventilateur seul. Les deux paroles durent 3,47 s et 3,63 s.

Whisper small :

- 5,87–9,33 s : « Bonjour, ceci est un test de Microphone Safe IA. »
- 16,49–20,12 s : « pour utiliser d'un test de microphone safety. »

La phrase calme est reconnaissable. La phrase bruitée est plus déformée par la transcription. Le segment commence au saut d'énergie, et sa durée est celle d'une phrase complète.

## Décision

| Contrôle | Observé |
| --- | --- |
| Deux paroles | 5,87–9,33 s et 16,49–20,12 s |
| Ventilateur seul | Aucun segment vocal |
| Qualité | Bruit mesuré, sous le seuil 0,04, statut OK |
| Horodatages, alignement, pertes | 0 |
| Dernier bloc | 0,78 s conservé |
| Incidents | Aucun |

**Accepté avec réserve.** La parole reste détectée, le bruit n'est pas pris pour de la voix, et `DEGRADED` n'avait pas à s'afficher. La réserve porte sur le seuil 0,04, qui n'a pas été franchi, et sur la phrase bruitée, moins nette pour Whisper. R04 n'est pas lancé.
