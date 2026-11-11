# R06 — Reprise, 8 novembre 2026

Une seule personne devant le microphone. La deuxième voix est une vidéo, haut-parleur rapproché du micro. La prise précédente, où la vidéo n'entrait pas, est `data/local_dev/r06_deux_locuteurs_prise1.wav`. Aucune modification du code. Aucune identité attribuée.

Fichier de cette reprise : `data/local_dev/r06_deux_locuteurs.wav`

637 191 échantillons, 39,824 s, 16 kHz mono. SHA-256 `3a670e8fba553405ba8c74510a0d5d4b13813f133556009adfc4e529d1c9cab9`. Dernier bloc : 13 191 échantillons, soit 0,824 s.

## Sortie du programme

```
Duration           : 39.82 s
Chunks published   : 40
Chunks retained    : 40
Timestamp errors   : 0
Speech alignment   : 0
Signal status      : ACTIVE
Speech segments    : 7
Publisher drops    : 0

 0.0-5.8    Silence    OK
 5.8-9.8    Parole     OK
 9.8-13.1   Silence    OK
13.1-13.7   Parole     OK
13.7-14.3   Silence    OK
14.3-15.0   Parole     OK
15.0-16.3   Silence    OK
16.3-18.9   Parole     OK
18.9-22.4   Silence    OK
22.4-25.0   Parole     OK
25.0-25.4   Silence    OK
25.4-29.2   Parole     DEGRADED    background_noise
29.2-29.8   Silence    OK
29.8-36.5   Parole     DEGRADED    background_noise
36.5-39.8   Silence    DEGRADED    background_noise
```

Aucun incident. Horodatage et pertes à 0.

| Moment demandé | Ce que SafeAI a écrit | Whisper sur cet extrait |
| --- | --- | --- |
| Toi seule, 5–10 s | Parole 5,83–9,85 s | « Bonjour, je suis la personne A du Test Safe IA. » |
| Vidéo seule, 12–17 s | Trois paroles : 13,07–13,72 s, 14,31–14,97 s, 16,27–18,90 s | « Pour savoir avant de venir à la réunion, en 1, si vous n'aimez pas les fortes chaleurs, venez pendant la... » |
| Toi seule, 20–24 s | Parole 22,41–25,01 s | « Alors une fraise goutte » |
| Vidéo seule, 24–28 s | Parole 25,42–29,21 s, qualité `DEGRADED` | « C'est un de l'univers austral, c'est de mai à octobre et là... » |
| Toi et la vidéo, 28–35 s | Une seule parole 29,77–36,50 s, qualité `DEGRADED`, crête 0,46 | « Et voilà, nous sommes là, on est de Paris, on est de l'Ontario et on est de l'Ontario. » |

La vidéo est cette fois dans le fichier : son niveau dépasse celui de la voix seule, et Silero la marque comme de la parole. Le passage ensemble reste un seul segment. Le motif `DEGRADED / background_noise` vient du niveau, au-dessus de 0,04 avec un signal rugueux. Le silence de fin hérite de cette estimation. Ce n'est pas une coupure, et ce n'est pas un nom de locuteur.

## Décision

**Accepté, réserve.** L'écoute du segment `data/local_dev/r06_segments/segment_006_29.77-36.50.wav` (29,77–36,50 s) confirme que deux voix distinctes sont audibles en même temps. SafeAI n'en fait qu'un seul `SpeechSegment` : Silero répond à « y a-t-il de la parole », pas à « qui parle ». Aucune identité n'est attribuée. La transcription Whisper mélangée n'est pas un défaut de l'acquisition.

Réserve : la deuxième voix est une vidéo diffusée par un haut-parleur, pas une deuxième personne dans la pièce. Cette prise n'est pas encore un test de réunion naturelle. La prise 1, où cette vidéo n'entrait pas dans le micro, reste hors décision.
