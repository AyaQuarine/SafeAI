# R07 — Capture continue, 8 novembre 2026

Microphone 1, 300 s demandées. Une seule personne. Le planning affiché est arrivé après le début du fichier : la première minute a donc tourné avant les consignes. Aucune modification du code d'acquisition.

Fichier : `data/local_dev/r07_longue.wav`

4 792 942 échantillons, 299,559 s, 16 kHz mono. SHA-256 `a96b683b8c68c18bb6a109fc9e5583c8d92d0dbb2868162224fbb7d46eccddb3`. Dernier bloc : 8 942 échantillons, soit 0,559 s.

## Console

```
Duration           : 299.56 s
Chunks published   : 300
Chunks retained    : 180
Timestamp errors   : 0
Speech alignment   : 0
Signal status      : ACTIVE
VAD backend        : silero
Speech segments    : 45
Publisher drops    : 0
```

Aucun incident. Arrêt normal, code de sortie 0. Le WAV est lisible et contient tous les échantillons : 300 blocs publiés, pas 180. Les 180 blocs retenus sont la fenêtre du rapport en direct (plafond fixé à 180 pour le microphone). Les événements de qualité plus anciens sortent de cette fenêtre. Les 45 segments de parole, moins nombreux, y restent tous. D'où les lignes `UNKNOWN / insufficient_information` entre 0 et 54 s : la parole de ce début est bien listée, son étiquette de qualité ne l'est plus. Ce n'est pas une coupure, et `chunks_retained` inférieur à `chunks_published` n'est pas une perte.

## Parole et silence

| Période du fichier | Silero | Niveau |
| --- | --- | --- |
| 0–54 s | Parole entrecoupée, dès 14,1 s | RMS 0,008–0,012, crête 0,16 |
| 54–161 s | Un seul silence | 135–160 s au plancher (RMS environ 0,002). 110–135 s plus fort, crête 0,20 à 120–125 s, quand même classé silence |
| 161–223 s | Parole en plusieurs segments | RMS 0,008–0,014, crête 0,13 |
| 223–255 s | Silence | RMS 0,004–0,010 |
| 255–298 s | Parole et silence alternés | RMS jusqu'à 0,011, crête 0,11 |

Clipping 0 sur toutes les fenêtres de 20 s. Crête du fichier 0,20.

Le silence demandé vers 60–120 s est bien sans segment de parole dans le rapport en direct. La parole reprend à 161 s. La dernière minute alterne parole et silence, comme demandé.

## Passage 124–132 s

L'extrait est `data/local_dev/r07_124_132.wav`. Le rapport en direct le laisse dans le silence 54,4–161 s.

Le rejeu des 300 s, depuis le début, avec les mêmes blocs que la capture (299 blocs de 16 000 échantillons, puis le reste de 8 942), produit les mêmes 45 segments. Aucune remise à zéro de Silero : les horodatages s'enchaînent. Dans cette passe continue, la probabilité entre 124 et 132 s ne dépasse pas 0,27, et aucune fenêtre n'atteint 0,5. Le direct et ce rejeu intégral disent donc la même chose.

La probabilité 0,99 venait d'un rejeu commencé au milieu, ou de l'extrait seul : Silero partait alors sans l'état des 124 premières secondes. Une remise à zéro volontaire de cet état à 124 s, sur les mêmes échantillons, fait remonter la probabilité jusqu'à 0,995, avec des fenêtres au-dessus de 0,5 de 125 à 131 s. Le découpage des blocs n'explique pas l'écart. C'est l'état interne emporté depuis le début de la session.

Ce même état détecte à nouveau de la parole à partir de 161 s. Il n'est pas bloqué pour le reste de la prise.

Whisper small, sur l'extrait seul, rend trois phrases : « J'ai dit, écoute-moi. », « Ok, je vais pleurer maintenant. », « Et puis j'espère que je vais finir cette assemblée. » C'est une transcription automatique, pas une confirmation d'oreille. Le passage contient donc de la parole que la session continue a laissée sous le seuil 0,5. Le seuil n'a pas été modifié.

## Mémoire du processus Python

Relevé sur l'interpréteur qui porte Silero, pas sur le petit lanceur du venv (environ 4 Mo). De 79 s à 295 s : mémoire résidente 300,2 Mo puis 312,1 Mo, mémoire privée 744,6 Mo puis 756,2 Mo. La hausse d'environ 12 Mo accompagne le remplissage de l'historique de 180 blocs, puis le niveau se stabilise autour de 311 Mo. Pas de croissance continue jusqu'à la fin de la prise. Les 79 premières secondes n'ont pas été échantillonnées.

## Décision

**Accepté, réserve.** La session de cinq minutes reste entière : durée proche de 300 s, dernier bloc conservé, horodatages à 0, pertes à 0, signal ACTIVE, arrêt propre, mémoire bornée. Le direct et le rejeu intégral concordent. La réserve porte sur 124–132 s : de la parole y est audible pour un modèle froid et pour Whisper, mais l'état Silero emporté depuis le début la garde sous 0,5. Le seuil n'a pas été changé.
