# R08 — Interruption du périphérique, 8 novembre 2026

Pas de microphone USB branché. Les casques Realtek et manette sont absents. La tentative porte sur le microphone Intel déjà utilisé, par une coupure Windows réversible de l'endpoint de capture, pas du contrôleur audio. Ce n'est pas un mute : R05 n'est pas rejoué. Aucune modification du code.

Fichier : `data/local_dev/r08_interruption.wav`

476 863 échantillons, 29,804 s, 16 kHz mono. SHA-256 `0e994bce99702dc576ca3e2091706700c31bbfd5947d2171e69e137788a38130`. Dernier bloc conservé (la durée n'est pas un nombre entier de secondes).

Endpoint visé, et seulement celui-là :

`Microphone Array (Intel® Smart Sound Technology for Digital Microphones)`

`SWD\MMDEVAPI\{0.0.1.00000000}.{EEE3C168-2D29-4613-95A5-4908887618DF}`

## Ce que Windows a fait

```
DISABLED t=10.06 status=Error
RESTORED status=OK
```

L'endpoint est passé à `Error` à 10,06 s et a été remis à `OK` à la fin. Il est de nouveau `OK` après la prise.

## Ce que SafeAI a écrit

```
Duration           : 29.80 s
Chunks published   : 30
Chunks retained    : 30
Timestamp errors   : 0
Speech alignment   : 0
Signal status      : ACTIVE
Speech segments    : 2
Publisher drops    : 0

 0.0-24.0   Silence    OK           silence
24.0-26.1   Parole     OK           ok
26.1-26.8   Silence    OK           ok
26.8-28.2   Parole     OK           ok
28.2-29.8   Silence    OK           ok
```

Aucun incident. Pas d'état `UNAVAILABLE` ni `INTERRUPTED`.

De 0 à 23 s le niveau reste au plancher du micro ouvert (RMS environ 0,0001), pas à des zéros exacts. La parole de 24 à 28 s, crête 0,10, est arrivée pendant que l'endpoint était encore en `Error`, avant la restauration. Le flux PortAudio déjà ouvert a donc continué.

## Décision

**Non concluant** pour la recette matérielle. La coupure de l'endpoint change l'état Windows sans interrompre la capture en cours. SafeAI ne pouvait pas signaler une indisponibilité qu'il n'a pas subie. Le micro Intel a été réactivé. Cette manipulation n'est pas à refaire, et le contrôleur audio Intel n'est pas à désactiver.

## Preuve logicielle, distincte

Un flux PortAudio injecté livre 2,0 s de signal, puis devient inactif. Ce n'est pas un débranchement. Le test `test_injected_stream_stop_is_an_incident_not_silence` vérifie que SafeAI :

- publie les 2 blocs déjà reçus, 32 000 échantillons, et les écrit dans le WAV ;
- s'arrête sur `UNAVAILABLE`, qualité `UNUSABLE`, motif `technical_incident` ;
- enregistre un incident « flux microphone interrompu » ;
- ne prolonge pas la ligne de temps au-delà de 2,0 s par du silence ;
- ferme le service et le thread d'écriture.

Les deux tests d'interruption simulée passent. Ils ne valident pas R08 comme recette de débranchement.
