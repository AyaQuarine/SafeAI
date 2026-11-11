# R05 — Microphone coupé, 8 novembre 2026

Microphone 1, 30 s. Le mute Windows du périphérique Intel a été appliqué par le contrôle de capture, de 10,004 s à 20,028 s après l'ouverture du WAV, puis retiré. La personne parlait avant, était invitée à continuer pendant le mute, puis à reparler. Aucune modification du code d'acquisition.

Fichier à examiner : `data/local_dev/r05_mute.wav`

470 681 échantillons, 29,418 s, 16 kHz mono. SHA-256 `07531bc94e383be01cf6ff271737684b411a11c502fab5c1af2d955bbf2ed54a`.

`data/local_dev/r05_mute_prise1.wav` est une prise invalidée : le script de mute se réarmait après 20 s. Elle ne fait pas partie de la recette.

Le contrôle Windows, via `IAudioEndpointVolume.SetMute` et `GetMute`, a renvoyé :

```
MUTE_ON 10.004
set mute=True hr=0 now=True Microphone Array (Intel® Smart Sound Technology for Digital Microphones)
MUTE_OFF 20.028
set mute=False hr=0 now=False Microphone Array (Intel® Smart Sound Technology for Digital Microphones)
RESTORE
set mute=False hr=1 now=False Microphone Array (Intel® Smart Sound Technology for Digital Microphones)
```

`SetIntel()` vise chaque endpoint de capture actif dont le nom contient « Intel ». Ici un seul endpoint répondait, et c'était bien ce réseau. Ce n'est pas une identification du numéro PortAudio 1. Le retour de la voix à 21,2 s, après la levée du mute à 20,03 s, mêle le moment où la parole reprend et les marges de Silero. Ce n'est pas une latence de réactivation de 1,2 s.

## Console

```
Duration           : 29.42 s
Chunks published   : 30
Chunks retained    : 30
Timestamp errors   : 0
Speech alignment   : 0
Signal status      : ACTIVE
VAD backend        : silero
Speech segments    : 2
Publisher drops    : 0

 0.0-5.7    Silence
 5.7-9.1    Parole
 9.1-21.2   Silence
21.2-26.0   Parole
26.0-29.4   Silence
```

Aucun incident. Dernier bloc : 6 681 échantillons, soit 0,418 s. Le flux n'a pas été interrompu.

## Niveau

| Période | RMS | Lecture |
| --- | --- | --- |
| 0–5 s | ≤ 0,0006 | silence, micro ouvert |
| 5–9 s | 0,009 à 0,015 | voix, crête 0,069 |
| 10–19 s | 0 | mute Windows, échantillons nuls |
| 19–21 s | 0,0015 | reprise du périphérique |
| 21–26 s | 0,010 à 0,023 | voix de retour, crête 0,088 |
| 26–29,4 s | ≤ 0,0001 | silence |

Le mute envoie du silence numérique. Il ne coupe pas le flux : le statut reste `ACTIVE`, sans incident et sans trou d'horodatage. SafeAI ne lit pas l'état mute de ce microphone. La sonde retourne `UNKNOWN`. La console ne l'affiche pas. Ce n'est pas une détection de mute.

## Décision

| Contrôle | Observé |
| --- | --- |
| Voix avant | 5,7–9,1 s |
| Pendant le mute | Signal nul, aucune parole détectée |
| Voix après | 21,2–26,0 s, environ 1,2 s après la levée du mute |
| MuteStatus | Non observable, `UNKNOWN` |
| Horodatages et pertes | 0 |
| Dernier bloc | Conservé |
| Incident | Aucun. Aucune coupure inventée |

**Accepté, limite documentée.** Le signal vocal disparaît puis revient. Le système décrit cette absence comme du silence sur un flux actif, pas comme un microphone coupé. R06 n'est pas lancé.
