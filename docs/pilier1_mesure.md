# Mesure — acquisition française et départ transcription

Date : 2026-11-02  
VAD : Silero, seuil 0,5, trames de 32 ms (512 échantillons à 16 kHz)  
Les annotations viennent des corpus (SUMM-RE, Accueil_UBS). Elles n'ont pas été réécoutées une à une.

Les planchers 0,55 / 0,50 des tests automatiques sont des tests de fonctionnement. Les chiffres ci-dessous sont la mesure.

## Corpus

7 fichiers, 240,0 s d'audio, tous en français. Aucun corpus anglais.

| Fichier | Provenance | Locuteurs | Durée | Plage annotée | Parole annotée |
| --- | --- | ---: | ---: | ---: | ---: |
| fr_reunion_mix_00_30s.wav | SUMM-RE `004c_PAPH` | 4 | 30,0 s | 30,0 s | 22,8 s |
| fr_reunion_mix_30_60s.wav | même réunion | 4 | 30,0 s | 30,0 s | 23,5 s |
| fr_reunion_mix_90_45s.wav | même réunion | 4 | 45,0 s | 44,6 s | 38,2 s |
| fr_reunion_closetalk_00_30s.wav | même réunion, 1 micro | 1 | 30,0 s | 30,0 s | 5,9 s |
| fr_accueil_ubs_01_40s.wav | Accueil UBS | 2 | 40,0 s | 20,4 s | 20,4 s |
| fr_accueil_ubs_02_24s.wav | Accueil UBS | 2 | 25,0 s | 11,4 s | 11,4 s |
| fr_accueil_ubs_03_40s.wav | Accueil UBS | 2 | 40,0 s | 13,0 s | 13,0 s |

Les trois mix et la piste individuelle sont des extraits de **la même prise** `004c_PAPH`. Ils ne doivent pas être répartis entre calibration et test. Les trois appels UBS sont des prises distinctes, mais seul le début est annoté : 60 s en fin de fichier ne sont pas scorés.

Licence : CC BY-SA. Audio hors Git. Ces voix publiées restent des données personnelles : la licence autorise certains usages, elle n'annule pas les obligations de confidentialité, de conservation et de minimisation. Aucune prise personnelle du projet n'est versionnée.

## VAD Silero

Score calculé seulement sur la plage annotée. Effectifs de trames de 32 ms.

| Condition | Trames | TP | FP | FN | Précision | Rappel | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mix réunion, 4 locuteurs, 105 s | 3270 | 2536 | 226 | 106 | 0,918 | 0,960 | 0,939 |
| Micro individuel, 5,9 s de parole / 30 s | 938 | 170 | 16 | 13 | 0,914 | 0,929 | 0,921 |
| Dialogue téléphonique annoté, 44,8 s | 1401 | 1281 | 0 | 120 | 1,000 | 0,914 | 0,955 |
| **Ensemble** | **5609** | **3987** | **242** | **239** | **0,943** | **0,943** | **0,943** |

Détail par fichier : `data/conversations_fr/vad_metrics.json`. Horodatage : 0 erreur sur les 7 fichiers.

Lecture :

- Le rappel réel est autour de 0,94, pas de 0,55. Ce 0,55 n'était que le plancher du test de fumée.
- Le dialogue UBS a une précision de 1 sur la partie annotée, qui est presque entièrement de la parole. Ça ne mesure pas les fausses détections dans les 60 s non annotées.
- La piste individuelle est le cas le plus creux (183 trames de parole). Le F1 y reste à 0,92, sur un effectif trop petit pour conclure.
- Les chevauchements ne sont pas scorés à part. Une réunion facile peut encore masquer une voix faible ou une interruption.

Ce corpus suffit pour brancher Whisper. Il ne suffit pas pour déclarer l'acquisition validée. R01–R08, au microphone, ne sont pas faits.

## Cas limites ajoutés à la suite

| Cas | Avant | Après |
| --- | --- | --- |
| Deux blocs séparés par une seconde absente | 16 000 zéros insérés, relus comme silence | `timeline_gap`, 16 000 échantillons perdus, aucun zéro dans le segment. Le reste de fenêtre VAD ne traverse pas le trou. |
| Parole de 0,2 s à 0,4 s puis silence | `silence_duration_s = 0` | 0,60 s, puis 1,60 s si la seconde suivante est silencieuse |

## Transcription

Module : `src/safeai/transcription/`. `speaker_id` reste `null`. `vad_filter` est faux. Un temps local de 0,35 s dans un segment qui commence à 12,40 s devient 12,75 s. Couvert par `tests/transcription/test_timestamps.py`.

Modèle : faster-whisper `small`, CPU, INT8, local. Le cache Hugging Face avait laissé `model.bin` à 0 octet ; le fichier utile (461 Mo) est dans `.cache/faster-whisper-small/`, hors Git. L'audio 16 kHz est passé tel quel, sans repasser par PyAV.

La référence est le texte du corpus, normalisé (minuscules, `_` et ponctuation retirés). Les nombres ne sont pas réécrits en lettres. Ce n'est pas une transcription corrigée à la main.

| Fichier | Durée comparée | Segments | WER |
| --- | ---: | ---: | ---: |
| fr_accueil_ubs_02_24s.wav | 0 – 11,42 s | 2 | **0,583** |
| fr_reunion_mix_00_30s.wav | 0 – 30,00 s | 6 | **0,554** |

Le dialogue UBS donne, en clair :

- Référence : « UBS bonjour. Oui bonjour, ce serait pour savoir comment obtenir un dossier de candidature. Oui, je vais vous passer le secrétariat. »
- Hypothèse : « Bonjour. Oui, bonjour. Ça fait pour savoir comment obtenir un dossier de candidature pour le DSS, droit des affaires, droit d'EPM. Oui, je vais vous passer le tutoriel. »

Deux écarts de nature différente. « Ça fait » pour « ce serait », et « tutoriel » pour « secrétariat », sont des erreurs du modèle. Les mots « DSS, droit des affaires » sont absents de la référence alors que le tour dure 7,5 s : la référence corpus est plus courte que la parole. Une partie du 0,583 vient de là.

Sur la réunion, Whisper suit le fil (date, fin mai, première semaine de juin, 24 mai) mais perd des tours (« attention aux jours fériés », « gruyère ») et déforme « événement » et « férié ». La référence contient les hésitations et les chevauchements. Le 0,554 n'est pas un WER de conditions calmes.

Objectif du guide : WER ≤ 0,20 en conditions calmes, et rappel ≥ 0,90 des phrases d'alerte. Ce point de départ est au-dessus de 0,50. Il ne valide pas le pilier 2. Il fixe le niveau d'où on part, avec `small` sur CPU.

Les deux fichiers ci-dessus viennent de prises déjà dans le corpus. Aucune réunion supplémentaire n'a été téléchargée.

## Acquisition puis Whisper

`fr_reunion_mix_90_45s.wav`, 45 s, 4 locuteurs, même réunion `004c_PAPH`. Silero extrait les `SpeechSegment`, faster-whisper `small` INT8 les transcrit. Les temps du texte restent dans le segment source. `speaker_id` reste null.

| Mesure | Valeur |
| --- | ---: |
| Segments de parole | 6 |
| Segments de texte | 13 |
| Acquisition | 2,57 s |
| Transcription | 30,35 s |
| WER contre la référence corpus | **0,299** |

Le fil est reconnaissable (comité scientifique, 3 et 10, petite marge, affiner sur la semaine). Les écarts nets sont « local » → « l'eau câlin », « Timone » → « Timon », « juin » → « joints », « doodle » → « doule d'œil ». Le 0,299 reste au-dessus de 0,20. La référence contient encore les hésitations et les chevauchements.

## Fichier complet ou segments Silero

Même audio, coupé à la dernière annotation, même texte de corpus, modèle `small` INT8, CPU, faisceau 5, VAD interne de Whisper coupé. Relancer : `python scripts/compare_transcription.py`. Le détail est dans `data/conversations_fr/transcription_compare.json`.

La référence n'a pas été réécoutée. L'écart entre les deux méthodes isole la stratégie. Il ne mesure pas encore la qualité absolue.

| Fichier | A fichier complet | B Silero puis Whisper | Omissions A | Omissions B | Calcul A | Calcul B | Premier texte B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mix 0–30 s | 0,554 | 0,604 | 46 | 37 | 9,8 s | 20,2 s | 5,9 s |
| mix 30–60 s | 0,457 | 0,457 | 29 | 34 | 11,3 s | 17,8 s | 3,9 s |
| mix 90 s, 45 s | 0,228 | 0,299 | 25 | 32 | 13,2 s | 26,1 s | 3,8 s |
| dialogue UBS 02, 11,4 s annotés | 0,583 | 0,625 | 3 | 3 | 9,7 s | 9,9 s | 5,0 s |

Les horodatages de B restent dans leur segment de parole (0 dépassement).

Sur l'extrait de 45 s, le fichier complet entend ce que les segments perdent : « local », « Timone », « 3 juin », « doodle ». Les segments donnent « l'eau câlin », « Timon », « trois joints », « doule d'œil ». Une partie du 0,299 vient donc du découpage, pas du modèle. Le 0,228 qui reste est surtout la référence : hésitations, répétitions, et « match » rendu par « marche ».

Sur 0–30 s, les deux méthodes restent autour de 0,55 et oublient les mêmes tours (« jours fériés », « gruyère »). Là, changer le découpage ne suffit pas. Le texte corpus, avec chevauchements et amorces, pèse plus que la stratégie.

Le fichier complet est plus juste et plus rapide au total. Les segments publient le premier texte plus tôt (environ 4 à 6 s après le début du traitement, contre 10 à 13 s pour le fichier entier). La latence moyenne une fois le segment fermé est de 3,5 à 4,7 s.

Les trois mix sont la même prise `004c_PAPH`. Aucune autre taille de modèle n'a été lancée : le texte de référence n'est pas encore corrigé à l'écoute, et les essais microphone R01–R08 ne sont pas faits.

## Méthode C — fenêtres glissantes

Même audio, même référence corpus non réécoutée, même modèle. Ce n'est pas un choix d'architecture.

| Variante | Découpe |
| --- | --- |
| C12 | Fenêtres de 12 s, recouvrement 2 s, fenêtres silencieuses sautées |
| C4 | Fenêtres de 4 s, recouvrement 1 s |
| Cp | Regroupement sur les pauses Silero d'au plus 0,6 s, plafond 12 s |

Un mot n'est publié que si son début tombe dans la région commise. Si le segment est coupé, les jetons sont recollés et une apostrophe initiale reste attachée au mot précédent. Une répétition réelle (« non non », 200 ms d'écart) est conservée. Seul un mot réémis par une autre fenêtre sur le même intervalle acoustique est retiré. `TranscriptSegment.validate()` n'a pas été assoupli. Les chiffres du tableau ont été mesurés avant cette correction : ils restent une comparaison de stratégies, pas une mesure à rejouer tant que les références ne sont pas réécoutées.

| Fichier | A | B | C12 | C4 | Cp |
| --- | ---: | ---: | ---: | ---: | ---: |
| mix 0–30 s | 0,554 | 0,604 | 0,554 | 0,644 | 0,624 |
| mix 30–60 s | 0,457 | 0,457 | 0,500 | 0,446 | 0,435 |
| mix 90 s, 45 s | 0,228 | 0,299 | 0,335 | 0,347 | 0,305 |
| dialogue UBS 02 | 0,583 | 0,625 | 0,583 | 0,667 | 0,583 |

Sur l'extrait de 45 s, C12 garde « local », « 3 juin » et « doodle », mais coupe « Timone » en « t mod » et « le 10 » en « le disque », et perd l'amorce. C4 garde « Timone » et « le 10 », et republie « on peut », « sachant » et « parce ». Cp retombe sur les erreurs de B (« l'occal », « Timon », « trois joints », « doule d'œil »), parce que les pauses réelles redonnent des segments courts.

Temps de calcul sur ces 45 s : A 14,7 s, Cp 17,3 s, C12 20,7 s, B 29,6 s, C4 50,0 s. Ces durées sont le calcul hors ligne une fois l'audio déjà là. Elles ne mesurent pas la latence de publication d'un flux capturé en direct.

Aucune variante progressive ne bat le fichier complet sur l'ensemble des quatre extraits. C12 égale A sur 0–30 s et sur le dialogue UBS, et reste au-dessus sur la réunion de 45 s, là où le contexte long sert. Le plafond de 8 à 12 s n'est pas retenu pour le MVP. Silero reste le détecteur de parole et de pauses. L'unité de transcription de référence reste le plus long contexte disponible, jusqu'à une référence réécoutée et une simulation de latence en flux.

## Bornes d'horodatage

`tests/test_transcription_boundaries.py` reproduit deux sorties simulées : un segment Whisper qui commence avant l'audio fourni, et un mot qui commence avant son segment. L'adaptateur ramène le segment dans `[offset, fin de l'audio]`, puis les mots dans ce segment, et journalise un déplacement supérieur à 20 ms. Les assertions de `validate()` sont inchangées.

Un intervalle de parole plus long que le plafond de `pause_windows` est coupé en morceaux adjacents. Une fenêtre fournie à une autre fréquence que 16 kHz est rééchantillonnée avant Whisper, et cette conversion est journalisée.

## Corpus court et rejeu en direct

Les sept WAV français font environ quatre minutes au total, en extraits de 24 à 45 s. Ils suffisent à comparer des stratégies et à corriger les contrats. Ils ne démontrent pas une réunion qui continue.

`scripts/replay_live.py` relit un WAV par blocs de 1 s, à l'heure où ils se termineraient en capture. Le fichier entier n'est plus chargé avant le premier bloc. En mode direct, `run_source` ne garde en mémoire qu'une fenêtre bornée par la file (64 blocs par défaut) ; `chunks_published` compte toute la session. L'historique complet reste disponible avec `retain_history=True` pour une mesure.

Sur le fichier artificiel de 300 s, avant cette borne, Silero avait suivi le rythme : horloge 305,8 s, 300 blocs, 0 perte. Retard d'émission p95 15 ms, traitement p95 29 ms, pic 90 Mio. Ce pic incluait le fichier entier et tous les blocs. Il ne décrit plus le mode direct.

`WhisperWorker` consomme les segments de parole sur un autre thread. Quatre scénarios passent avec un transcripteur simulé : parole courte publiée pendant que la capture continue, conversation en plusieurs segments, silence prolongé sans texte, consommateur ralenti dont la file saturée est comptée dans les incidents. Le contexte conservé est le texte déjà publié, dans une fenêtre bornée. Il n'est pas renvoyé à Whisper.

Sur `fr_accueil_ubs_02_24s.wav` rejoué en direct avec faster-whisper `small` INT8, 4 segments ont produit un texte, sans échec ni perte. L'horloge de publication est la même que l'origine de session : le délai entre la fin du segment et le texte reste de 7,4 s en médiane et 13,0 s au 95e percentile. L'horloge totale a duré 46,1 s pour 25,0 s d'audio. La cible du guide, médiane au plus 3 s et p95 au plus 5 s, n'est pas atteinte sur ce CPU avec un segment transcrit après sa fermeture.

Répéter ces extraits pour remplir cinq minutes sert à vérifier que le programme tient dans la durée. Cette piste artificielle ne mesure ni le WER ni une conversation réelle. Une session continue de 5 à 15 minutes, avec consentement s'il s'agit d'une nouvelle prise, reste à constituer avant de choisir le moteur progressif.
