# SAFEAI — CAHIER DES CHARGES TECHNIQUE N° 02

Module : Diarisation des locuteurs et préparation de l'alignement audio–texte  
Version : 1.0 — Proposition pour validation  
Date : 9 novembre 2026  
Projet : SafeAI — traitement local et explicable de conversations  
Statut : À valider par l'équipe et le responsable de projet

**Promesse client en une phrase.** Après avoir enregistré correctement une conversation, SafeAI doit pouvoir distinguer quand chaque voix intervient et retrouver ses interventions au fil de la session, sans prétendre connaître l'identité réelle des personnes.

## 1. Contexte et articulation avec le cahier n° 01

Le cahier des charges n° 01 a produit un module d'acquisition qui capte un microphone ou lit un WAV, normalise l'audio en 16 kHz, mono, float32, horodate les données, détecte la présence de parole avec Silero, évalue la qualité du signal et publie les événements vers les modules suivants.

État au 9 novembre 2026 : 101 tests automatisés réussis selon le rapport local ; 7 recettes terrain R01–R07 acceptées, plusieurs avec réserves ; R08 matériel non concluant, mais interruption de flux simulée correctement signalée. Le cahier n° 01 n'a pas encore fait l'objet d'une clôture formelle.

Une limite importante a été identifiée sur R07 : le VAD en continu a manqué de la parole entre 124 et 132 s du fait de son état interne ; le rejeu complet reproduisait le comportement en direct, tandis qu'une remise à zéro de Silero permet de détecter ce passage. Conséquence d'architecture : la diarisation ne doit pas dépendre exclusivement des `SpeechSegment` du VAD. Elle doit pouvoir lire le flux audio normalisé continu, y compris les zones étiquetées silence.

La transcription Whisper existe déjà partiellement sous forme de `WhisperTranscriber`, `TranscriptSegment` et `WhisperWorker`, mais sa qualité et sa latence en continu restent à valider. La diarisation est une branche parallèle à Whisper, et non une étape qui attend la fin de Whisper.

## 2. Objectif métier du module

Lors d'une réunion, SafeAI doit pouvoir indiquer :

- Combien de voix distinctes ont vraisemblablement participé à une session (estimation, non garantie absolue) ;
- Quand chacune commence et termine ses interventions ;
- Quand deux personnes parlent simultanément, dans la mesure où les données permettent de le détecter ;
- Quand une attribution est incertaine, sans inventer de locuteur ;
- Comment rapprocher ultérieurement les tours de parole du texte transcrit par Whisper.

Exemple vulgarisé :

- 00:03–00:07 : « Voix 1 » parle.
- 00:08–00:11 : « Voix 2 » répond.
- 00:10–00:11 : les deux voix se chevauchent.
- 00:12–00:15 : silence ou aucune voix détectée.

Les étiquettes Voix 1 / Voix 2 sont temporaires et propres à une session. Elles ne sont ni des noms, ni une reconnaissance biométrique d'identité.

## 3. Périmètre et exclusions

### Inclus dans la V1 (obligatoire)

- Traitement hors ligne d'un WAV complet provenant de l'acquisition, avec ses timestamps de session.
- Diarisation automatique des tours de parole ; estimation du nombre de locuteurs sans imposer artificiellement « 2 » pour tous les fichiers.
- Attribution stable des IDs anonymes à l'intérieur d'une même session quand un locuteur reprend la parole.
- Représentation explicite des chevauchements ; possibilité d'attribution `unknown`.
- Sortie structurée `SpeakerTurn`, validée et exploitable par les futurs modules.
- Mesure reproductible sur des données annotées : DER, erreurs de comptage, tours manqués, chevauchements et temps de calcul.
- Gestion des erreurs, journaux sans audio brut, exécution locale et tests automatiques.

### Prévu comme extension, hors critères de réception V1

- Diarisation progressive pendant une réunion en direct, avec gestion de l'identité d'une fenêtre à l'autre ;
- Alignement mot-par-mot avec Whisper et publication du texte attribué ;
- Optimisation de la latence bout en bout en conditions réelles ;
- Amélioration des performances par adaptation ou entraînement d'un modèle si une évaluation le justifie.

### Explicitement exclu

- Identifier une personne par son nom, son visage ou une base d'empreintes vocales ;
- Déduire son âge, sa personnalité, sa santé, ses émotions ou son niveau de risque à partir du locuteur ;
- Séparer parfaitement deux signaux acoustiques simultanés dans deux pistes audio indépendantes (source separation) ;
- Déclencher une alerte ; cette décision appartiendra à la fusion et à la machine à états SafeAI.

## 4. Architecture fonctionnelle et interfaces

```
Microphone ou WAV
        │
        ▼
[CDC 01 — Acquisition / AudioChunk continu horodaté]
        │
        ├──────────────────────┐
        ▼                      ▼
[CDC 02 — Diarisation]   [Transcription Whisper — déjà amorcée]
  SpeakerTurn                TranscriptSegment
        │                      │
        └──────────┬───────────┘
                   ▼
    [Alignement audio–texte — étape suivante]
                   ▼
       « Voix 2 a dit ... à 00:18 »
                   ▼
    Analyses texte et voix → fusion → états de risque
```

Règle de conception : l'acquisition transmet une horloge de session unique. Les intervalles de diarisation ne sont pas recalés arbitrairement sur le début d'un extrait. Si un fichier représente un extrait commençant à 120 s de la session, la sortie doit porter des temps absolus de session (ou un offset explicite).

### 4.1 Entrées du module

| Entrée | Provenance | Contraintes |
| --- | --- | --- |
| `AudioChunk` ou WAV normalisé | CDC 01 | Audio mono float32, 16 kHz ; temps cohérents |
| `session_id` | Acquisition | Identifiant non ambigu de la session |
| `session_origin` / `offset_s` | Acquisition | Référentiel temporel commun |
| Événements de qualité | Acquisition | Signalement du bruit, de l'indisponibilité, des pertes |
| Configuration de diarisation | `configs/diarization.yaml` | Modèle, mode, bornes de locuteurs facultatives, ressources |

La diarisation reçoit le flux audio complet. Les résultats Silero peuvent accélérer ou guider le traitement, mais ils ne sont pas l'unique voie d'accès à l'audio.

### 4.2 Sorties : contrat `SpeakerTurn`

```json
{
  "session_id": "session_001",
  "start_s": 18.20,
  "end_s": 21.60,
  "speaker_id": "speaker_01",
  "overlap": false,
  "confidence": null,
  "status": "completed"
}
```

Exemple illustratif, non extrait d'une exécution. Champs minimaux obligatoires : `session_id`, `start_s`, `end_s`, `speaker_id` (nullable pour `unknown`), `status`. `overlap` peut être dérivé de l'intersection des tours ; `confidence` est facultatif et ne doit pas être inventé lorsqu'un modèle ne fournit pas de score exploitable.

Convention temporelle : `duration_s` est la durée du fichier traité, `offset_s` la position de début dans la session, `start_s` et `end_s` les temps absolus de session. Un extrait de 10 s commençant à 120 s accepte un tour 122–125 s, car la fenêtre valide est `[offset_s, offset_s + duration_s]`.

`overlap` est un chevauchement vocal : deux identifiants distincts et connus se recouvrent. L'intersection temporelle d'un même locuteur, ou d'un tour `unknown`, n'est pas un chevauchement vocal confirmé.

Invariants : `start_s >= 0`, `end_s > start_s`, intervalles dans la fenêtre `[offset_s, offset_s + duration_s]`, `speaker_id` anonyme et limité à la session, aucune identité personnelle, événements d'erreur explicites. Deux tours peuvent se chevaucher ; il est interdit de les écraser pour imposer artificiellement un seul locuteur.

### 4.3 Sortie de session `DiarizationResult`

- `session_id`, `duration_s`, `model_id`, `model_version`, paramètres et hash de configuration ;
- liste horodatée des `SpeakerTurn` ;
- `speakers_estimated`, `processing_seconds`, `real_time_factor` ;
- informations sur les segments `unknown`, les chevauchements et les erreurs ;
- statut `completed`, `partial`, `empty` ou `failed`.

Les métadonnées sont sauvegardables en JSON ou RTTM pour l'évaluation ; l'audio brut reste local et n'est jamais ajouté aux journaux.

## 5. Composants techniques attendus

| Composant | Responsabilité | Livrable |
| --- | --- | --- |
| `DiarizationAdapter` | Adapter l'API du modèle, isoler ses dépendances | Interface stable et erreurs gérées |
| `SpeakerTurn` / `DiarizationResult` | Contrats et validation | `contracts.py` et tests |
| `DiarizationService` | Orchestrer le traitement d'un WAV / buffer | `service.py` |
| `SpeakerTurnPublisher` | Diffuser les tours vers les modules suivants | API de souscription testée |
| `DiarizationEvaluator` | Comparer aux annotations | DER et rapports reproductibles |
| `diarization inspect` | Interface de diagnostic locale | CLI + export JSON/RTTM |

Une bibliothèque de référence envisagée est pyannote.audio, avec le pipeline préentraîné `pyannote/speaker-diarization-community-1`, ou un backend équivalent validé par benchmark. Ce modèle travaille sur de l'audio mono 16 kHz, accepte une utilisation locale et fournit des sorties de diarisation normales ainsi qu'une variante dite exclusive pour faciliter l'alignement texte ; les conditions d'accès au modèle doivent être acceptées préalablement. Il faudra vérifier les versions, les licences et les temps d'exécution sur l'ordinateur cible avant de figer le choix.

La sortie exclusive est utile pour un alignement texte où un seul locuteur doit être choisi ; elle ne remplace pas la sortie de diarisation standard qui doit conserver l'information sur les chevauchements.

Le système doit continuer à fonctionner avec des erreurs explicites si le modèle n'est pas téléchargé ou si le réseau est indisponible. Un échec de téléchargement ne doit pas être maquillé en résultat « zéro locuteur ».

## 6. Exigences fonctionnelles et vérification

| ID | Exigence | Critère de validation |
| --- | --- | --- |
| DIA-01 | Lire l'audio normalisé du CDC 01 | WAV / buffer 16 kHz mono traité sans perte de durée |
| DIA-02 | Garder le temps de session | Timestamps valides et offset appliqué une seule fois |
| DIA-03 | Détecter des tours de parole | Tours horodatés sur une conversation annotée |
| DIA-04 | Estimer les locuteurs | Nombre estimé reporté ; erreurs de sur/sous-segmentation mesurées |
| DIA-05 | Réutiliser l'ID d'une voix | Même voix reconnue après une pause dans la même session, sinon incertitude signalée |
| DIA-06 | Représenter les chevauchements | Deux tours pouvant coexister dans une même plage temporelle |
| DIA-07 | Gérer `unknown` et l'ambiguïté | Aucune identité inventée lorsqu'une attribution n'est pas fiable |
| DIA-08 | Tolérer le silence et le bruit | Pas de locuteur inventé sur un silence pur ; incertitude en présence de bruit |
| DIA-09 | Ne pas dépendre uniquement de Silero | WAV complet traité même lorsque le VAD ne produit aucun segment |
| DIA-10 | Exposer une sortie stable | JSON / RTTM et API `subscribe_speaker_turns()` (ou équivalent) |
| DIA-11 | Gérer erreurs / sessions | Modèle indisponible ou fichier invalide → erreur explicite ; sessions isolées |
| DIA-12 | Mesurer la qualité | DER calculé sur annotations, résultats détaillés et reproductibles |
| DIA-13 | Maîtriser les ressources | Temps et mémoire mesurés sur un WAV de 5 min ; limites documentées |
| DIA-14 | Préserver la confidentialité | Inférence locale ; consentement ; aucun audio brut / token dans Git ou les logs |
| DIA-15 | Rendre le module intégrable | Tests et documentation ; aucun couplage indispensable à l'interface graphique |

## 7. Stratégie d'évaluation et critères d'acceptation

### 7.1 Données d'essai

- Données de démarrage : R01 (une voix), R02 (silence), R06 (voix directe + haut-parleur), R07 (session cinq minutes). Ce sont des tests de fonctionnement, pas une vérité terrain suffisante pour le DER.
- Données de référence : sélectionner des conversations françaises multi-locuteurs dont l'utilisation est autorisée et pour lesquelles les débuts/fins de parole et identifiants de locuteurs sont annotés ; sinon, créer des annotations manuelles vérifiées.
- Jeu de validation minimal proposé : au moins deux enregistrements distincts avec au moins deux locuteurs, dont un passage avec chevauchement, et un extrait mono-locuteur / silence. La durée réellement annotée doit être inscrite dans le rapport. Ce minimum suffit à démarrer ; il ne démontre pas la généralisation à toutes les réunions.
- Séparer fichiers servant à choisir les paramètres et fichiers servant au verdict final ; conserver les versions des annotations.

### 7.2 Mesures obligatoires

| Mesure | Interprétation | Décision |
| --- | --- | --- |
| DER (Diarization Error Rate) | Temps de parole manqué, fausse parole et confusion entre locuteurs | Mesure principale sur références annotées |
| Erreur de nombre de locuteurs | Surestimation ou sous-estimation | À documenter par enregistrement |
| Couverture du chevauchement | Détection des périodes où plusieurs voix parlent | Qualifier séparément ces intervalles |
| Stabilité d'identité | Une même voix conserve son label à travers ses retours | Test ciblé par session |
| Temps de calcul / RTF | Coût de calcul versus durée audio | Rapport CPU/GPU et plateforme |
| Mémoire max | Consommation pour 5 minutes | Suivi reproductible |

Objectif qualité indicatif à valider avec le responsable : DER ≤ 25 %, si ce seuil est bien retenu par le guide SafeAI pour ce lot. Il s'agit d'une cible proposée, non d'un résultat acquis. Le protocole DER doit préciser le traitement des chevauchements et la tolérance autour des frontières (collar) ; les mêmes choix sont appliqués à toutes les méthodes comparées.

Aucune valeur « DER = X % » ne sera annoncée sans fichier annoté et script de calcul reproductible. La contrainte générale de latence SafeAI (médiane ≤ 3 s, p95 ≤ 5 s) concerne la chaîne audio → résultat affiché, pas la seule diarisation hors ligne ; elle sera évaluée lors de l'intégration progressive, et n'est pas présumée atteinte.

### 7.3 Jeux de tests de recette DIA

| Test | Scénario | Résultat attendu |
| --- | --- | --- |
| D01 | WAV silencieux | Aucune voix attribuée |
| D02 | Une seule personne | Un identifiant cohérent, pas de faux locuteurs répétés |
| D03 | Deux voix en alternance | Deux identifiants anonymes, interventions temporelles plausibles |
| D04 | Une voix revient après une pause | Label réutilisé autant que possible |
| D05 | Deux voix simultanées | Chevauchement représenté ou limitation explicitement mesurée |
| D06 | Bruit / voix faible | Incertitudes et erreurs rapportées sans noms inventés |
| D07 | R07, parole manquée par le VAD | La diarisation reçoit l'audio 124–132 s même si Silero ne l'a pas sélectionné |
| D08 | WAV 5 min | Pas de panne, horodatages et ressources mesurés |
| D09 | Sessions successives | Pas de réutilisation accidentelle des speaker IDs entre sessions |
| D10 | Modèle absent / téléchargement expiré | Erreur explicite et mode local maîtrisé |
| D11 | Fichier invalide ou durée nulle | Échec contrôlé, aucun résultat mensonger |
| D12 | Evaluation DER | Rapport par fichier + agrégat sur annotations versionnées |

### 7.4 Definition of Done — Diarisation V1

Le cahier n° 02 est considéré recevable lorsque : le traitement WAV hors ligne fonctionne ; les contrats sont validés ; les voix anonymes et les chevauchements sont représentés ; les erreurs sont gérées ; les tests D01–D12 sont exécutés et documentés ; un jeu annoté permet de calculer un DER ; les ressources sont mesurées ; la confidentialité est respectée ; une démonstration locale et un rapport de limites sont disponibles. L'acceptation peut comporter des réserves explicites si le responsable les approuve.

## 8. Contraintes transversales et confidentialité

- Traitement local par défaut ; tout recours à un service distant nécessite une autorisation distincte.
- Consentement des participants avant toute prise ; voix = information personnelle potentiellement identifiante.
- Fichiers tests personnels dans `data/local_dev/`, exclus de Git ; durée de conservation minimale justifiée.
- Aucun token Hugging Face, empreinte vocale ou échantillon audio brut dans les journaux.
- Le module ne doit pas « reconnaître » des personnes nommément ni effectuer de décision médicale ou d'urgence.
- Un audio altéré, un manque de contexte, un chevauchement ou une perte d'échantillons peuvent imposer le statut `unknown` / `partial`.
- Les versions du code, du modèle, de la configuration et des références annotées doivent figurer dans les rapports.

## 9. Plan de réalisation technique du cahier n° 02

| Phase | Tâche concrète | Livrable / preuve |
| --- | --- | --- |
| D0 | Verrouiller le périmètre, les exigences et la confidentialité | Cahier n° 02 approuvé |
| D1 | Créer `SpeakerTurn`, `DiarizationResult`, configuration, validations | Contrats et tests unitaires |
| D2 | Brancher un backend préentraîné local sur un WAV | CLI et premier JSON horodaté |
| D3 | Tester R01, R02, R06 et R07 en mode hors ligne | Démonstrations et cas limites |
| D4 | Annoter des conversations multi-locuteurs ; calculer DER | Rapport d'évaluation et erreurs |
| D5 | Publier les tours vers le système et documenter les interfaces | `subscribe_speaker_turns()` + tests |
| D6 (suite) | Prototyper la diarisation continue et l'alignement avec Whisper | Mesures de latence et attributs mot/voix |

Priorité de développement immédiate : ne pas démarrer par l'alignement ou un modèle entraîné. Créer les contrats et obtenir le premier résultat de diarisation sur un WAV autorisé, puis le comparer à une annotation.

### Organisation conseillée du dépôt

```
src/safeai/
  acquisition/                  # CDC 01, déjà présent
  transcription/                # Whisper, développement amorcé
  diarization/
    __init__.py
    contracts.py
    config.py
    adapters.py
    service.py
    evaluation.py
    cli.py
configs/diarization.yaml
tests/diarization/
scripts/evaluate_diarization.py
docs/cahier_des_charges_02_diarisation.md
```

## 10. Références techniques et hypothèses

- Cahier des charges n° 01 SafeAI (document interne) ; rapports de recette de novembre 2026.
- pyannote.audio : https://github.com/pyannote/pyannote-audio
- Fiche officielle du pipeline Community-1 : https://huggingface.co/pyannote/speaker-diarization-community-1
- Les choix de backend, de DER cible, de durée du corpus et de calendrier sont des propositions à valider ; ils ne sont pas des résultats de tests déjà obtenus.

## 11. Feuille de route client : ce que SafeAI réalisera, étape par étape

Le client n'a pas besoin de connaître les noms des bibliothèques pour comprendre la valeur. Chaque cahier des charges apporte une capacité visible et testable.

| Étape | Formulation simple pour le client | Exemple du résultat | Situation au 09/11/2026 |
| --- | --- | --- | --- |
| CDC 01 — Entendre correctement | Le programme écoute ou ouvre un enregistrement, différencie parole / silence / bruit et signale les problèmes techniques | « Audio disponible ; parole de 00:05 à 00:09 ; qualité OK » | Développé, 7/8 recettes acceptées avec réserves ; clôture formelle en attente |
| CDC 02 — Distinguer les voix | Le programme repère quand « Voix 1 », « Voix 2 » parlent et quand elles se chevauchent, sans savoir leur nom | « Voix 1 : 00:05–00:09 ; Voix 2 : 00:10–00:15 » | À démarrer maintenant |
| CDC 03 — Comprendre les mots et les attribuer | Le programme écrit ce qui se dit et rapproche chaque phrase de la bonne voix quand c'est possible | « Voix 2 : “Pouvez-vous m'aider ?” » | Whisper déjà amorcé ; qualité, latence et alignement à valider |
| CDC 04 — Relever des indices observables | Deux branches évaluent le texte et la voix : mots préoccupants, hésitations, intensité, rythme, etc., sans diagnostic émotionnel automatique | « Demande d'aide explicite ; hausse du volume ; transcription incertaine » | À développer en parallèle selon la répartition de l'équipe |
| CDC 05 — Mettre les indices en contexte | Le programme confronte les signaux et leur chronologie, distingue une discussion ordinaire d'une situation justifiant l'attention | « Attention : demande d'aide répétée avec contexte concordant » | À construire après stabilisation des indices |
| CDC 06 — Expliquer et afficher | Une interface montre la conversation horodatée, les indices et le niveau de risque, avec justification et contrôle humain | « Niveau Attention — éléments observés et limites visibles » | À intégrer et tester en fin de MVP |

### Exemple de parcours client complet (illustratif)

Une réunion commence. SafeAI reçoit le son (CDC 01), repère les deux voix (CDC 02), transcrit les échanges (CDC 03), identifie des indices observables dans la voix et les mots (CDC 04), les recoupe dans le temps (CDC 05), puis affiche une synthèse explicable (CDC 06). Le système demeure pédagogique, n'établit aucun diagnostic et ne déclenche pas d'urgence de manière autonome.

### Prochaines décisions attendues du responsable de projet

- Approuver le périmètre du CDC 02 et préciser la priorité entre qualité hors ligne et démonstration progressive.
- Confirmer l'objectif DER (25 % proposé) et le jeu d'évaluation autorisé.
- Accepter ou refuser formellement les réserves encore ouvertes du CDC 01, notamment R07 et la limite matérielle R08.
- Valider le cadre de consentement, d'accès aux modèles et de conservation des enregistrements.

Décision de supervision proposée : autoriser dès maintenant le développement du CDC 02 ; conserver le CDC 01 en clôture documentaire avec réserves, sans bloquer le projet ; ne déclarer aucune qualité de diarisation acquise avant une évaluation annotée.

### Plan d'action immédiatement après validation du présent document

| Ordre | Ce que nous faisons | Ce que le client pourra voir |
| --- | --- | --- |
| 1 | Clore administrativement le CDC 01 avec ses réserves et geler le socle de capture | Rapport « acquisition disponible », limites affichées |
| 2 | Créer le module de diarisation et ses tests d'interface | Un WAV analysé, avec des plages « Voix 1 / Voix 2 » |
| 3 | Tester sur de vraies conversations annotées | Rapport des erreurs, chevauchements et estimation des locuteurs |
| 4 | Connecter la sortie des locuteurs à Whisper déjà développé | Des phrases horodatées associées à une voix, ou `unknown` |
| 5 | Ajouter indices textuels et vocaux, puis fusion et interface | Un compte rendu compréhensible avec les éléments observés |

Validation immédiate attendue : approuver ce CDC 02, choisir un enregistrement annoté autorisé, puis démarrer D1 (contrats et tests) avant de télécharger ou d'entraîner un modèle.
