# SafeAI Common V1.0 — Démarrage de l'équipe

> **Statut** : pipeline commun **hors ligne** fonctionnel, non validé pour la transcription continue ni une prise de décision médicale. Les essais locaux déclarés R01 = `completed` avec réserve, R02 = `empty`, R06 = `partial` ; WER/DER/latence restent à qualifier.
>
> Ce guide accompagne la **version du code déjà présente dans le dépôt de développement**. Ne jamais publier de WAV réel, de jeton ou de modèle téléchargé. Aucun push GitHub n'est effectué automatiquement.

## 1. Quelles données avons-nous ?

| Producteur | Objet / fichier | Usage |
|---|---|---|
| Acquisition | `AudioChunk`, `SpeechSegment`, `AudioQualityEvent` | VAD, qualité et audio pour l'analyse vocale |
| Whisper | `TranscriptSegment`, `WordTiming` | Texte français et horodatages |
| Community-1 | `DiarizationResult`, `SpeakerTurn` | Tours de parole anonymes et chevauchements |
| Alignement | `AlignedSession`, `AlignedUtterance` | Texte avec locuteur ou incertitude explicite |
| Orchestration | `CommonPipeline.run_bundle()` | `aligned`, `diarization` et métadonnées d'acquisition |

**Horloge** : timestamps en secondes, relatifs au début de la même `session_id`. `speaker_00` ne correspond pas à une identité personnelle persistante. `overlap_count=2` peut signifier deux tours marqués chevauchants pour **une intersection**.

## 2. Installation locale Windows (recommandée pour le vrai modèle)

Depuis la racine du dépôt, dans PowerShell :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ui,vad,transcription]"
.\.venv\Scripts\python.exe -m pytest tests -q
```

Pour Community-1, utiliser l'environnement de diarisation déjà validé, si présent :

```powershell
.\.venv-diarization\Scripts\python.exe -c "import faster_whisper, pyannote.audio, silero_vad; print('imports OK')"
```

Sinon créer l'environnement en local (`py -3.12 -m venv .venv-diarization` puis installer `.[ui,vad,transcription,diarization]`). Accepter séparément les conditions du modèle Community-1 sur Hugging Face ; **ne pas partager de jeton dans le dépôt**. La première récupération des modèles exige un accès autorisé et peut nécessiter Internet. Une fois les poids en cache, l'inférence s'exécute localement.

## 3. Démo visuelle sans modèles

```powershell
.\.venv\Scripts\python.exe -m safeai.acquisition ui
```

Ouvrir `http://127.0.0.1:8765/common` : choisir un des exemples **fictifs** (`completed`, `empty`, `partial`) ou charger deux JSON locaux. Cette page ne lance **aucune inférence** et le navigateur ne téléverse pas les JSON sélectionnés au serveur.

`http://127.0.0.1:8765/` conserve l'ancien diagnostic d'acquisition. Il ne présente pas encore le pipeline Common V1.

## 4. Démo réelle hors ligne (avec modèles)

Garder les WAV autorisés dans `data/local_dev/` — ce dossier est ignoré par Git. Depuis la racine :

```powershell
.\.venv-diarization\Scripts\python.exe -m safeai.pipeline inspect `
  --file data/local_dev/r06_deux_locuteurs.wav `
  --out data/local_dev/common_r06.json `
  --whisper-model .cache/faster-whisper-small `
  --session-id session_common_r06
```

La commande crée deux fichiers avec le même `session_id` : `common_r06.json` et `common_r06_diarization.json`. Ouvrir la page `/common` et choisir ces deux fichiers. Le dashboard ne modifie pas les résultats des modèles.

## 5. Règles d'interprétation et de sécurité

- `completed` : transcription alignée disponible, sans rejet forcé (la justesse du texte n'est **pas** garantie).
- `empty` : absence de parole **confirmée** par acquisition/Silero/qualité, pas simplement absence de sortie des modèles.
- `partial` : certaines hypothèses ASR ont été écartées ou de la parole existe sans texte exploitable.
- `failed` : erreur ou preuve acoustique insuffisante ; ne pas convertir en faux silence.
- Les diagnostics peuvent mentionner des hypothèses hallucinées ; ne pas les transférer comme paroles confirmées.
- L'équipe **texte** consomme `AlignedSession`. L'équipe **voix** consomme `DiarizationResult` et, pour analyser la voix, les vrais blocs et segments audio dans un contexte autorisé.

## 6. Tests d'acceptation avant transfert

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv-diarization\Scripts\python.exe -m pytest tests -q
```

Sur l'environnement Windows de référence : **174** tests au gel backend Common V1.0 ; **177** après ajout de la page `/common` (+3 tests UI).  
Sans le corpus local `data/conversations_fr/` (exclu du dépôt) : **173** réussis et **4** ignorés (`tests/acquisition/test_french_conversations.py`). Chaque nouvelle installation doit relancer les tests.

Vérifier aussi la compatibilité d'identifiant et de durée des deux JSON ; les exemples fictifs sont fournis pour que chacun puisse commencer même sans accès aux modèles.

## 7. Limites de V1

Pas de transcription continue démontrée, pas de WER/DER satisfaisants prouvés, pas de classement clinique de risque, pas d'identification de personne, pas de microphone dans Docker Desktop. Sur R06, la parole non transcrite est conservée comme diagnostic, pas inventée. Les contrats des **sorties métier** des sous-équipes seront convenus séparément.

## 8. Docker (deux profils)

Voir `docs/DOCKER_EQUIPE.md`. `docker compose up --build demo` lance la démonstration web ; l'inférence complète se prépare dans le profil `models`, sans mise à disposition publique des données.
