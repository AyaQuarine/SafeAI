# Docker — SafeAI Common V1.0

> Docker sert d'abord à **reproduire la démonstration** et, en option, à exécuter le pipeline hors ligne sur CPU. Les images n'embarquent **aucun modèle ni WAV réel**. Les constructions complètes doivent être testées localement : la compatibilité pyannote / torchcodec / FFmpeg dépend des versions installées.
>
> L'image `demo` est volontairement légère : pas de `ffmpeg`, pas de `silero-vad`/Torch CUDA (le `pyproject` principal les tire sinon). Elle sert la page `/common` et le diagnostic UI. Le profil `offline` ajoute FFmpeg + modèles.

## A. Démonstration légère

Prérequis : Docker Desktop + Docker Compose v2. Dans la racine du projet :

```powershell
docker compose build demo
docker compose up -d demo
```

Le Compose publie la démo sur **`127.0.0.1:8766`** (conteneur 8765) pour coexister avec le serveur natif sur `8765`.

- Docker : `http://127.0.0.1:8766/common`
- Natif : `http://127.0.0.1:8765/common`

Le mapping reste borné à `127.0.0.1` (pas d'exposition réseau). Le **microphone n'est pas disponible dans le conteneur** (`SAFEAI_DISABLE_MICROPHONE=1`). Pour tester le micro Windows, utiliser le Python natif. Aucun jeton Hugging Face n'est requis pour le service `demo`.

Pour arrêter :

```powershell
docker compose down
```

Le port est exposé sur `127.0.0.1` uniquement. Ne pas publier ce service sur une adresse publique.

## B. Pipeline complet hors ligne (option lourde)

**CPU uniquement par défaut ; pas de garantie temps réel.** Le build inclut `faster-whisper`, `pyannote.audio`, Silero et FFmpeg et peut être volumineux. Il doit être vérifié sur la machine de l'équipe :

```powershell
docker compose --profile models build offline
mkdir data\local_dev -ErrorAction SilentlyContinue
mkdir outputs -ErrorAction SilentlyContinue
```

Chaque membre accepte les conditions du modèle `pyannote/speaker-diarization-community-1` sur Hugging Face. Pour le tout premier chargement, fournir une autorisation **hors Git**, par exemple par la variable temporaire `HF_TOKEN` du shell (attention : les variables d'environnement peuvent être visibles aux administrateurs de la machine ou dans l'inspection du conteneur). L'accès au modèle peut nécessiter Internet au départ. Éviter le token dans un fichier commité, dans les arguments de commande et dans les logs.

```powershell
# WAV autorisé, présent sur le poste : data/local_dev/r06_deux_locuteurs.wav
# Exemple : choisir la taille small (premier lancement = éventuel téléchargement)
docker compose --profile models run --rm offline `
  python -m safeai.pipeline inspect `
  --file data/local_dev/r06_deux_locuteurs.wav `
  --out /app/outputs/common_r06.json `
  --whisper-model small `
  --session-id session_common_r06
```

Les deux JSON sont écrits dans `outputs/` (montage côté hôte), puis peuvent être chargés dans `/common` depuis le navigateur. Les modèles téléchargés restent dans les volumes Docker (`hf_models`, `whisper_models`), pas dans le dépôt. Pour une véritable exécution *air-gapped*, il faut précharger explicitement tous les poids et leurs dépendances et les rendre accessibles au backend ; ce Compose n'est **pas** une garantie de fonctionnement sans Internet dès le premier démarrage.

## C. Ce qui reste à valider chez chaque membre

- `docker compose config` (syntaxe), puis construction de `demo` et ouverture `/common`.
- Tests locaux de l'application (`pytest`) : **174** = gel backend ; **177** = backend + tests UI `/common`. Ces chiffres ne sont pas automatiquement reproduits dans l'image Docker.
- Installation du profil `offline`, compatibilité CPU/TorchCodec/FFmpeg et permissions Hugging Face.
- Aucun token, fichier réel ni poids de modèle dans Git ou l'image construite.
- Docker Desktop sous Windows n'expose généralement pas directement le microphone Intel à un conteneur Linux. Conserver la capture micro **en natif**.

## D. Référence

- <https://huggingface.co/pyannote/speaker-diarization-community-1> — conditions d'utilisation et cache/offline.
- <https://github.com/pyannote/pyannote-audio> — dépendance FFmpeg pour TorchCodec.
