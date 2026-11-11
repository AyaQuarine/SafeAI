# Interface commune — analyses textuelle et vocale

SafeAI Common V1.0 fournit deux sorties complémentaires pour un même `session_id` et la même horloge de session :

| Sortie | Destinataire | Contenu |
| --- | --- | --- |
| `aligned_session.json` (`AlignedSession`) | Équipe texte / intégration | Texte, temps, attribution, incertitude, diagnostics |
| `*_diarization.json` (`DiarizationResult`) | Équipe voix / intégration | Tous les tours Community-1, locuteurs, chevauchements |

Aucun `AlignedUtterance` n'est inventé pour représenter un chevauchement sans transcription. Les chevauchements restent dans la branche diarisation.

## Jeu fictif (sans modèles)

Pour coder sans attendre un WAV réel : `data/interface_commune/session_demo.json` (`provenance: "synthetic_fixture"`). Ce n'est **pas** un résultat Community-1 ni une transcription de réunion réelle.

Fichier principal : **quatre énoncés**, pas cinq. La fenêtre 10–11 s n'est représentée **qu'une fois**, comme deux tours `overlap`.

| Temps | Locuteur | Texte | `attribution` | Audio |
| --- | --- | --- | --- | --- |
| 5,0–7,2 s | `speaker_00` | « Bonjour à tous. » | `attributed` | Pas de WAV (`audio_available: false`) |
| 8,1–10,0 s | `speaker_01` | « Je voudrais parler. » | `attributed` | idem |
| 10,0–10,8 s | `speaker_00` | « D'accord. » | `overlap` | idem |
| 10,2–11,0 s | `speaker_01` | « Je voudrais parler. » | `overlap` | idem |

Les deux lignes `overlap` décrivent **un seul chevauchement**. Silence légitime : `data/interface_commune/session_empty.json`, `status: "empty"`, `utterances: []`.

```python
from safeai.alignment import load_aligned_session

session = load_aligned_session("data/interface_commune/session_demo.json")
for item in session.utterances:
    print(item.start_s, item.speaker_id, item.attribution, item.text)
```

## Pipeline réel hors ligne

```powershell
$env:PYTHONPATH = "src"
.\.venv-diarization\Scripts\python.exe -m safeai.pipeline inspect `
  --file data/local_dev/r06_deux_locuteurs.wav `
  --out data/local_dev/common_r06.json `
  --whisper-model .cache/faster-whisper-small `
  --session-id session_common_r06
```

Écrit aussi `common_r06_diarization.json`. Les WAV et modèles restent hors Git (`data/local_dev/`, `.cache/`).

Statuts de session utiles aux deux équipes :

| Statut | Signification |
| --- | --- |
| `completed` | Transcription et alignement livrés sans rejet forcé |
| `empty` | Silence **confirmé** (preuve Silero + qualité) ; pas d'`utterances` |
| `partial` | Une partie du texte est écartée ou de la parole acoustique manque de texte |
| `failed` | Incohérence ou silence **non** vérifié (preuve acoustique absente) |

Diagnostics typiques : hallucination ASR écartée, texte suspect, token non lexical trop long, parole Community-1 non transcrite.

## Qui consomme quoi

| Équipe | Entrées |
| --- | --- |
| Texte | `AlignedSession` (+ diagnostics) ; `TranscriptSegment` bruts si besoin |
| Voix | `DiarizationResult` (tours, overlaps) ; `AudioChunk` / `SpeechSegment` / qualité du cahier n° 01 |

`audio_available: true` signifie qu'un WAV de session **existe** pour l'intervalle, hors Git. Les identifiants `speaker_00` / `speaker_01` sont anonymes, pas des personnes.

Une entrée `utterances` mal formée (non-objet, `"false"` booléen, clé obligatoire absente) est **rejetée** (`InvalidAlignmentError`).

## Ce que ce n'est pas

- Pas une preuve de performance WER / DER.
- Pas une identité de personne.
- Pas la transcription en continu.
- Pas l'intégration des analyses textuelle et vocale métier.
