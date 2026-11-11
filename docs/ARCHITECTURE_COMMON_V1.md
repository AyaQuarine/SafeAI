# Architecture — Common V1.0 (hors ligne)

```text
WAV autorisé (même fichier pour les modèles)
    │
    ├── AcquisitionService → Silero VAD + qualité → AcquisitionEvidence
    │
    ├── WhisperTranscriber (fichier entier) → TranscriptSegment + WordTiming
    │
    └── Community-1 (fichier entier) → DiarizationResult + SpeakerTurn
                           │
                 CommonPipeline.run_bundle()
                           │
             contrôle de cohérence acoustique
                 + filtrage temporel mot à mot
                           │
                  AlignmentEngine
                           │
               CommonBundle (en mémoire)
                 ├── aligned : AlignedSession
                 ├── diarization : DiarizationResult
                 └── acquisition : AcquisitionSnapshot
                           │
             CLI → JSON aligné + JSON diarisation
                           │
            Frontend /common (lecture de JSON)
```

## Principes stables

- Même `session_id`, durées compatibles, timestamps relatifs à la session.
- `completed` ne signifie pas WER ou DER validé ; `empty` exige preuve de silence, `partial` conserve les incertitudes, `failed` indique absence de preuve/échec.
- `speaker_id` anonyme ; `unknown` est un résultat légitime. Un overlap acoustique peut exister sans texte Whisper et doit rester dans `DiarizationResult`.
- Les diagnostics conservent les hypothèses ASR écartées, par exemple les tokens non lexicaux anormalement longs.
- La partie commune fournit les **données** ; les sous-équipes métier définiront ensuite leurs indicateurs de sortie et la fusion.
- Microphone et capture temps réel ne font pas partie du pipeline commun **hors ligne** validé. Docker n'apporte pas ce support.
