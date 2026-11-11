"""Acquisition Silero puis Whisper sur un WAV français déjà annoté.

Les temps Whisper sont vérifiés contre les bornes du SpeechSegment.
Le WER compare le texte reconstruit à la référence du manifeste.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safeai.acquisition.service import run_file  # noqa: E402
from safeai.acquisition.vad import build_vad  # noqa: E402
from safeai.transcription.config import TranscriptionConfig  # noqa: E402
from safeai.transcription.scoring import normalize_french, word_error_rate  # noqa: E402
from safeai.transcription.whisper import WhisperTranscriber  # noqa: E402

MANIFEST = ROOT / "data" / "conversations_fr" / "manifest.json"
DEFAULT = ROOT / "data" / "conversations_fr" / "audio" / "fr_reunion_mix_90_45s.wav"
MODEL = ROOT / ".cache" / "faster-whisper-small"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    item = next(entry for entry in manifest["files"] if (ROOT / entry["path"]).resolve() == path.resolve())
    reference = " ".join(turn["text"] for turn in sorted(item["gold_turns"], key=lambda turn: turn["start_s"]))

    started = time.perf_counter()
    acquired = run_file(path, vad=build_vad())
    acquire_s = time.perf_counter() - started
    transcriber = WhisperTranscriber(
        TranscriptionConfig(model_size=str(MODEL), device="cpu", compute_type="int8", vad_filter=False)
    )
    transcripts = []
    transcribe_s = 0.0
    for index, speech in enumerate(acquired.speech_segments, start=1):
        t0 = time.perf_counter()
        pieces = transcriber.transcribe_speech(speech, f"speech_{index:03d}")
        transcribe_s += time.perf_counter() - t0
        for piece in pieces:
            piece.validate()
            if piece.start_s < speech.start_s - 0.02 or piece.end_s > speech.end_s + 0.02:
                raise SystemExit(
                    f"Horodatage hors segment {speech.start_s:.2f}-{speech.end_s:.2f} : "
                    f"{piece.start_s:.2f}-{piece.end_s:.2f}"
                )
            if piece.speaker_id is not None or piece.status != "completed":
                raise SystemExit(f"Contrat inattendu : {piece.status} speaker={piece.speaker_id}")
            transcripts.append(piece)
            print(f"[{piece.start_s:.2f} - {piece.end_s:.2f}] {piece.text}")

    hypothesis = " ".join(piece.text for piece in transcripts)
    score = word_error_rate(reference, hypothesis)
    print(f"\nSegments parole : {len(acquired.speech_segments)}")
    print(f"Segments texte  : {len(transcripts)}")
    print(f"Acquisition     : {acquire_s:.2f} s")
    print(f"Transcription   : {transcribe_s:.2f} s")
    print(f"Audio           : {item['duration_s']:.2f} s")
    print(f"Référence       : {normalize_french(reference)}")
    print(f"Hypothèse       : {normalize_french(hypothesis)}")
    print(f"WER             : {score:.3f}" if score is not None else "WER : référence vide")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
