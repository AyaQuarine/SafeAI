"""Première transcription de référence avec faster-whisper (small, INT8, CPU).

Le fichier par défaut est un dialogue français déjà dans le corpus.
La référence est le texte du manifeste, limité à la durée annotée.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safeai.transcription.config import TranscriptionConfig  # noqa: E402
from safeai.transcription.scoring import normalize_french, word_error_rate  # noqa: E402
from safeai.transcription.whisper import WhisperTranscriber  # noqa: E402

MANIFEST = ROOT / "data" / "conversations_fr" / "manifest.json"
DEFAULT_FILE = ROOT / "data" / "conversations_fr" / "audio" / "fr_accueil_ubs_02_24s.wav"


def _reference_for(path: Path) -> tuple[str, float] | None:
    if not MANIFEST.is_file():
        return None
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    target = path.resolve()
    for item in manifest["files"]:
        if (ROOT / item["path"]).resolve() != target:
            continue
        turns = sorted(item["gold_turns"], key=lambda turn: turn["start_s"])
        if not turns:
            return None
        text = " ".join(turn["text"] for turn in turns)
        return text, float(turns[-1]["end_s"])
    return None


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Transcription locale faster-whisper.")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--model", default="small")
    parser.add_argument("--full", action="store_true", help="Transcrire tout le fichier, pas seulement la partie annotée.")
    args = parser.parse_args(argv)
    if not args.file.is_file():
        print(f"Fichier introuvable : {args.file}", file=sys.stderr)
        return 2

    reference = _reference_for(args.file)
    duration_s = None if args.full or reference is None else reference[1]
    transcriber = WhisperTranscriber(
        TranscriptionConfig(model_size=args.model, device="cpu", compute_type="int8", vad_filter=False)
    )
    segments = transcriber.transcribe_file(
        args.file,
        session_id="whisper_baseline",
        source_segment_id=args.file.stem,
        duration_s=duration_s,
    )
    hypothesis = " ".join(segment.text for segment in segments)
    for segment in segments:
        print(f"[{segment.start_s:.2f} - {segment.end_s:.2f}] {segment.text}")
    print(f"\nSegments : {len(segments)}  speaker_id : null  vad_filter : false")
    if reference is None:
        print("Pas de référence dans le manifeste : WER non calculé.")
        return 0
    ref_text, annotated_end = reference
    score = word_error_rate(ref_text, hypothesis)
    print(f"Durée comparée : 0 – {annotated_end:.2f} s")
    print(f"Référence : {normalize_french(ref_text)}")
    print(f"Hypothèse : {normalize_french(hypothesis)}")
    print(f"WER : {score:.3f}" if score is not None else "WER : référence vide")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
