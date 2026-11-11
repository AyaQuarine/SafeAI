"""Compare Whisper sur le fichier entier et Whisper sur les segments Silero.

Même audio coupé à la plage annotée, même référence corpus.
La référence n'a pas été corrigée à l'écoute : l'écart A/B isole la stratégie,
pas la qualité absolue.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safeai.acquisition.service import run_file  # noqa: E402
from safeai.acquisition.vad import build_vad  # noqa: E402
from safeai.transcription.config import TranscriptionConfig  # noqa: E402
from safeai.transcription.scoring import alignment_counts, normalize_french  # noqa: E402
from safeai.transcription.whisper import WhisperTranscriber  # noqa: E402
from safeai.transcription.windows import (  # noqa: E402
    covering_speech,
    pause_windows,
    sliding_windows,
    transcribe_windows,
)

MANIFEST = ROOT / "data" / "conversations_fr" / "manifest.json"
OUT = ROOT / "data" / "conversations_fr" / "transcription_compare.json"
MODEL = ROOT / ".cache" / "faster-whisper-small"
FILES = (
    "fr_reunion_mix_00_30s.wav",
    "fr_reunion_mix_30_60s.wav",
    "fr_reunion_mix_90_45s.wav",
    "fr_accueil_ubs_02_24s.wav",
)


def _clip(source: Path, duration_s: float) -> Path:
    samples, rate = sf.read(str(source), dtype="float32", always_2d=False)
    keep = min(len(samples), int(round(duration_s * rate)))
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    path = Path(handle.name)
    handle.close()
    sf.write(str(path), samples[:keep], rate, subtype="PCM_16")
    return path


def _text(segments) -> str:
    return " ".join(segment.text for segment in segments if segment.status == "completed" and segment.text)


def _measure(reference: str, hypothesis: str, started: float) -> dict:
    counts = alignment_counts(reference, hypothesis) or {}
    return {
        "wer": counts.get("wer"),
        "deletions": counts.get("deletions", 0),
        "substitutions": counts.get("substitutions", 0),
        "insertions": counts.get("insertions", 0),
        "hits": counts.get("hits", 0),
        "missed_words": counts.get("missed_words", []),
        "hypothesis": normalize_french(hypothesis),
        "compute_s": round(time.perf_counter() - started, 3),
    }


def _one(item: dict, transcriber: WhisperTranscriber) -> dict:
    source = ROOT / item["path"]
    turns = sorted(item["gold_turns"], key=lambda turn: turn["start_s"])
    annotated_end = max(turn["end_s"] for turn in turns)
    reference = " ".join(turn["text"] for turn in turns)
    clipped = _clip(source, annotated_end)
    try:
        started = time.perf_counter()
        full = transcriber.transcribe_file(
            clipped,
            session_id=source.stem,
            source_segment_id="file",
            duration_s=annotated_end,
        )
        method_a = _measure(reference, _text(full), started)
        method_a["segments"] = len([segment for segment in full if segment.status == "completed"])
        method_a["empty"] = len([segment for segment in full if segment.status == "empty"])
        method_a["publication_latency_s"] = method_a["compute_s"]

        pipeline = time.perf_counter()
        acquired = run_file(clipped, vad=build_vad())
        acquire_s = time.perf_counter() - pipeline
        pieces = []
        per_segment_s = []
        first_text_s = None
        for index, speech in enumerate(acquired.speech_segments, start=1):
            tick = time.perf_counter()
            produced = transcriber.transcribe_speech(speech, f"speech_{index:03d}")
            spent = time.perf_counter() - tick
            per_segment_s.append(round(spent, 3))
            pieces.extend(produced)
            if first_text_s is None and any(segment.status == "completed" for segment in produced):
                first_text_s = round(time.perf_counter() - pipeline, 3)
        method_b = _measure(reference, _text(pieces), pipeline)
        method_b["acquire_s"] = round(acquire_s, 3)
        method_b["speech_segments"] = len(acquired.speech_segments)
        method_b["segments"] = len([segment for segment in pieces if segment.status == "completed"])
        method_b["empty"] = len([segment for segment in pieces if segment.status == "empty"])
        method_b["publication_latency_mean_s"] = (
            round(sum(per_segment_s) / len(per_segment_s), 3) if per_segment_s else 0.0
        )
        method_b["publication_latency_max_s"] = max(per_segment_s) if per_segment_s else 0.0
        method_b["first_text_s"] = first_text_s
        outside = [
            segment
            for segment in pieces
            if segment.status == "completed"
            and not any(speech.start_s - 0.02 <= segment.start_s and segment.end_s <= speech.end_s + 0.02 for speech in acquired.speech_segments)
        ]
        samples, rate = sf.read(str(clipped), dtype="float32", always_2d=False)
        regions = [(speech.start_s, speech.end_s) for speech in acquired.speech_segments]
        methods_c = {}
        plans = {
            "method_c_window_12s": sliding_windows(annotated_end, 12.0, 2.0),
            "method_c_window_4s": sliding_windows(annotated_end, 4.0, 1.0),
            "method_c_pauses": pause_windows(regions, max_window_s=12.0, max_pause_s=0.6),
        }
        for name, windows in plans.items():
            selected = windows if name == "method_c_pauses" else covering_speech(windows, regions)
            started_c = time.perf_counter()
            produced = transcribe_windows(
                transcriber,
                samples,
                selected,
                session_id=source.stem,
                sample_rate=rate,
            )
            measured = _measure(reference, _text(produced), started_c)
            measured["windows"] = len(selected)
            measured["segments"] = len(produced)
            methods_c[name] = measured
    finally:
        clipped.unlink(missing_ok=True)

    return {
        "file": source.name,
        "kind": item["kind"],
        "meeting_id": item.get("meeting_id"),
        "annotated_s": round(annotated_end, 3),
        "reference": normalize_french(reference),
        "timestamps_outside_speech": len(outside),
        "method_a_full_file": method_a,
        "method_b_silero_segments": method_b,
        **methods_c,
    }


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    chosen = []
    for name in FILES:
        match = [item for item in manifest["files"] if item["path"].endswith(name)]
        if not match or not (ROOT / match[0]["path"]).is_file():
            print(f"ABSENT {name}")
            continue
        chosen.append(match[0])
    if not chosen:
        print("Aucun audio français local.")
        return 2
    transcriber = WhisperTranscriber(
        TranscriptionConfig(model_size=str(MODEL), device="cpu", compute_type="int8", vad_filter=False, language="fr", beam_size=5)
    )
    rows = [_one(item, transcriber) for item in chosen]
    payload = {
        "model": "faster-whisper small",
        "device": "cpu",
        "compute_type": "int8",
        "language": "fr",
        "beam_size": 5,
        "vad_filter": False,
        "vad": "silero",
        "vad_threshold": 0.5,
        "reference": "texte corpus SUMM-RE / Accueil_UBS, non corrigé à l'écoute",
        "scope": "audio limité à la fin de la dernière annotation",
        "method_c": {
            "window_12s": "fenêtre 12 s, chevauchement 2 s, hypothèse non validée",
            "window_4s": "fenêtre 4 s, chevauchement 1 s",
            "pauses": "regroupement sur pauses <= 0,6 s, plafond 12 s",
        },
        "files": rows,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'fichier':32} {'A':6} {'B':6} {'C12':6} {'C4':6} {'Cp':6}")
    for row in rows:
        print(
            f"{row['file']:32} "
            f"{row['method_a_full_file']['wer']:.3f}  "
            f"{row['method_b_silero_segments']['wer']:.3f}  "
            f"{row['method_c_window_12s']['wer']:.3f}  "
            f"{row['method_c_window_4s']['wer']:.3f}  "
            f"{row['method_c_pauses']['wer']:.3f}"
        )
    print(f"\nEcrit {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
