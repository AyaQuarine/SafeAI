"""Mesure VAD Silero sur le corpus français déjà présent. N'ajoute aucun fichier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safeai.acquisition.config import AcquisitionConfig  # noqa: E402
from safeai.acquisition.eval import Interval, frame_metrics, segments_to_intervals  # noqa: E402
from safeai.acquisition.service import run_file  # noqa: E402
from safeai.acquisition.vad import build_vad  # noqa: E402

MANIFEST = ROOT / "data" / "conversations_fr" / "manifest.json"
FRAME_S = 512 / 16000


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    vad = build_vad(AcquisitionConfig().vad)
    rows = []
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.is_file():
            print(f"ABSENT {item['path']}")
            continue
        duration = float(sf.info(str(path)).duration)
        gold = [Interval(turn["start_s"], turn["end_s"]) for turn in item["gold_speech"]]
        annotated_end = max(turn.end_s for turn in gold)
        result = run_file(path, vad=vad)
        predicted = [
            Interval(interval.start_s, min(interval.end_s, annotated_end))
            for interval in segments_to_intervals(result.speech_segments)
            if interval.start_s < annotated_end
        ]
        metrics = frame_metrics(gold, predicted, duration_s=annotated_end, frame_s=FRAME_S)
        gold_s = sum(turn.end_s - turn.start_s for turn in gold)
        rows.append(
            {
                "file": Path(item["path"]).name,
                "kind": item["kind"],
                "speakers": item["speakers"],
                "meeting_id": item.get("meeting_id"),
                "duration_s": round(duration, 3),
                "annotated_s": round(annotated_end, 3),
                "gold_speech_s": round(gold_s, 3),
                "frames": metrics.frames,
                "tp": metrics.true_positives,
                "fp": metrics.false_positives,
                "fn": metrics.false_negatives,
                "precision": round(metrics.precision, 4),
                "recall": round(metrics.recall, 4),
                "f1": round(metrics.f1, 4),
                "segments": len(result.speech_segments),
                "timestamp_errors": result.timestamp_errors,
            }
        )
        print(
            f"{rows[-1]['file']:32} P={metrics.precision:.3f} R={metrics.recall:.3f} "
            f"F1={metrics.f1:.3f} frames={metrics.frames} "
            f"tp={metrics.true_positives} fp={metrics.false_positives} fn={metrics.false_negatives} "
            f"dur={duration:.1f}s annot={annotated_end:.1f}s"
        )
    out = ROOT / "data" / "conversations_fr" / "vad_metrics.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nEcrit {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
