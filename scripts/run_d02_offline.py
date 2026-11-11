"""Lance R01, R02 et R06 hors ligne. Écrit JSON et RTTM dans data/local_dev."""

from __future__ import annotations

from pathlib import Path

from safeai.diarization.errors import DiarizationError
from safeai.diarization.service import DiarizationService

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "local_dev"
FILES = (
    ("r01_officiel.wav", "d02_r01"),
    ("r02_silence.wav", "d02_r02"),
    ("r06_deux_locuteurs.wav", "d02_r06"),
)


def main() -> int:
    service = DiarizationService()
    failures = 0
    for wav_name, stem in FILES:
        wav = OUT / wav_name
        print(f"=== {wav_name} ===")
        try:
            result = service.run_file(wav, session_id=stem)
        except DiarizationError as exc:
            print(f"Erreur : {exc}")
            failures += 1
            continue
        json_path = OUT / f"{stem}.json"
        rttm_path = OUT / f"{stem}.rttm"
        json_path.write_text(
            __import__("json").dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rttm_path.write_text(result.to_rttm(stem), encoding="utf-8")
        print(
            f"status={result.status} speakers={result.speakers_estimated} "
            f"turns={len(result.turns)} overlaps={len(result.overlapping_turns)} "
            f"processing_s={result.processing_seconds}"
        )
        print(f"JSON {json_path}")
        print(f"RTTM {rttm_path}")
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
