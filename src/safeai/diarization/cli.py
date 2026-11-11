"""Interface locale de diagnostic de la diarisation hors ligne."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from safeai.diarization.config import load_config
from safeai.diarization.errors import DiarizationError
from safeai.diarization.service import DiarizationService


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        prog="safeai-diarization",
        description="SafeAI — diarisation hors ligne (module local, non médical).",
    )
    parser.add_argument("--config", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Diariser un WAV complet.")
    inspect.add_argument("--file", type=Path, required=True)
    inspect.add_argument("--json", action="store_true")
    inspect.add_argument("--json-out", type=Path, default=None)
    inspect.add_argument("--rttm-out", type=Path, default=None)
    inspect.set_defaults(func=cmd_inspect)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DiarizationError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2


def cmd_inspect(args: argparse.Namespace) -> int:
    service = DiarizationService(config=load_config(args.config))
    result = service.run_file(args.file)
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Source             : {args.file}")
        print(f"Duration           : {result.duration_s:.2f} s")
        print(f"Offset             : {result.offset_s:.2f} s")
        print(f"Status             : {result.status}")
        print(f"Speakers estimated : {result.speakers_estimated}")
        print(f"Turns              : {len(result.turns)}")
        print(f"Overlaps           : {len(result.overlapping_turns)}")
        print(f"Unknown            : {len(result.unknown_turns)}")
        if result.processing_seconds is not None:
            print(f"Processing         : {result.processing_seconds:.2f} s")
        if result.real_time_factor is not None:
            print(f"RTF                : {result.real_time_factor:.2f}")
        print(f"Model              : {result.model_id}")
        for turn in result.turns:
            speaker = turn.speaker_id or "unknown"
            flag = " overlap" if turn.overlap else ""
            print(f"{turn.start_s:7.2f}-{turn.end_s:<7.2f} {speaker}{flag}")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON               : {args.json_out}")
    if args.rttm_out is not None:
        args.rttm_out.parent.mkdir(parents=True, exist_ok=True)
        args.rttm_out.write_text(result.to_rttm(args.file.stem), encoding="utf-8")
        print(f"RTTM               : {args.rttm_out}")
    return 0
