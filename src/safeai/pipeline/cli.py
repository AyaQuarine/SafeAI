"""Interface locale du pipeline commun hors ligne."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from safeai.alignment.contracts import AlignedSession
from safeai.pipeline.errors import PipelineError
from safeai.pipeline.offline import CommonBundle, CommonPipeline

DEFAULT_WHISPER_CACHE = Path(".cache") / "faster-whisper-small"


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        prog="safeai-pipeline",
        description="SafeAI — pipeline commun hors ligne (module local, non médical).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Analyser un WAV complet vers AlignedSession.")
    inspect.add_argument("--file", type=Path, required=True)
    inspect.add_argument("--out", type=Path, default=None, help="JSON AlignedSession")
    inspect.add_argument(
        "--diarization-out",
        type=Path,
        default=None,
        help="JSON DiarizationResult (tours et chevauchements, même session_id)",
    )
    inspect.add_argument("--session-id", type=str, default=None)
    inspect.add_argument(
        "--whisper-model",
        type=str,
        default=None,
        help="Taille Hugging Face ou chemin local (ex. .cache/faster-whisper-small)",
    )
    inspect.add_argument("--json", action="store_true")
    inspect.set_defaults(func=cmd_inspect)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except PipelineError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2


def cmd_inspect(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    pipeline = build_default_pipeline(whisper_model=args.whisper_model)
    bundle = pipeline.run_bundle(args.file, session_id=args.session_id)
    elapsed = time.perf_counter() - started
    session = bundle.aligned
    _print_session(args.file, session, as_json=args.json, elapsed_s=elapsed)
    if bundle.diarization is not None:
        print(f"Diarization turns  : {len(bundle.diarization.turns)}")
        print(f"Diarization overlap: {len(bundle.diarization.overlapping_turns)}")
    out_aligned = args.out
    out_diar = args.diarization_out
    if out_aligned is not None and out_diar is None:
        out_diar = out_aligned.with_name(f"{out_aligned.stem}_diarization.json")
    if out_aligned is not None:
        out_aligned.parent.mkdir(parents=True, exist_ok=True)
        out_aligned.write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON               : {out_aligned}")
    if out_diar is not None and bundle.diarization is not None:
        out_diar.parent.mkdir(parents=True, exist_ok=True)
        payload = bundle.diarization.to_dict()
        out_diar.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Diarization JSON   : {out_diar}")
    return 0 if session.status != "failed" else 2


def resolve_whisper_model(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if DEFAULT_WHISPER_CACHE.is_dir():
        return str(DEFAULT_WHISPER_CACHE)
    return "small"


def build_default_pipeline(*, whisper_model: str | None = None) -> CommonPipeline:
    """Construit le pipeline avec les adaptateurs locaux disponibles.

    Community-1 doit être installé dans l'environnement courant
    (souvent ``.venv-diarization``). Aucun jeton n'est lu ici explicitement :
    l'adaptateur pyannote utilise le cache Hugging Face local.
    """
    from safeai.diarization.service import DiarizationService
    from safeai.pipeline.offline import (
        DiarizationServiceAdapter,
        LocalAcquisition,
        WhisperFileTranscriber,
    )
    from safeai.transcription.config import TranscriptionConfig
    from safeai.transcription.whisper import WhisperTranscriber

    model = resolve_whisper_model(whisper_model)
    return CommonPipeline(
        LocalAcquisition(),
        WhisperFileTranscriber(
            WhisperTranscriber(TranscriptionConfig(model_size=model))
        ),
        DiarizationServiceAdapter(DiarizationService()),
    )


def _print_session(
    path: Path,
    session: AlignedSession,
    *,
    as_json: bool,
    elapsed_s: float | None = None,
) -> None:
    payload = session.to_dict()
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    speakers = sorted(
        {
            item.speaker_id
            for item in session.utterances
            if item.speaker_id is not None
        }
    )
    print(f"Source             : {path}")
    print(f"Session            : {session.session_id}")
    print(f"Duration           : {session.duration_s:.2f} s")
    print(f"Status             : {session.status}")
    print(f"Utterances         : {len(session.utterances)}")
    print(f"Speakers           : {len(speakers)}")
    print(f"Provenance         : {session.provenance}")
    if elapsed_s is not None:
        print(f"Wall time          : {elapsed_s:.2f} s")
    if session.error:
        print(f"Error              : {session.error}")
    for note in session.diagnostics:
        print(f"Diagnostic         : {note}")
    for item in session.utterances:
        speaker = item.speaker_id or "unknown"
        flag = f" {item.attribution}"
        if item.overlap:
            flag += " overlap"
        print(f"{item.start_s:7.2f}-{item.end_s:<7.2f} {speaker}{flag}  {item.text}")
