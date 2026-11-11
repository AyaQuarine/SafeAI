"""Interface en ligne de commande du module d'acquisition."""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
from pathlib import Path

from safeai.acquisition.config import load_config
from safeai.acquisition.contracts import SourceConfig, TARGET_SAMPLE_RATE
from safeai.acquisition.errors import AcquisitionError
from safeai.acquisition.service import PipelineResult, run_file, run_synthetic
from safeai.acquisition.signals import demo_meeting_signal
from safeai.acquisition.sources import SyntheticSource, list_input_devices
from safeai.acquisition.vad import EnergyVADDetector, build_vad


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except AcquisitionError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safeai-acquisition",
        description="SafeAI — acquisition audio, qualité et VAD (module local, non médical).",
    )
    parser.add_argument("--config", type=Path, default=None, help="Fichier YAML de configuration.")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Analyser un fichier ou un microphone.")
    source = inspect.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path, help="Chemin d'un WAV local.")
    source.add_argument("--device", help="Identifiant du microphone.")
    inspect.add_argument("--consent", action="store_true", help="Consentement de capture (obligatoire pour le micro).")
    inspect.add_argument("--duration", type=float, default=8.0, help="Durée de capture micro en secondes.")
    inspect.add_argument(
        "--record",
        type=Path,
        default=None,
        help="Écrire le WAV 16 kHz capté au microphone. Ignoré pour un fichier.",
    )
    inspect.add_argument("--json", action="store_true", help="Sortie JSON (sans échantillons bruts).")
    inspect.add_argument("--energy-vad", action="store_true", help="Forcer le VAD énergétique.")
    inspect.set_defaults(func=cmd_inspect)

    devices = sub.add_parser("devices", help="Lister les microphones détectés.")
    devices.set_defaults(func=cmd_devices)

    demo = sub.add_parser("demo", help="Exécuter le scénario synthétique de diagnostic.")
    demo.add_argument("--json", action="store_true")
    demo.set_defaults(func=cmd_demo)

    ui = sub.add_parser("ui", help="Ouvrir l'interface de diagnostic locale.")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8765)
    ui.set_defaults(func=cmd_ui)

    export = sub.add_parser(
        "export-segments",
        help="Ecrire les SpeechSegment d'un fichier en WAV pour reecoute locale.",
    )
    export.add_argument("--file", type=Path, required=True)
    export.add_argument("--out", type=Path, default=Path("data/local_dev/segments"))
    export.add_argument("--energy-vad", action="store_true")
    export.set_defaults(func=cmd_export_segments)
    return parser


def cmd_inspect(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    vad = EnergyVADDetector(config.vad) if args.energy_vad else build_vad(config.vad)
    if args.record is not None and args.file is not None:
        print("Erreur : --record sert à une capture microphone, pas à un fichier déjà écrit.", file=sys.stderr)
        return 2
    if args.file:
        result = run_file(args.file, config=config, vad=vad)
        _print_report(result, as_json=args.json, source_label=str(args.file))
        return 0

    from safeai.acquisition.service import AcquisitionService

    if not args.consent:
        print("Erreur : --consent est obligatoire pour une capture microphone.", file=sys.stderr)
        return 2
    service = AcquisitionService(config=config, vad=vad)
    result = _capture_microphone(
        service,
        device_id=str(args.device),
        duration_s=max(1.0, args.duration),
        chunk_duration_s=config.chunk_duration_s,
        record_path=args.record,
    )
    _print_report(result, as_json=args.json, source_label=f"microphone:{args.device}")
    if args.record is not None:
        print(f"Enregistrement      : {args.record}")
    return 0


def _capture_microphone(service, device_id: str, duration_s: float, chunk_duration_s: float, record_path: Path | None):
    """Démarre la capture, puis seulement ensuite l'écriture du WAV.

    Si le microphone refuse de démarrer, aucun fichier n'est ouvert.
    L'arrêt du service est terminé avant le signal d'arrêt du thread d'écriture,
    afin que le dernier bloc publié soit encore écrit. Le thread et le fichier
    sont toujours fermés, y compris en cas d'erreur.
    """
    writer: threading.Thread | None = None
    handle = None
    stop_write = threading.Event()
    audio_q = service.subscribe_audio() if record_path is not None else None
    try:
        service.start(
            SourceConfig(
                source_type="microphone",
                device_id=device_id,
                recording_consent=True,
                chunk_duration_s=chunk_duration_s,
            )
        )
        if record_path is not None and audio_q is not None:
            import soundfile as sf

            record_path.parent.mkdir(parents=True, exist_ok=True)
            handle = sf.SoundFile(
                str(record_path),
                mode="w",
                samplerate=TARGET_SAMPLE_RATE,
                channels=1,
                subtype="FLOAT",
            )

            def drain() -> None:
                assert handle is not None
                while True:
                    try:
                        chunk = audio_q.get(timeout=0.2)
                    except queue.Empty:
                        if stop_write.is_set():
                            break
                        continue
                    handle.write(chunk.samples)

            writer = threading.Thread(target=drain, name="safeai-record", daemon=True)
            writer.start()
        time.sleep(duration_s)
    finally:
        try:
            service.stop()
        finally:
            stop_write.set()
            if writer is not None:
                writer.join(timeout=5.0)
            if handle is not None and audio_q is not None:
                while True:
                    try:
                        chunk = audio_q.get_nowait()
                    except queue.Empty:
                        break
                    handle.write(chunk.samples)
            if handle is not None:
                handle.close()
    return service.collect_result()


def cmd_devices(_args: argparse.Namespace) -> int:
    devices = list_input_devices()
    if not devices:
        print("Aucun microphone n'a été détecté, ou sounddevice ne peut pas interroger le système.")
        return 0
    for device in devices:
        print(
            f"{device['id']:>4}  {device['name']}  "
            f"({device['max_input_channels']} ch, {device['default_samplerate']:.0f} Hz)"
        )
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    samples = demo_meeting_signal(TARGET_SAMPLE_RATE)
    source = SyntheticSource(samples, session_id="session_demo", chunk_duration_s=config.chunk_duration_s)
    result = run_synthetic(source, config=config, vad=EnergyVADDetector(config.vad))
    _print_report(result, as_json=args.json, source_label="synthetic-demo")
    return 0


def cmd_export_segments(args: argparse.Namespace) -> int:
    import soundfile as sf

    config = load_config(args.config)
    vad = EnergyVADDetector(config.vad) if args.energy_vad else build_vad(config.vad)
    result = run_file(args.file, config=config, vad=vad)
    args.out.mkdir(parents=True, exist_ok=True)
    if not result.speech_segments:
        print("Aucun SpeechSegment n'a ete produit.")
        return 0
    for index, segment in enumerate(result.speech_segments):
        segment.validate()
        name = f"segment_{index:03d}_{segment.start_s:.2f}-{segment.end_s:.2f}.wav"
        path = args.out / name
        if segment.samples is None:
            print(f"IGNORE {name}: audio_available={segment.audio_available}")
            continue
        sf.write(str(path), segment.samples, segment.sample_rate)
        duration = len(segment.samples) / segment.sample_rate
        print(
            f"{path.name}  start={segment.start_s:.3f}s  end={segment.end_s:.3f}s  "
            f"samples={len(segment.samples)}  duration={duration:.3f}s"
        )
    print(f"VAD backend        : {result.vad_backend}")
    print(f"Speech alignment   : {result.speech_alignment_errors}")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    import uvicorn

    from safeai.ui.app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
    return 0


def _print_report(result: PipelineResult, as_json: bool, source_label: str) -> None:
    if as_json:
        print(json.dumps(serialize_result(result, source_label), ensure_ascii=False, indent=2))
        return

    meta = result.metadata
    duration = result.chunks[-1].end_s if result.chunks else 0.0
    print(f"Source             : {source_label}")
    print(f"Input rate         : {meta.original_sample_rate} Hz")
    print(f"Output rate        : {TARGET_SAMPLE_RATE} Hz")
    print(f"Channels           : {meta.original_channels} -> 1 ({meta.downmix})")
    print(f"Duration           : {duration:.2f} s")
    print(f"Chunks published   : {result.chunks_published}")
    print(f"Chunks retained    : {len(result.chunks)}")
    print(f"Timestamp errors   : {result.timestamp_errors_total}")
    print(f"Speech alignment   : {result.speech_alignment_errors}")
    print(f"Signal status      : {result.quality_events[-1].stream_status.value if result.quality_events else 'UNKNOWN'}")
    print(f"VAD backend        : {result.vad_backend}")
    print(f"Speech segments    : {len(result.speech_segments)}")
    print(f"Publisher drops    : {result.publisher_drops}")
    if result.incidents:
        print("Incidents          :")
        for incident in result.incidents:
            print(f"  - {incident}")
    if result.timeline:
        print()
        print("Segments émis vers la suite du pipeline")
        print(f"{'Intervalle':<16} {'Type':<10} {'Qualité':<12} Motif")
        for row in result.timeline:
            print(
                f"{row.start_s:4.1f}-{row.end_s:<6.1f} "
                f"{row.to_dict()['label']:<10} {row.quality_status.value:<12} {row.reason}"
            )


def serialize_result(result: PipelineResult, source_label: str | None = None) -> dict[str, object]:
    duration = result.chunks[-1].end_s if result.chunks else 0.0
    latencies = result.processing_latency_s
    return {
        "source": source_label or result.metadata.file_path or result.metadata.device_name or result.metadata.source_type,
        "session_id": result.session.session_id,
        "started_at_utc": result.session.started_at_utc.isoformat(),
        "input_rate": result.metadata.original_sample_rate,
        "output_rate": TARGET_SAMPLE_RATE,
        "original_channels": result.metadata.original_channels,
        "downmix": result.metadata.downmix,
        "duration_s": duration,
        "chunks_published": result.chunks_published,
        "chunks_retained": len(result.chunks),
        "chunks_generated": result.chunks_published,
        "timestamp_errors": result.timestamp_errors_total,
        "speech_alignment_errors": result.speech_alignment_errors,
        "publisher_drops": result.publisher_drops,
        "vad_backend": result.vad_backend,
        "signal_status": result.quality_events[-1].stream_status.value if result.quality_events else "UNKNOWN",
        "incidents": result.incidents,
        "latency": {
            "median_s": _percentile(latencies, 50),
            "p95_s": _percentile(latencies, 95),
            "processing_median_s": _percentile(latencies, 50),
            "processing_p95_s": _percentile(latencies, 95),
            "vad_median_s": _percentile(result.vad_processing_s, 50),
            "capture_to_publish_median_s": _percentile(result.capture_to_publish_s, 50),
            "max_s": max(latencies) if latencies else 0.0,
        },
        "speech_segments": [
            {
                "session_id": segment.session_id,
                "start_s": segment.start_s,
                "end_s": segment.end_s,
                "speech_detected": segment.speech_detected,
                "vad_mean_probability": segment.vad_mean_probability,
                "audio_available": segment.audio_available,
            }
            for segment in result.speech_segments
        ],
        "quality_events": [event.to_dict() for event in result.quality_events],
        "timeline": [row.to_dict() for row in result.timeline],
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((q / 100) * (len(ordered) - 1)))))
    return ordered[index]
