"""Rejoue un WAV au rythme réel dans le pipeline d'acquisition.

Sans option, le rapport mesure l'acquisition. Avec --transcribe, un
WhisperWorker publie le texte pendant la lecture et le délai est celui
qui sépare la fin du segment de parole de la publication du texte.
"""

from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safeai.acquisition.contracts import new_session_id  # noqa: E402
from safeai.acquisition.service import AcquisitionService  # noqa: E402
from safeai.acquisition.sources import LiveReplaySource  # noqa: E402


def _percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((percent / 100) * (len(ordered) - 1))
    return ordered[min(len(ordered) - 1, max(0, index))]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = [item for item in sys.argv[1:] if item != "--transcribe"]
    transcribe = "--transcribe" in sys.argv[1:]
    if len(args) != 1:
        print("Usage : python scripts/replay_live.py <fichier.wav> [--transcribe]")
        return 2
    path = Path(args[0])
    if not path.is_file():
        print(f"Fichier introuvable : {path}")
        return 2

    source = LiveReplaySource(path, session_id=new_session_id(), chunk_duration_s=1.0)
    service = AcquisitionService()
    worker = None
    if transcribe:
        from safeai.transcription.config import TranscriptionConfig
        from safeai.transcription.whisper import WhisperTranscriber
        from safeai.transcription.worker import WhisperWorker

        model = ROOT / ".cache" / "faster-whisper-small"
        worker = WhisperWorker(
            WhisperTranscriber(
                TranscriptionConfig(
                    model_size=str(model if model.is_dir() else "small"),
                    device="cpu",
                    compute_type="int8",
                    vad_filter=False,
                )
            ),
            service.subscribe_speech_segments(),
            clock=source.monotonic,
            session_origin_s=source.session_origin_s,
        )
        worker.start()
    tracemalloc.start()
    started = time.perf_counter()
    try:
        result = service.run_source(source)
    finally:
        if worker is not None:
            worker.finish(timeout_s=180.0)
    wall_s = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    duration_s = result.metadata.duration_s or 0.0
    ends = [chunk.end_s for chunk in result.chunks]
    monotonic = all(right > left for left, right in zip(ends, ends[1:]))
    print(f"fichier              {path.name}")
    print(f"duree_audio_s        {duration_s:.3f}")
    print(f"horloge_s            {wall_s:.3f}")
    print(f"blocs                {result.chunks_published}")
    print(f"blocs_en_memoire     {len(result.chunks)}")
    print(f"segments_parole      {result.speech_segments_published}")
    print(f"horodatages_continus {monotonic}")
    print(f"erreurs_temps        {result.timestamp_errors}")
    print(f"pertes_file          {result.publisher_drops}")
    print(f"incidents            {len(result.incidents)}")
    print(f"retard_emission_med_s {_percentile(source.emission_lag_s, 50):.3f}")
    print(f"retard_emission_p95_s {_percentile(source.emission_lag_s, 95):.3f}")
    print(f"traitement_med_s     {_percentile(result.capture_to_publish_s, 50):.3f}")
    print(f"traitement_p95_s     {_percentile(result.capture_to_publish_s, 95):.3f}")
    print(f"memoire_pic_Mio      {peak / (1024 * 1024):.1f}")
    print("vad                  " + result.vad_backend)
    if worker is not None:
        lags = list(worker.lag_s)
        print(f"textes               {worker.received_count}")
        print(f"echecs_transcription {worker.errors}")
        print(f"latence_texte_med_s  {_percentile(lags, 50):.3f}")
        print(f"latence_texte_p95_s  {_percentile(lags, 95):.3f}")
        preview = " ".join(item.segment.text for item in worker.published if item.segment.text).strip()
        print(f"apercu               {preview[:240]}")
    if "artificiel" in path.name:
        print("lecture              fichier long artificiel : stabilite logicielle, pas une reunion")
    for incident in result.incidents[:8]:
        print(f"incident             {incident}")
    return 0 if monotonic and result.timestamp_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
