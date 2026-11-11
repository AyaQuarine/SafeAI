"""Tests optionnels sur conversations françaises réelles (hors Git).

Les WAV sont locaux : pytest saute si le manifeste ou l'audio est absent.
Relancer : python scripts/fetch_french_conversations.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import soundfile as sf

from safeai.acquisition.config import AcquisitionConfig
from safeai.acquisition.contracts import TARGET_SAMPLE_RATE
from safeai.acquisition.eval import Interval, frame_metrics, segments_to_intervals
from safeai.acquisition.errors import ConfigurationError
from safeai.acquisition.service import run_file
from safeai.acquisition.vad import build_vad

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data" / "conversations_fr" / "manifest.json"


def _load_manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        pytest.skip("Manifeste conversations_fr absent.")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _entries(kind: str | None = None) -> list[dict]:
    files = _load_manifest()["files"]
    if kind:
        files = [item for item in files if item["kind"] == kind]
    available = []
    for item in files:
        path = ROOT / item["path"]
        if path.is_file():
            available.append({**item, "abs_path": path})
    if not available:
        pytest.skip(f"Audio conversations_fr manquant ({kind or 'tous'}). Relancer fetch_french_conversations.py")
    return available


@pytest.fixture(scope="module")
def silero_vad():
    try:
        return build_vad(AcquisitionConfig().vad)
    except ConfigurationError as exc:
        pytest.skip(f"Silero indisponible : {exc}")


def test_manifest_is_french_only_and_multi_speaker() -> None:
    manifest = _load_manifest()
    assert manifest["priority"] == "fr"
    files = manifest["files"]
    assert files
    assert all(item["language"] == "fr" for item in files)
    assert all("en" not in item["language"].lower() for item in files)
    mixes = [item for item in files if item["kind"] == "reunion_mix"]
    dialogues = [item for item in files if item["kind"] == "dialogue_telephone"]
    assert mixes, "Il faut au moins un mix de reunion FR"
    assert dialogues, "Il faut au moins un dialogue telephone FR"
    assert all(item["speakers"] >= 3 for item in mixes)
    assert all(item["speakers"] >= 2 for item in dialogues)


def test_french_wavs_are_16k_mono() -> None:
    for item in _entries():
        info = sf.info(str(item["abs_path"]))
        assert info.samplerate == TARGET_SAMPLE_RATE
        assert info.channels == 1
        assert info.duration == pytest.approx(item["duration_s"], abs=0.05)


def test_silero_detects_speech_on_french_meeting(silero_vad) -> None:
    """Plancher de fonctionnement, pas un critère de qualité.

    Les valeurs mesurées sont dans data/conversations_fr/vad_metrics.json.
    """
    item = _entries("reunion_mix")[0]
    result = run_file(item["abs_path"], vad=silero_vad)
    assert result.timestamp_errors == 0
    assert result.speech_segments
    metrics = _annotated_metrics(item, result.speech_segments)
    assert metrics.recall >= 0.55
    assert metrics.precision >= 0.45


def test_silero_detects_speech_on_french_phone_dialogue(silero_vad) -> None:
    item = _entries("dialogue_telephone")[0]
    result = run_file(item["abs_path"], vad=silero_vad)
    assert result.timestamp_errors == 0
    assert result.speech_segments
    metrics = _annotated_metrics(item, result.speech_segments)
    assert metrics.recall >= 0.50


def _annotated_metrics(item: dict, segments) -> object:
    gold = [Interval(turn["start_s"], turn["end_s"]) for turn in item["gold_speech"]]
    annotated_end = max(turn.end_s for turn in gold)
    predicted = [
        interval
        for interval in segments_to_intervals(segments)
        if interval.start_s < annotated_end
    ]
    clipped = [
        Interval(interval.start_s, min(interval.end_s, annotated_end)) for interval in predicted
    ]
    return frame_metrics(gold, clipped, duration_s=annotated_end)
