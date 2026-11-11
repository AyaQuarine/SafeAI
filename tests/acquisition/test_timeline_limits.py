"""Cas limites SC11 : trou de flux et silence de fin de fenêtre."""

from __future__ import annotations

import numpy as np
import pytest

from safeai.acquisition.config import AcquisitionConfig, VADSettings
from safeai.acquisition.contracts import (
    AudioChunk,
    MuteStatus,
    SourceMetadata,
    SpeechActivity,
    StreamStatus,
)
from safeai.acquisition.quality import QualityMonitor
from safeai.acquisition.segmenter import AudioSegmenter
from safeai.acquisition.service import run_synthetic
from safeai.acquisition.vad import EnergyVADDetector


def _chunk(samples: np.ndarray, start_s: float, end_s: float) -> AudioChunk:
    return AudioChunk(
        session_id="limits",
        source_type="synthetic",
        start_s=start_s,
        end_s=end_s,
        sample_rate=16000,
        samples=np.asarray(samples, dtype=np.float32),
    )


def _longest_zero_run(samples: np.ndarray) -> int:
    best = current = 0
    for value in samples:
        if value == 0.0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def test_missing_second_is_a_loss_not_inserted_silence() -> None:
    segmenter = AudioSegmenter(
        VADSettings(
            threshold=0.5,
            min_speech_duration_s=0.15,
            min_silence_duration_s=0.4,
            speech_pad_s=0.15,
        )
    )
    tone = np.full(16000, 0.25, dtype=np.float32)
    frames_on = [
        type("F", (), {"start_s": t, "end_s": t + 0.032, "probability": 0.9})()
        for t in np.arange(0.0, 1.0, 0.032)
    ]
    segmenter.ingest(_chunk(tone, 0.0, 1.0), frames_on)
    frames_later = [
        type("F", (), {"start_s": 2.0 + t, "end_s": 2.0 + t + 0.032, "probability": 0.9})()
        for t in np.arange(0.0, 1.0, 0.032)
    ]
    _, _, segments = segmenter.ingest(_chunk(tone, 2.0, 3.0), frames_later)
    gaps = segmenter.take_gaps()
    assert len(gaps) == 1
    assert gaps[0].samples_missing == 16000
    assert gaps[0].start_s == pytest.approx(1.0)
    assert gaps[0].end_s == pytest.approx(2.0)
    assert segments, "La parole ouverte doit être fermée au trou, pas prolongée par des zéros"
    for segment in segments:
        assert segment.samples is not None
        assert _longest_zero_run(segment.samples) < 16000
        assert segment.end_s <= 1.0 + 1e-3


def test_service_records_timeline_gap_without_growing_silence() -> None:
    class Gapped:
        session_id = "gap"
        metadata = SourceMetadata(source_type="synthetic", original_sample_rate=16000, original_channels=1, duration_s=3.0)

        def read_chunks(self):
            tone = np.full(16000, 0.25, dtype=np.float32)
            yield _chunk(tone, 0.0, 1.0)
            yield _chunk(np.zeros(16000, dtype=np.float32), 2.0, 3.0)

        def close(self) -> None:
            return None

    config = AcquisitionConfig(vad=VADSettings(backend="energy"))
    result = run_synthetic(Gapped(), config=config, vad=EnergyVADDetector(config.vad))
    gap_events = [event for event in result.quality_events if event.reason == "timeline_gap"]
    assert len(gap_events) == 1
    assert gap_events[0].samples_dropped == 16000
    assert gap_events[0].speech_activity is SpeechActivity.UNKNOWN
    assert gap_events[0].stream_status is StreamStatus.INTERRUPTED
    assert any("trou de flux" in item for item in result.incidents)
    silence_after = [
        event
        for event in result.quality_events
        if event.start_s >= 2.0 and event.speech_activity is SpeechActivity.NON_SPEECH
    ]
    assert silence_after
    assert max(event.silence_duration_s for event in silence_after) < 1.5


def test_trailing_silence_inside_speech_window_is_counted() -> None:
    monitor = QualityMonitor()
    samples = np.zeros(16000, dtype=np.float32)
    samples[int(0.2 * 16000) : int(0.4 * 16000)] = 0.2
    event = monitor.observe(
        _chunk(samples, 0.0, 1.0),
        SpeechActivity.SPEECH,
        speech_active_at_end=False,
    )
    assert event.silence_duration_s == pytest.approx(0.6, abs=0.03)
    assert event.mute_status is MuteStatus.UNKNOWN

    following = monitor.observe(
        _chunk(np.zeros(16000, dtype=np.float32), 1.0, 2.0),
        SpeechActivity.NON_SPEECH,
        speech_active_at_end=False,
    )
    assert following.silence_duration_s == pytest.approx(1.6, abs=0.03)
