from __future__ import annotations

import numpy as np

from safeai.acquisition.contracts import (
    AudioChunk,
    MuteStatus,
    QualityStatus,
    SpeechActivity,
    StreamStatus,
    TARGET_SAMPLE_RATE,
)
from safeai.acquisition.quality import QualityMonitor
from safeai.acquisition.signals import background_noise, clipped_signal, intense_clean, silence, speech_like
from safeai.acquisition.vad import EnergyVADDetector


def _chunk(samples: np.ndarray, start_s: float = 0.0, **kwargs) -> AudioChunk:
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    chunk = AudioChunk(
        session_id="session_q",
        source_type="synthetic",
        start_s=start_s,
        end_s=start_s + len(samples) / TARGET_SAMPLE_RATE,
        sample_rate=TARGET_SAMPLE_RATE,
        samples=samples,
        **kwargs,
    )
    chunk.validate()
    return chunk


def test_clipping_is_detected() -> None:
    event = QualityMonitor().observe(_chunk(clipped_signal(1.0)), SpeechActivity.SPEECH)
    assert event.clipping_ratio > 0.01
    assert event.quality_status in {QualityStatus.DEGRADED, QualityStatus.UNUSABLE}
    assert event.reason == "clipping"


def test_silence_is_not_reported_as_technical_failure() -> None:
    event = QualityMonitor().observe(_chunk(silence(1.0)), SpeechActivity.NON_SPEECH)
    assert event.quality_status is QualityStatus.OK
    assert event.mute_status is MuteStatus.UNKNOWN
    assert event.stream_status is StreamStatus.ACTIVE
    assert event.reason == "silence"


def test_near_zero_without_mute_info_stays_unknown() -> None:
    event = QualityMonitor().observe(_chunk(silence(1.0) + 1e-6), SpeechActivity.NON_SPEECH)
    assert event.mute_status is MuteStatus.UNKNOWN
    assert event.stream_status is StreamStatus.ACTIVE


def test_confirmed_mute_is_not_confused_with_silence() -> None:
    chunk = _chunk(silence(1.0), mute_status=MuteStatus.MUTED)
    event = QualityMonitor().observe(chunk, SpeechActivity.NON_SPEECH)
    assert event.mute_status is MuteStatus.MUTED
    assert event.quality_status is QualityStatus.UNUSABLE
    assert event.reason == "microphone_muted"


def test_unavailable_stream_is_unusable() -> None:
    chunk = _chunk(silence(1.0), stream_status=StreamStatus.UNAVAILABLE)
    event = QualityMonitor().observe(chunk, SpeechActivity.UNKNOWN)
    assert event.stream_status is StreamStatus.UNAVAILABLE
    assert event.quality_status is QualityStatus.UNUSABLE


def test_intense_clean_signal_is_not_a_quality_incident() -> None:
    event = QualityMonitor().observe(_chunk(intense_clean(1.0)), SpeechActivity.SPEECH)
    assert event.peak > 0.8
    assert event.clipping_ratio < 0.01
    assert event.quality_status is QualityStatus.OK


def test_background_noise_marks_degraded_quality() -> None:
    monitor = QualityMonitor()
    noise = background_noise(1.0, amplitude=0.09)
    monitor.observe(_chunk(noise), SpeechActivity.NON_SPEECH)
    event = monitor.observe(_chunk(np.clip(speech_like(1.0) + noise, -1, 1), start_s=1.0), SpeechActivity.SPEECH)
    assert event.quality_status is QualityStatus.DEGRADED
    assert event.reason == "background_noise"


def test_energy_vad_separates_speech_and_silence() -> None:
    vad = EnergyVADDetector()
    speech_frames = vad.process(_chunk(speech_like(1.0)))
    vad.reset()
    silent_frames = vad.process(_chunk(silence(1.0)))
    assert speech_frames
    assert silent_frames
    assert np.mean([frame.probability for frame in speech_frames]) > 0.5
    assert np.mean([frame.probability for frame in silent_frames]) < 0.2
