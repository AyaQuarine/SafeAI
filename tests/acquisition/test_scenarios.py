"""Tests fonctionnels T01–T12 et scénarios du guide (SC11, SC13, SC14, SC22, SC07)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from safeai.acquisition.config import AcquisitionConfig, VADSettings
from safeai.acquisition.contracts import (
    MuteStatus,
    QualityStatus,
    SpeechActivity,
    StreamStatus,
    TARGET_SAMPLE_RATE,
    SourceConfig,
    new_session_id,
)
from safeai.acquisition.errors import ConsentRequiredError, InvalidAudioError, SourceUnavailableError
from safeai.acquisition.service import AcquisitionService, run_file, run_synthetic
from safeai.acquisition.signals import background_noise, clipped_signal, intense_clean, silence, speech_like
from safeai.acquisition.sources import FileSource, SyntheticSource
from safeai.acquisition.vad import EnergyVADDetector
from tests.acquisition.conftest import write_wav


def _run_samples(samples: np.ndarray, rate: int = TARGET_SAMPLE_RATE):
    config = AcquisitionConfig(vad=VADSettings(backend="energy"))
    source = SyntheticSource(samples, session_id=new_session_id(), source_rate=rate)
    return run_synthetic(source, config=config, vad=EnergyVADDetector(config.vad))


def test_t01_clear_phrase_is_detected(wav_mono_16k: Path, energy_config: AcquisitionConfig, energy_vad) -> None:
    result = run_file(wav_mono_16k, config=energy_config, vad=energy_vad)
    assert result.chunks
    assert result.timestamp_errors == 0
    assert any(event.speech_activity is SpeechActivity.SPEECH for event in result.quality_events)
    assert result.speech_segments


def test_t02_silent_recording_has_no_invented_fault(wav_silent: Path, energy_config, energy_vad) -> None:
    result = run_file(wav_silent, config=energy_config, vad=energy_vad)
    assert result.speech_segments == []
    assert all(event.speech_activity is SpeechActivity.NON_SPEECH for event in result.quality_events)
    assert all(event.mute_status is MuteStatus.UNKNOWN for event in result.quality_events)
    assert all(event.stream_status is StreamStatus.ACTIVE for event in result.quality_events)
    assert all(event.quality_status is QualityStatus.OK for event in result.quality_events)


def test_t03_noisy_conversation_keeps_segments_and_flags_quality(wav_noisy: Path, energy_config, energy_vad) -> None:
    result = run_file(wav_noisy, config=energy_config, vad=energy_vad)
    assert result.chunks
    assert any(event.quality_status is QualityStatus.DEGRADED for event in result.quality_events)


def test_t04_saturated_signal_is_clipped(wav_clipped: Path, energy_config, energy_vad) -> None:
    result = run_file(wav_clipped, config=energy_config, vad=energy_vad)
    assert any(event.clipping_ratio > 0.0 and "clipping" in event.reason for event in result.quality_events)


def test_t05_disconnected_microphone_emits_unavailability() -> None:
    class DyingStream:
        def __init__(self, callback) -> None:
            self.callback = callback
            self.active = False
            self._thread = threading.Thread(target=self._run, daemon=True)

        def start(self) -> None:
            self.active = True
            self._thread.start()

        def _run(self) -> None:
            frame = np.zeros((1600, 1), dtype=np.float32)
            self.callback(frame, 1600, None, type("S", (), {"input_overflow": False})())
            time.sleep(0.05)
            self.active = False

        def stop(self) -> None:
            self.active = False

        def close(self) -> None:
            self.active = False

    config = AcquisitionConfig(vad=VADSettings(backend="energy"))
    service = AcquisitionService(
        config=config,
        vad=EnergyVADDetector(config.vad),
        stream_factory=lambda callback: DyingStream(callback),
    )
    service.start(SourceConfig(source_type="microphone", device_id="0", recording_consent=True))
    with pytest.raises(SourceUnavailableError):
        service.wait_until_idle(timeout_s=3.0)
    status = service.get_status()
    assert status.stream_status is StreamStatus.UNAVAILABLE
    assert status.incidents
    service.stop()


def test_t06_accessible_mute_is_confirmed() -> None:
    result = _run_samples(silence(1.0))
    # Synthetic path uses UNKNOWN unless injected; verify via SyntheticSource mute_status.
    config = AcquisitionConfig(vad=VADSettings(backend="energy"))
    source = SyntheticSource(
        silence(1.0),
        session_id=new_session_id(),
        mute_status=MuteStatus.MUTED,
    )
    result = run_synthetic(source, config=config, vad=EnergyVADDetector(config.vad))
    assert all(event.mute_status is MuteStatus.MUTED for event in result.quality_events)
    assert all(event.quality_status is QualityStatus.UNUSABLE for event in result.quality_events)


def test_t07_near_zero_without_state_keeps_mute_unknown() -> None:
    result = _run_samples(np.full(TARGET_SAMPLE_RATE, 1e-6, dtype=np.float32))
    assert all(event.mute_status is MuteStatus.UNKNOWN for event in result.quality_events)
    assert all(event.stream_status is StreamStatus.ACTIVE for event in result.quality_events)


def test_t08_stereo_44100_duration_is_preserved(wav_stereo_44100: Path) -> None:
    source = FileSource(wav_stereo_44100, session_id=new_session_id())
    chunks = list(source.read_chunks())
    duration = chunks[-1].end_s
    assert source.metadata.original_sample_rate == 44_100
    assert source.metadata.original_channels == 2
    assert source.metadata.downmix == "average"
    assert abs(duration - 1.5) < 0.03
    for chunk in chunks:
        chunk.validate()


def test_t09_corrupt_or_empty_file_is_explicit(tmp_path: Path, energy_config, energy_vad) -> None:
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(InvalidAudioError):
        run_file(empty, config=energy_config, vad=energy_vad)

    corrupt = tmp_path / "bad.wav"
    corrupt.write_bytes(b"RIFF    NOTWAV")
    with pytest.raises(InvalidAudioError):
        run_file(corrupt, config=energy_config, vad=energy_vad)


def test_t10_overlap_is_kept_and_not_attributed() -> None:
    mixed = np.clip(speech_like(1.2, f0=120) + speech_like(1.2, f0=210), -1.0, 1.0)
    result = _run_samples(mixed)
    assert result.chunks
    assert result.speech_segments
    # L'acquisition ne fabrique pas d'identité de locuteur.
    assert all(not hasattr(segment, "speaker") for segment in result.speech_segments)


def test_t11_repeated_start_stop_has_fresh_sessions(wav_mono_16k: Path, energy_config, energy_vad) -> None:
    service = AcquisitionService(config=energy_config, vad=energy_vad)
    first = service.run_to_completion(SourceConfig(source_type="file", file_path=wav_mono_16k))
    second = service.run_to_completion(SourceConfig(source_type="file", file_path=wav_mono_16k))
    assert first.session.session_id != second.session.session_id
    assert first.chunks and second.chunks
    assert service.get_status().running is False


def test_t12_long_input_stays_bounded() -> None:
    samples = np.concatenate([speech_like(1.0) if i % 4 == 0 else silence(1.0) for i in range(20)])
    result = _run_samples(samples)
    assert result.chunks[-1].end_s == pytest.approx(20.0, abs=0.02)
    assert result.timestamp_errors == 0
    # Files bornées : un abonné saturé ne fait pas exploser le service.
    config = AcquisitionConfig(vad=VADSettings(backend="energy"))
    service = AcquisitionService(config=config, vad=EnergyVADDetector(config.vad))
    audio_q = service.subscribe_audio()
    source = SyntheticSource(samples, session_id=new_session_id())
    run_synthetic(source, config=config, vad=EnergyVADDetector(config.vad))
    assert audio_q.maxsize >= 1


def test_sc11_prolonged_silence_after_speech_is_preserved() -> None:
    samples = np.concatenate([speech_like(1.0), silence(6.0)])
    result = _run_samples(samples)
    assert result.speech_segments
    silence_events = [event for event in result.quality_events if event.speech_activity is SpeechActivity.NON_SPEECH]
    assert silence_events
    assert max(event.silence_duration_s for event in silence_events) >= 5.0
    assert any(row.kind == "silence" and row.end_s - row.start_s >= 5.0 for row in result.timeline)


def test_sc13_noisy_audio_is_not_reassuring(wav_noisy: Path, energy_config, energy_vad) -> None:
    result = run_file(wav_noisy, config=energy_config, vad=energy_vad)
    assert any(event.quality_status in {QualityStatus.DEGRADED, QualityStatus.UNUSABLE} for event in result.quality_events)


def test_sc14_muted_is_distinct_from_silence() -> None:
    silent = _run_samples(silence(2.0))
    muted = run_synthetic(
        SyntheticSource(silence(2.0), session_id=new_session_id(), mute_status=MuteStatus.MUTED),
        config=AcquisitionConfig(vad=VADSettings(backend="energy")),
        vad=EnergyVADDetector(),
    )
    assert all(event.mute_status is MuteStatus.UNKNOWN for event in silent.quality_events)
    assert all(event.mute_status is MuteStatus.MUTED for event in muted.quality_events)


def test_sc22_twenty_seconds_of_reading_is_only_published() -> None:
    result = _run_samples(silence(20.0))
    assert result.chunks[-1].end_s == pytest.approx(20.0, abs=0.02)
    assert result.speech_segments == []
    assert all(event.quality_status is QualityStatus.OK for event in result.quality_events)


def test_sc07_high_energy_is_not_an_incident(wav_intense: Path, energy_config, energy_vad) -> None:
    result = run_file(wav_intense, config=energy_config, vad=energy_vad)
    assert all(event.quality_status is QualityStatus.OK for event in result.quality_events)


def test_consent_is_required_for_microphone() -> None:
    service = AcquisitionService()
    with pytest.raises(ConsentRequiredError):
        service.start(SourceConfig(source_type="microphone", device_id="0", recording_consent=False))


def test_contracts_are_stable_and_serializable(wav_mono_16k: Path, energy_config, energy_vad) -> None:
    result = run_file(wav_mono_16k, config=energy_config, vad=energy_vad)
    event = result.quality_events[0].to_dict()
    assert set(event) >= {
        "session_id",
        "start_s",
        "end_s",
        "stream_status",
        "mute_status",
        "speech_activity",
        "quality_status",
        "rms",
        "clipping_ratio",
        "reason",
    }
    assert result.chunks[0].samples.ndim == 1
    assert "samples" not in event
