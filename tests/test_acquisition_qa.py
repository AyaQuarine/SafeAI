"""SafeAI acquisition: independent QA/regression checks for the uploaded source modules.

Place this file in tests/test_acquisition_qa.py in the project and run:
    python -m pytest -q tests/test_acquisition_qa.py

The tests marked as REGRESSION are EXPECTED TO FAIL on the uploaded
2026-11-01 code. Keep them failing until the defects are fixed; do not
weaken their assertions to make the suite green.
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from safeai.acquisition.config import AcquisitionConfig, VADSettings
from safeai.acquisition.contracts import (
    AudioChunk, MuteStatus, SpeechSegment, VADFrame,
)
from safeai.acquisition.publisher import AcquisitionPublisher
from safeai.acquisition.segmenter import AudioSegmenter
from safeai.acquisition.service import run_file
from safeai.acquisition.vad import EnergyVADDetector


def test_file_16k_and_48k_are_normalized_and_timestamped(tmp_path):
    for rate in (16000, 48000):
        path = tmp_path / f"silence_{rate}.wav"
        sf.write(path, np.zeros((rate * 3, 1), dtype=np.float32), rate)
        result = run_file(path, config=AcquisitionConfig(), vad=EnergyVADDetector())
        assert len(result.chunks) == 3
        assert result.timestamp_errors == 0
        assert all(chunk.sample_rate == 16000 for chunk in result.chunks)
        assert result.chunks[-1].end_s == pytest.approx(3.0, abs=1e-3)
        assert not result.speech_segments


def test_stereo_44100_downmix_to_16k(tmp_path):
    rate = 44100
    tone = 0.2 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.stack([tone, tone], axis=1), rate)
    result = run_file(path, config=AcquisitionConfig(), vad=EnergyVADDetector())
    assert result.metadata.original_channels == 2
    assert result.metadata.downmix == "average"
    assert result.timestamp_errors == 0
    assert len(result.chunks) == 1
    assert len(result.chunks[0].samples) == 16000


def test_publisher_counts_drops_when_subscriber_is_slow():
    publisher = AcquisitionPublisher()
    queue = publisher.subscribe_audio()
    for i in range(70):
        publisher.publish_audio(AudioChunk(
            session_id="qa", source_type="synthetic", start_s=float(i),
            end_s=float(i+1), sample_rate=16000,
            samples=np.zeros(16000, dtype=np.float32),
        ))
    assert queue.qsize() == 64
    assert publisher.audio.dropped == 6
    # Further acceptance: expose this counter through AcquisitionStatus and incidents.


def _vad_frames(chunk_start: float, speech_from: float, speech_to: float):
    """Create deterministic VAD frames; timings in seconds on a 16-kHz stream."""
    step = 512 / 16000
    count = int(1.0 / step)
    return [VADFrame(
        start_s=chunk_start + i*step,
        end_s=chunk_start + (i+1)*step,
        probability=0.99 if (speech_from <= chunk_start + i*step < speech_to) else 0.01,
    ) for i in range(count)]


def _chunk(i: int, samples: np.ndarray) -> AudioChunk:
    return AudioChunk(
        session_id="qa", source_type="synthetic", start_s=float(i),
        end_s=float(i+1), sample_rate=16000,
        samples=np.asarray(samples, dtype=np.float32),
    )


def test_regression_short_utterance_inside_single_chunk_contains_audio():
    """REGRESSION: a completed short utterance must never be returned with samples=None."""
    segmenter = AudioSegmenter(VADSettings(
        threshold=.5, min_speech_duration_s=.15,
        min_silence_duration_s=.4, speech_pad_s=.15,
    ))
    samples = np.zeros(16000, dtype=np.float32)
    samples[3000:6500] = 0.5
    frames = _vad_frames(0.0, .224, .416)
    _, _, segments = segmenter.ingest(_chunk(0, samples), frames)
    assert segments, "Known utterance should be segmented"
    assert all(s.audio_available and s.samples is not None for s in segments), (
        "Segmenter lost the audio for utterances opened and closed in the same chunk"
    )


def test_regression_speech_samples_match_absolute_timestamps():
    """REGRESSION: timestamp start must match the FIRST source sample in the returned segment."""
    segmenter = AudioSegmenter(VADSettings(
        threshold=.5, min_speech_duration_s=.15,
        min_silence_duration_s=.4, speech_pad_s=.15,
    ))
    # Each sample stores its absolute source time, allowing direct alignment check.
    sample_time = np.arange(16000, dtype=np.float32) / 16000.0
    segmenter.ingest(_chunk(0, sample_time), _vad_frames(0.0, .512, .704))
    _, _, segments = segmenter.ingest(_chunk(1, np.zeros(16000)), _vad_frames(1.0, -1, -1))
    assert segments, "A speech segment should have been closed"
    segment = segments[0]
    assert segment.audio_available
    assert segment.samples is not None
    assert segment.samples[0] == pytest.approx(segment.start_s, abs=2 / 16000), (
        "Segment timestamp disagrees with the actual first audio sample"
    )


def test_regression_speech_contract_rejects_mismatched_duration():
    """REGRESSION: speech timestamps and delivered samples must have matching duration."""
    bad = SpeechSegment(
        session_id="qa", start_s=.35, end_s=1.35,
        speech_detected=True, vad_mean_probability=.9, audio_available=True,
        samples=np.zeros(50, dtype=np.float32),
    )
    with pytest.raises(ValueError):
        bad.validate()


def test_regression_unavailable_silero_must_not_silently_become_energy(monkeypatch):
    """REGRESSION: fail explicitly when the production Silero backend is missing."""
    import safeai.acquisition.vad as vad_module

    def raise_missing():
        raise ImportError("Silero unavailable (injected)")

    monkeypatch.setattr(vad_module, "_load_silero", raise_missing)
    with pytest.raises(Exception):
        vad_module.build_vad(VADSettings(backend="silero"))


def test_regression_any_speech_inside_window_not_classified_as_silence():
    """REGRESSION: a 1-s window containing a short utterance must not be labeled all-silence."""
    segmenter = AudioSegmenter(VADSettings(
        threshold=.5, min_speech_duration_s=.15,
        min_silence_duration_s=.4, speech_pad_s=.15,
    ))
    samples = np.zeros(16000, dtype=np.float32)
    samples[3000:6500] = .5
    frames = _vad_frames(0.0, .224, .416)
    activity, _, segments = segmenter.ingest(_chunk(0, samples), frames)
    from safeai.acquisition.contracts import SpeechActivity
    assert activity is SpeechActivity.SPEECH, (
        "The VAD saw speech but the published chunk activity is NON_SPEECH"
    )
