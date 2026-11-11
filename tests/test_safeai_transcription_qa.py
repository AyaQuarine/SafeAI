"""SafeAI — additional transcription and acquisition integration tests.

Place this file in tests/test_safeai_transcription_qa.py and run:
    python -m pytest -q tests/test_safeai_transcription_qa.py

Tests marked REGRESSION intentionally expose issues in the uploaded
2026-11-02 code. Do not weaken them to make the suite green.

No model download or microphone is required.
"""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from safeai.acquisition.contracts import AudioChunk, SpeechActivity
from safeai.acquisition.service import AcquisitionService
from safeai.acquisition.vad import EnergyVADDetector
from safeai.transcription.whisper import WhisperTranscriber


class FakeWhisper:
    """A model stub: tests adapter contracts, not ASR accuracy."""

    def __init__(self, start=0.35, end=0.85, text=" Bonjour"):
        self.start = start
        self.end = end
        self.text = text
        self.received_audio = None

    def transcribe(self, audio, **kwargs):
        self.received_audio = audio
        raw = SimpleNamespace(
            start=self.start,
            end=self.end,
            text=self.text,
            words=[SimpleNamespace(start=self.start, end=self.end, word=self.text)],
        )
        return iter([raw]), SimpleNamespace(language="fr")


def test_timestamps_are_global_for_transcript_and_words():
    stub = FakeWhisper()
    adapter = WhisperTranscriber(model=stub)
    # A fake model should work independently of external model packages.
    with patch("safeai.transcription.whisper._accept_pcm_directly", return_value=None):
        result = adapter.transcribe_samples(
            np.zeros(16000, np.float32),
            session_id="s",
            source_segment_id="segment_1",
            offset_s=12.40,
        )
    assert len(result) == 1
    assert result[0].start_s == pytest.approx(12.75)
    assert result[0].end_s == pytest.approx(13.25)
    assert result[0].words[0].start_s == pytest.approx(12.75)
    assert result[0].words[0].end_s == pytest.approx(13.25)
    assert result[0].speaker_id is None


def test_adapter_does_not_patch_faster_whisper_internals():
    """REGRESSION: injecting a fake model must not patch global library internals."""
    stub = FakeWhisper()
    adapter = WhisperTranscriber(model=stub)
    with patch(
        "safeai.transcription.whisper._accept_pcm_directly",
        side_effect=AssertionError("Do not modify faster-whisper internal functions"),
    ):
        result = adapter.transcribe_samples(
            np.zeros(16000, np.float32),
            session_id="s",
            source_segment_id="segment_1",
            offset_s=0.0,
        )
    assert result


def test_stereo_input_is_rejected_not_flattened():
    """REGRESSION: 16k stereo must not become an accidental 32k mono signal."""
    stub = FakeWhisper()
    adapter = WhisperTranscriber(model=stub)
    with patch("safeai.transcription.whisper._accept_pcm_directly", return_value=None):
        with pytest.raises(ValueError, match="mono|ndim|dimension|canaux|channels|shape"):
            adapter.transcribe_samples(
                np.zeros((16000, 2), np.float32),
                session_id="s",
                source_segment_id="stereo",
                offset_s=0.0,
            )


def test_requested_duration_cannot_exceed_available_audio(tmp_path):
    """REGRESSION: word/segment timestamp cannot exceed file's actual duration."""
    path = tmp_path / "one_second.wav"
    sf.write(path, np.zeros(16000, np.float32), 16000)
    stub = FakeWhisper(start=0.1, end=2.5)
    adapter = WhisperTranscriber(model=stub)
    with patch("safeai.transcription.whisper._accept_pcm_directly", return_value=None):
        result = adapter.transcribe_file(
            path,
            session_id="s",
            source_segment_id="file",
            duration_s=4.0,
        )
    assert all(0 <= seg.start_s < seg.end_s <= 1.0 for seg in result)


def test_gap_breaks_continuous_silence_clock():
    """REGRESSION: 0-1s silence, 1-2s missing, 2-3s silence != 2s continuous silence."""
    service = AcquisitionService(vad=EnergyVADDetector())
    service._vad = EnergyVADDetector()  # direct deterministic pipeline-level check
    for start in (0.0, 2.0):
        service._process_chunk(AudioChunk(
            session_id="s",
            source_type="synthetic",
            start_s=start,
            end_s=start + 1.0,
            sample_rate=16000,
            samples=np.zeros(16000, dtype=np.float32),
        ))
    events = list(service._collected_quality)
    assert any(e.reason == "timeline_gap" for e in events)
    after_gap = [e for e in events if e.start_s == 2.0]
    assert len(after_gap) == 1
    assert after_gap[0].speech_activity == SpeechActivity.NON_SPEECH
    assert after_gap[0].silence_duration_s == pytest.approx(1.0, abs=0.03)
