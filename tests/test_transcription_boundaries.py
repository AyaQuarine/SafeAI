"""SafeAI: supplementary boundary tests for Whisper timestamps (mock-only).

Copy to tests/test_transcription_boundaries.py then run:
    python -m pytest -q tests/test_transcription_boundaries.py

These regressions are expected to fail against whisper(2).py.
"""
from types import SimpleNamespace as Row
import numpy as np
from safeai.transcription.whisper import WhisperTranscriber


class FakeWhisper:
    def __init__(self, segment):
        self.segment = segment

    def transcribe(self, audio, **kwargs):
        return [self.segment], Row(language="fr")


def run_with(segment, offset=4.0):
    return WhisperTranscriber(model=FakeWhisper(segment)).transcribe_samples(
        np.zeros(16000, dtype=np.float32),
        session_id="qa",
        source_segment_id="speech_01",
        offset_s=offset,
    )


def test_negative_local_whisper_timestamp_does_not_precede_input_audio():
    raw = Row(start=-0.1, end=0.5, text="bonjour", words=[])
    result = run_with(raw)
    assert result and result[0].status == "completed"
    assert 4.0 <= result[0].start_s < result[0].end_s <= 5.0
    result[0].validate()


def test_word_boundary_jitter_is_handled_without_crashing():
    raw = Row(
        start=0.3, end=0.9, text="bonjour",
        words=[Row(start=0.1, end=0.6, word="bonjour")],
    )
    result = run_with(raw)
    assert result and result[0].status == "completed"
    result[0].validate()
    assert all(
        result[0].start_s <= word.start_s < word.end_s <= result[0].end_s
        for word in result[0].words
    )
