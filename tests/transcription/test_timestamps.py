"""Horodatage global : le temps Whisper est local au segment envoyé."""

from __future__ import annotations

import numpy as np

from safeai.transcription.config import TranscriptionConfig
from safeai.transcription.whisper import WhisperTranscriber


class _Word:
    def __init__(self, start: float, end: float, word: str) -> None:
        self.start = start
        self.end = end
        self.word = word


class _Raw:
    def __init__(self) -> None:
        self.start = 0.35
        self.end = 1.10
        self.text = " Aidez-moi, je ne me sens pas bien."
        self.words = [_Word(0.35, 0.80, " Aidez-moi")]


class _Info:
    language = "fr"


class _Model:
    def __init__(self) -> None:
        self.kwargs = None

    def transcribe(self, path, **kwargs):
        self.kwargs = kwargs
        return [_Raw()], _Info()


def test_local_whisper_time_becomes_session_time() -> None:
    model = _Model()
    transcriber = WhisperTranscriber(TranscriptionConfig(vad_filter=False), model=model)
    samples = np.zeros(int(2.0 * 16000), dtype=np.float32)
    produced = transcriber.transcribe_samples(
        samples,
        session_id="session_001",
        source_segment_id="speech_004",
        offset_s=12.40,
        end_limit_s=14.40,
    )
    assert model.kwargs["vad_filter"] is False
    assert model.kwargs["language"] == "fr"
    assert len(produced) == 1
    segment = produced[0]
    assert segment.start_s == 12.75
    assert segment.end_s == 13.5
    assert segment.speaker_id is None
    assert segment.status == "completed"
    assert segment.language == "fr"
    assert segment.words[0].start_s == 12.75
    assert "Aidez-moi" in segment.text


def test_silence_yields_no_invented_text() -> None:
    class Empty:
        def transcribe(self, path, **kwargs):
            return [], _Info()

    transcriber = WhisperTranscriber(model=Empty())
    produced = transcriber.transcribe_samples(
        np.zeros(16000, dtype=np.float32),
        session_id="session_001",
        source_segment_id="speech_000",
        offset_s=0.0,
    )
    assert len(produced) == 1
    assert produced[0].status == "empty"
    assert produced[0].text == ""
    assert produced[0].speaker_id is None
    produced[0].validate()
