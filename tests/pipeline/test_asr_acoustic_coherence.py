"""Régressions : silence confirmé vs parole faible conservée."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from safeai.diarization.contracts import DiarizationResult, SpeakerTurn
from safeai.pipeline.offline import AcquisitionSnapshot, CommonPipeline, SpeechInterval
from safeai.transcription.contracts import TranscriptSegment, WordTiming

SID = "session_acoustic_coherence"


class _Acq:
    def __init__(self, snapshot: AcquisitionSnapshot) -> None:
        self.snapshot = snapshot

    def run_file(self, path: Path, *, session_id: str) -> AcquisitionSnapshot:
        return self.snapshot


class _Asr:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe_file(self, path: Path, *, session_id: str, duration_s: float | None = None):
        return [
            TranscriptSegment(
                session_id=session_id,
                source_segment_id="file",
                start_s=1.0,
                end_s=2.0,
                text=self.text,
                language="fr",
                status="completed",
                words=(WordTiming(1.0, 2.0, self.text),),
            )
        ]


class _DiarEmpty:
    def run_file(self, path: Path, *, session_id: str) -> DiarizationResult:
        return DiarizationResult(
            session_id=session_id,
            duration_s=20.0,
            status="empty",
            turns=(),
            speakers_estimated=0,
        )


def _wav(tmp_path: Path) -> Path:
    path = tmp_path / "clip.wav"
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
    return path


def test_confirmed_silence_discards_asr_hallucination(tmp_path: Path):
    snapshot = AcquisitionSnapshot(
        session_id=SID,
        duration_s=20.0,
        source="r02",
        evidence_available=True,
        speech_intervals=(),
        signal_usable=True,
        silence_like=True,
    )
    result = CommonPipeline(_Acq(snapshot), _Asr("Sous"), _DiarEmpty()).run_file(
        _wav(tmp_path), session_id=SID
    )
    assert result.status == "empty"
    assert result.utterances == ()
    assert result.error is None
    assert any("hallucination" in note.lower() for note in result.diagnostics)


def test_weak_speech_with_silero_is_not_silently_dropped(tmp_path: Path):
    snapshot = AcquisitionSnapshot(
        session_id=SID,
        duration_s=20.0,
        source="weak",
        evidence_available=True,
        speech_intervals=(SpeechInterval(1.0, 2.0, 0.9),),
        signal_usable=True,
        silence_like=False,
    )
    result = CommonPipeline(_Acq(snapshot), _Asr("bonjour"), _DiarEmpty()).run_file(
        _wav(tmp_path), session_id=SID
    )
    assert result.status == "completed"
    assert len(result.utterances) == 1
    assert result.utterances[0].text == "bonjour"
    assert result.utterances[0].speaker_id is None
    assert result.utterances[0].attribution == "unknown"
