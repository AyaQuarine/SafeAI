from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from safeai.pipeline.errors import PipelineConfigurationError
from safeai.pipeline.offline import AcquisitionSnapshot, CommonPipeline
from safeai.diarization.contracts import DiarizationResult, SpeakerTurn
from safeai.transcription.contracts import TranscriptSegment, WordTiming


SID = "session_pipeline_001"


class FakeAcquisition:
    def __init__(self, snapshot: AcquisitionSnapshot) -> None:
        self.snapshot = snapshot
        self.calls: list[tuple[Path, str]] = []

    def run_file(self, path: Path, *, session_id: str) -> AcquisitionSnapshot:
        self.calls.append((path, session_id))
        return self.snapshot


class FakeTranscriber:
    def __init__(self, segments: list[TranscriptSegment] | Exception) -> None:
        self.segments = segments
        self.calls: list[tuple[Path, str, float | None]] = []

    def transcribe_file(
        self,
        path: Path,
        *,
        session_id: str,
        duration_s: float | None = None,
    ) -> list[TranscriptSegment]:
        self.calls.append((path, session_id, duration_s))
        if isinstance(self.segments, Exception):
            raise self.segments
        return self.segments


class FakeDiarizer:
    def __init__(self, result: DiarizationResult | Exception) -> None:
        self.result = result
        self.calls: list[tuple[Path, str]] = []

    def run_file(self, path: Path, *, session_id: str) -> DiarizationResult:
        self.calls.append((path, session_id))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _transcript(
    *,
    start: float = 1.0,
    end: float = 2.0,
    text: str = "bonjour",
    status: str = "completed",
) -> TranscriptSegment:
    words = ()
    if status == "completed" and text.strip():
        words = (WordTiming(start_s=start, end_s=end, text=text),)
    return TranscriptSegment(
        session_id=SID,
        source_segment_id="file",
        start_s=start,
        end_s=end,
        text=text,
        language="fr",
        status=status,  # type: ignore[arg-type]
        words=words,
    )


def _diarization(
    *,
    status: str = "completed",
    turns: tuple[SpeakerTurn, ...] | None = None,
    error: str | None = None,
    speakers: int | None = None,
) -> DiarizationResult:
    if turns is None:
        turns = (
            SpeakerTurn(
                session_id=SID,
                start_s=1.0,
                end_s=2.0,
                speaker_id="speaker_00",
                status="completed",
            ),
        )
    unique = {turn.speaker_id for turn in turns if turn.speaker_id is not None}
    if speakers is None:
        speakers = 0 if status in {"empty", "failed"} else len(unique)
    return DiarizationResult(
        session_id=SID,
        duration_s=4.0,
        status=status,  # type: ignore[arg-type]
        turns=turns,
        speakers_estimated=speakers,
        error=error,
    )


def test_pipeline_requires_all_components():
    with pytest.raises(PipelineConfigurationError):
        CommonPipeline(None, FakeTranscriber([]), FakeDiarizer(_diarization()))  # type: ignore[arg-type]


def test_run_file_uses_one_session_id(tmp_path: Path):
    wav = tmp_path / "clip.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
    acquisition = FakeAcquisition(
        AcquisitionSnapshot(session_id=SID, duration_s=4.0, source=str(wav))
    )
    transcriber = FakeTranscriber([_transcript()])
    diarizer = FakeDiarizer(_diarization())
    pipeline = CommonPipeline(acquisition, transcriber, diarizer)

    session = pipeline.run_file(wav, session_id=SID)

    assert session.session_id == SID
    assert acquisition.calls == [(wav, SID)]
    assert transcriber.calls == [(wav, SID, 4.0)]
    assert diarizer.calls == [(wav, SID)]
    assert session.status == "completed"
    assert session.provenance == "common_pipeline"
    assert session.utterances[0].speaker_id == "speaker_00"
    session.validate()


def test_missing_file_is_failed_not_empty(tmp_path: Path):
    missing = tmp_path / "absent.wav"
    pipeline = CommonPipeline(
        FakeAcquisition(AcquisitionSnapshot(SID, 0.0, str(missing))),
        FakeTranscriber([]),
        FakeDiarizer(_diarization(status="empty", turns=(), speakers=0)),
    )
    session = pipeline.run_file(missing, session_id=SID)
    assert session.status == "failed"
    assert session.utterances == ()
    assert "introuvable" in (session.error or "")


def test_diarizer_exception_is_failed_not_empty(tmp_path: Path):
    wav = tmp_path / "clip.wav"
    sf.write(wav, np.zeros(8000, dtype=np.float32), 16000)
    pipeline = CommonPipeline(
        FakeAcquisition(AcquisitionSnapshot(SID, 0.5, source=str(wav))),
        FakeTranscriber([_transcript(start=0.1, end=0.4, text="salut")]),
        FakeDiarizer(RuntimeError("Modèle de diarisation indisponible")),
    )
    session = pipeline.run_file(wav, session_id=SID)
    assert session.status == "failed"
    assert session.status != "empty"
    assert session.utterances == ()
    assert "diarisation" in (session.error or "").lower()
    assert "indisponible" in (session.error or "")


def test_empty_silence_from_models(tmp_path: Path):
    wav = tmp_path / "silence.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
    pipeline = CommonPipeline(
        FakeAcquisition(
            AcquisitionSnapshot(
                SID,
                1.0,
                source=str(wav),
                evidence_available=True,
                signal_usable=True,
                silence_like=True,
            )
        ),
        FakeTranscriber(
            [
                TranscriptSegment(
                    session_id=SID,
                    source_segment_id="file",
                    start_s=0.0,
                    end_s=1.0,
                    text="",
                    language="fr",
                    status="empty",
                )
            ]
        ),
        FakeDiarizer(_diarization(status="empty", turns=(), speakers=0)),
    )
    session = pipeline.run_file(wav, session_id=SID)
    assert session.status == "empty"
    assert session.utterances == ()
    assert session.error is None


def test_empty_models_without_acoustic_evidence_is_failed(tmp_path: Path):
    """ASR et diarisation vides ne prouvent pas le silence sans preuve Silero/qualité."""
    wav = tmp_path / "unverified.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
    pipeline = CommonPipeline(
        FakeAcquisition(AcquisitionSnapshot(SID, 1.0, source=str(wav))),
        FakeTranscriber(
            [
                TranscriptSegment(
                    session_id=SID,
                    source_segment_id="file",
                    start_s=0.0,
                    end_s=1.0,
                    text="",
                    language="fr",
                    status="empty",
                )
            ]
        ),
        FakeDiarizer(_diarization(status="empty", turns=(), speakers=0)),
    )
    session = pipeline.run_file(wav, session_id=SID)
    assert session.status == "failed"
    assert "Silence non vérifié" in (session.error or "")


def test_acquisition_session_mismatch_is_failed(tmp_path: Path):
    wav = tmp_path / "clip.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
    pipeline = CommonPipeline(
        FakeAcquisition(
            AcquisitionSnapshot(session_id="other", duration_s=1.0, source=str(wav))
        ),
        FakeTranscriber([_transcript()]),
        FakeDiarizer(_diarization()),
    )
    session = pipeline.run_file(wav, session_id=SID)
    assert session.status == "failed"
    assert "session incompatible" in (session.error or "")
