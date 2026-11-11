"""Intégration simulée du pipeline commun (sans Whisper ni Community-1 réels)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from safeai.diarization.contracts import DiarizationResult, SpeakerTurn
from safeai.pipeline.offline import AcquisitionSnapshot, CommonPipeline
from safeai.transcription.contracts import TranscriptSegment, WordTiming

SID = "session_integration_r06"


class _Acquisition:
    def run_file(self, path: Path, *, session_id: str) -> AcquisitionSnapshot:
        return AcquisitionSnapshot(session_id=session_id, duration_s=12.0, source=str(path))


class _Transcriber:
    def transcribe_file(
        self,
        path: Path,
        *,
        session_id: str,
        duration_s: float | None = None,
    ) -> list[TranscriptSegment]:
        return [
            TranscriptSegment(
                session_id=session_id,
                source_segment_id="file",
                start_s=1.0,
                end_s=3.0,
                text="Bonjour à tous",
                language="fr",
                status="completed",
                words=(
                    WordTiming(1.0, 1.6, "Bonjour"),
                    WordTiming(1.6, 1.8, "à"),
                    WordTiming(1.8, 3.0, "tous"),
                ),
            ),
            TranscriptSegment(
                session_id=session_id,
                source_segment_id="file",
                start_s=4.0,
                end_s=6.5,
                text="Je voudrais parler",
                language="fr",
                status="completed",
                words=(
                    WordTiming(4.0, 4.2, "Je"),
                    WordTiming(4.2, 5.0, "voudrais"),
                    WordTiming(5.0, 6.5, "parler"),
                ),
            ),
            TranscriptSegment(
                session_id=session_id,
                source_segment_id="file",
                start_s=6.4,
                end_s=7.2,
                text="D'accord",
                language="fr",
                status="completed",
                words=(WordTiming(6.4, 7.2, "D'accord"),),
            ),
        ]


class _Diarizer:
    def run_file(self, path: Path, *, session_id: str) -> DiarizationResult:
        return DiarizationResult(
            session_id=session_id,
            duration_s=12.0,
            status="completed",
            turns=(
                SpeakerTurn(session_id, 1.0, 3.2, "speaker_00"),
                SpeakerTurn(session_id, 4.0, 6.5, "speaker_01"),
                SpeakerTurn(session_id, 6.4, 7.0, "speaker_00"),
                SpeakerTurn(session_id, 6.6, 7.4, "speaker_01"),
            ),
            speakers_estimated=2,
        )


def test_two_speakers_and_overlap_are_preserved(tmp_path: Path):
    wav = tmp_path / "r06_like.wav"
    sf.write(wav, np.zeros(12 * 16000, dtype=np.float32), 16000)
    pipeline = CommonPipeline(_Acquisition(), _Transcriber(), _Diarizer())

    session = pipeline.run_file(wav, session_id=SID)

    assert session.status == "completed"
    assert session.session_id == SID
    assert session.duration_s == 12.0
    assert session.provenance == "common_pipeline"
    speakers = {item.speaker_id for item in session.utterances if item.speaker_id}
    assert speakers == {"speaker_00", "speaker_01"}
    overlapped = [item for item in session.utterances if item.attribution == "overlap"]
    assert overlapped
    assert all(item.speaker_id is None for item in overlapped)
    texts = " ".join(item.text for item in session.utterances)
    assert "Bonjour" in texts
    assert "parler" in texts
    assert texts.count("D'accord") == 1
    session.validate()


def test_failed_diarization_result_is_not_empty(tmp_path: Path):
    wav = tmp_path / "clip.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)

    class FailedDiarizer:
        def run_file(self, path: Path, *, session_id: str) -> DiarizationResult:
            return DiarizationResult(
                session_id=session_id,
                duration_s=1.0,
                status="failed",
                turns=(),
                speakers_estimated=0,
                error="Modèle de diarisation indisponible",
            )

    pipeline = CommonPipeline(_Acquisition(), _Transcriber(), FailedDiarizer())
    session = pipeline.run_file(wav, session_id=SID)
    assert session.status == "failed"
    assert session.utterances == ()
    assert session.error
