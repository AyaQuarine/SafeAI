from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from safeai.diarization.contracts import DiarizationResult, SpeakerTurn
from safeai.pipeline.offline import AcquisitionSnapshot, CommonPipeline, SpeechInterval
from safeai.transcription.contracts import TranscriptSegment, WordTiming

SID = "session_common_018"
DUR = 40.0


class _Acq:
    def run_file(self, path: Path, *, session_id: str) -> AcquisitionSnapshot:
        return AcquisitionSnapshot(
            session_id=session_id,
            duration_s=DUR,
            source=str(path),
            evidence_available=True,
            speech_intervals=(
                SpeechInterval(1.0, 3.0, 0.9),
                SpeechInterval(30.0, 34.0, 0.95),
            ),
            signal_usable=True,
            silence_like=False,
        )


class _Asr:
    def transcribe_file(self, path: Path, *, session_id: str, duration_s=None):
        return [
            TranscriptSegment(
                session_id=session_id,
                source_segment_id="a",
                start_s=1.0,
                end_s=2.0,
                text="Bonjour",
                language="fr",
                status="completed",
                words=(WordTiming(1.0, 2.0, "Bonjour"),),
            )
        ]


class _Diar:
    def run_file(self, path: Path, *, session_id: str) -> DiarizationResult:
        return DiarizationResult(
            session_id=session_id,
            duration_s=DUR,
            status="completed",
            turns=(
                SpeakerTurn(session_id, 1.0, 3.0, "speaker_00"),
                SpeakerTurn(session_id, 30.0, 34.0, "speaker_00"),
                SpeakerTurn(session_id, 32.5, 33.5, "speaker_01"),
            ),
            speakers_estimated=2,
        )


def test_bundle_keeps_overlap_turns_even_without_aligned_overlap_text(tmp_path: Path):
    wav = tmp_path / "r06.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
    bundle = CommonPipeline(_Acq(), _Asr(), _Diar()).run_bundle(wav, session_id=SID)
    assert bundle.diarization is not None
    assert bundle.diarization.session_id == SID
    assert len(bundle.diarization.overlapping_turns) >= 2
    assert bundle.aligned.session_id == SID
    # Pas d'énoncé inventé pour le chevauchement sans texte Whisper.
    assert not any(item.attribution == "overlap" for item in bundle.aligned.utterances)
