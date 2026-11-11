"""Regressions de securisation avant CommonPipeline V1.

Placer dans tests/alignment/test_alignment_pre_pipeline_qa.py.
Les assertions expriment les contrats cibles ; elles echouent sur engine.py
fourni le 10-11-2026 et doivent passer apres correction.
"""

import pytest

from safeai.alignment.engine import align_session
from safeai.alignment.errors import InvalidAlignmentError
from safeai.diarization.contracts import DiarizationResult, SpeakerTurn
from safeai.transcription.contracts import TranscriptSegment, WordTiming

SID = "session_common_001"


def _transcript(session_id=SID):
    return TranscriptSegment(
        session_id=session_id,
        source_segment_id="whisper_001",
        start_s=1.0,
        end_s=2.0,
        text="bonjour",
        language="fr",
        status="completed",
        words=(WordTiming(start_s=1.0, end_s=2.0, text="bonjour"),),
    )


def _diarization(session_id=SID, status="completed", turn_start=1.0, turn_end=2.0):
    return DiarizationResult(
        session_id=session_id,
        duration_s=4.0,
        status=status,
        turns=(
            SpeakerTurn(
                session_id=session_id,
                start_s=turn_start,
                end_s=turn_end,
                speaker_id="speaker_00",
                status="completed",
            ),
        ),
        speakers_estimated=1,
    )


def test_transcription_from_another_session_is_rejected():
    with pytest.raises(InvalidAlignmentError, match="session"):
        align_session([_transcript("another_session")], _diarization(), session_id=SID, duration_s=4.0)


def test_diarization_from_another_session_is_rejected():
    with pytest.raises(InvalidAlignmentError, match="session"):
        align_session([_transcript()], _diarization("another_session"), session_id=SID, duration_s=4.0)


def test_partial_diarization_is_not_silently_upgraded_to_completed():
    result = align_session([_transcript()], _diarization(status="partial"), session_id=SID, duration_s=4.0)
    assert result.status == "partial"
    result.validate()


def test_one_weak_speaker_covering_only_30_percent_of_word_remains_unknown():
    result = align_session(
        [_transcript()],
        _diarization(turn_start=1.0, turn_end=1.3),
        session_id=SID,
        duration_s=4.0,
    )
    assert len(result.utterances) == 1
    utterance = result.utterances[0]
    assert utterance.attribution == "unknown"
    assert utterance.speaker_id is None
