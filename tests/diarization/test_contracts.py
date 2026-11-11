from __future__ import annotations

import pytest

from safeai.diarization.config import DiarizationConfig, load_config
from safeai.diarization.contracts import DiarizationResult, SpeakerTurn, mark_overlaps
from safeai.diarization.errors import ConfigurationError


def _turn(**overrides) -> SpeakerTurn:
    values = {
        "session_id": "session_001",
        "start_s": 3.2,
        "end_s": 7.8,
        "speaker_id": "speaker_00",
        "status": "completed",
    }
    values.update(overrides)
    return SpeakerTurn(**values)


def test_speaker_turn_accepts_anonymous_ids():
    turn = _turn()
    turn.validate(duration_s=40.0)
    payload = turn.to_dict()
    assert payload["speaker_id"] == "speaker_00"
    assert payload["confidence"] is None
    assert payload["overlap"] is False


def test_unknown_turn_has_no_invented_identity():
    turn = _turn(speaker_id=None, status="unknown")
    turn.validate()
    assert turn.to_dict()["speaker_id"] is None


@pytest.mark.parametrize("speaker_id", ["Marie", "Voix 1", "SPEAKER_00", "speaker"])
def test_personal_or_display_labels_are_rejected(speaker_id):
    with pytest.raises(ValueError, match="anonyme"):
        _turn(speaker_id=speaker_id).validate()


def test_invalid_timestamps_are_rejected():
    with pytest.raises(ValueError):
        _turn(start_s=-0.1).validate()
    with pytest.raises(ValueError):
        _turn(end_s=3.2).validate()
    with pytest.raises(ValueError):
        _turn(end_s=41.0).validate(duration_s=40.0)


def test_confidence_is_optional_and_never_invented():
    turn = _turn()
    turn.validate()
    assert turn.confidence is None
    scored = _turn(confidence=0.71)
    scored.validate()
    with pytest.raises(ValueError):
        _turn(confidence=1.5).validate()


def test_overlapping_turns_are_kept_and_flagged():
    first = _turn(start_s=8.0, end_s=12.5, speaker_id="speaker_00")
    second = _turn(start_s=10.0, end_s=14.0, speaker_id="speaker_01")
    marked = mark_overlaps((first, second))
    assert marked[0].overlap is True
    assert marked[1].overlap is True
    assert marked[0].start_s == 8.0
    assert marked[1].end_s == 14.0
    result = DiarizationResult(
        session_id="session_001",
        duration_s=20.0,
        status="completed",
        turns=(first, second),
        speakers_estimated=2,
    )
    result.validate()
    assert len(result.turns) == 2
    assert len(result.overlapping_turns) == 2


def test_session_times_keep_an_explicit_offset():
    turn = _turn(start_s=122.0, end_s=125.0)
    result = DiarizationResult(
        session_id="session_001",
        duration_s=10.0,
        status="completed",
        turns=(turn,),
        speakers_estimated=1,
        offset_s=120.0,
    )
    result.validate()
    assert result.turns[0].start_s == 122.0
    assert result.turns[0].end_s == 125.0


def test_excerpt_rejects_turns_outside_the_session_window():
    with pytest.raises(ValueError, match="extrait"):
        _turn(start_s=118.0, end_s=122.0).validate(duration_s=10.0, offset_s=120.0)
    with pytest.raises(ValueError, match="durée"):
        _turn(start_s=128.0, end_s=131.0).validate(duration_s=10.0, offset_s=120.0)


def test_same_speaker_intersection_is_not_vocal_overlap():
    first = _turn(start_s=8.0, end_s=12.5, speaker_id="speaker_00")
    second = _turn(start_s=10.0, end_s=14.0, speaker_id="speaker_00")
    marked = mark_overlaps((first, second))
    assert marked[0].overlap is False
    assert marked[1].overlap is False


def test_unknown_intersection_stays_uncertain():
    known = _turn(start_s=8.0, end_s=12.5, speaker_id="speaker_00")
    unknown = _turn(start_s=10.0, end_s=14.0, speaker_id=None, status="unknown")
    marked = mark_overlaps((known, unknown))
    assert marked[0].overlap is False
    assert marked[1].overlap is False
    assert marked[1].speaker_id is None


def test_empty_silence_has_no_speaker():
    result = DiarizationResult(
        session_id="session_silence",
        duration_s=20.0,
        status="empty",
        speakers_estimated=0,
    )
    result.validate()
    assert result.turns == ()
    assert result.to_rttm() == ""


def test_failed_result_is_explicit():
    result = DiarizationResult(
        session_id="session_fail",
        duration_s=0.0,
        status="failed",
        error="Modèle de diarisation indisponible.",
    )
    result.validate()
    with pytest.raises(ValueError):
        DiarizationResult(
            session_id="session_fail",
            duration_s=0.0,
            status="failed",
        ).validate()


def test_turns_from_another_session_are_rejected():
    result = DiarizationResult(
        session_id="session_a",
        duration_s=10.0,
        status="completed",
        turns=(_turn(session_id="session_b"),),
        speakers_estimated=1,
    )
    with pytest.raises(ValueError, match="session"):
        result.validate()


def test_rttm_omits_unknown_turns_and_audio():
    known = _turn(start_s=1.0, end_s=2.5, speaker_id="speaker_00")
    unknown = _turn(start_s=3.0, end_s=4.0, speaker_id=None, status="unknown")
    result = DiarizationResult(
        session_id="session_001",
        duration_s=10.0,
        status="partial",
        turns=(known, unknown),
        speakers_estimated=1,
    )
    result.validate()
    text = result.to_rttm("r06")
    assert "speaker_00" in text
    assert "unknown" not in text
    assert "samples" not in text
    dumped = str(result.to_dict())
    assert "ndarray" not in dumped


def test_load_default_config_does_not_download_a_model():
    config = load_config()
    assert config.backend == "pyannote"
    assert config.exclusive is False
    assert config.min_speakers is None
    assert config.fingerprint() == DiarizationConfig().fingerprint()


def test_invalid_speaker_bounds_are_rejected():
    with pytest.raises(ConfigurationError):
        DiarizationConfig(min_speakers=3, max_speakers=1)
