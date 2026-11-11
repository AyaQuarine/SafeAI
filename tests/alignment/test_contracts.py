from __future__ import annotations

from pathlib import Path

import pytest

from safeai.alignment import load_aligned_session
from safeai.alignment.contracts import AlignedSession, AlignedUtterance, session_from_dict
from safeai.alignment.errors import InvalidAlignmentError

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "data" / "interface_commune" / "session_demo.json"
EMPTY_FIXTURE = ROOT / "data" / "interface_commune" / "session_empty.json"


def _utterance_payload(**overrides: object) -> dict[str, object]:
    item: dict[str, object] = {
        "session_id": "session_interface_demo",
        "start_s": 5.0,
        "end_s": 7.2,
        "text": "Bonjour à tous.",
        "language": "fr",
        "speaker_id": "speaker_00",
        "attribution": "attributed",
        "audio_available": False,
        "overlap": False,
    }
    item.update(overrides)
    return item


def _session_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "session_id": "session_interface_demo",
        "duration_s": 20.0,
        "offset_s": 0.0,
        "language": "fr",
        "status": "completed",
        "provenance": "synthetic_fixture",
        "utterances": [_utterance_payload()],
    }
    payload.update(overrides)
    return payload


def test_interface_fixture_is_valid_and_not_a_model_result():
    session = load_aligned_session(FIXTURE)
    assert session.provenance == "synthetic_fixture"
    assert session.status == "completed"
    assert session.utterances[0].speaker_id == "speaker_00"
    assert session.utterances[1].speaker_id == "speaker_01"
    assert all(item.audio_available is False for item in session.utterances)
    overlapped = [item for item in session.utterances if item.attribution == "overlap"]
    assert {item.speaker_id for item in overlapped} == {"speaker_00", "speaker_01"}
    assert not any(item.attribution == "unknown" for item in session.utterances)
    dumped = str(session.to_dict())
    assert "ndarray" not in dumped


def test_empty_session_is_legitimate_silence():
    session = load_aligned_session(EMPTY_FIXTURE)
    assert session.status == "empty"
    assert session.utterances == ()
    AlignedSession(
        session_id="session_r02",
        duration_s=20.0,
        utterances=(),
        status="empty",
        provenance="synthetic_fixture",
    ).validate()


def test_failed_session_requires_error_and_is_not_empty():
    session = AlignedSession(
        session_id="session_failed",
        duration_s=20.0,
        utterances=(),
        status="failed",
        error="Modèle indisponible",
        provenance="alignment_engine",
    )
    session.validate()
    assert session.status != "empty"
    with pytest.raises(ValueError, match="échec"):
        AlignedSession(
            session_id="session_failed",
            duration_s=20.0,
            utterances=(),
            status="failed",
        ).validate()


def test_completed_session_without_utterances_is_rejected():
    with pytest.raises(ValueError, match="aucun énoncé"):
        AlignedSession(
            session_id="session_interface_demo",
            duration_s=20.0,
            utterances=(),
            status="completed",
        ).validate()


def test_malformed_utterance_is_rejected():
    with pytest.raises(InvalidAlignmentError, match="mal formé"):
        session_from_dict(_session_payload(utterances=[_utterance_payload(), "pas un objet"]))


def test_audio_available_rejects_the_string_false():
    with pytest.raises(InvalidAlignmentError, match="booléen"):
        session_from_dict(
            _session_payload(utterances=[_utterance_payload(audio_available="false")])
        )


def test_missing_required_key_is_an_alignment_error():
    payload = _session_payload()
    del payload["duration_s"]
    with pytest.raises(InvalidAlignmentError, match="clé obligatoire absente : duration_s"):
        session_from_dict(payload)
    broken = _utterance_payload()
    del broken["start_s"]
    with pytest.raises(InvalidAlignmentError, match="clé obligatoire absente dans l'énoncé 0 : start_s"):
        session_from_dict(_session_payload(utterances=[broken]))


def test_overlap_is_not_a_unique_attribution():
    with pytest.raises(ValueError, match="chevauchement"):
        AlignedUtterance(
            session_id="session_interface_demo",
            start_s=10.0,
            end_s=11.0,
            text="mixte",
            language="fr",
            speaker_id="speaker_00",
            attribution="attributed",
            overlap=True,
        ).validate(duration_s=20.0)


def test_unknown_does_not_invent_a_speaker():
    with pytest.raises(ValueError, match="inventé"):
        AlignedUtterance(
            session_id="session_interface_demo",
            start_s=10.0,
            end_s=11.0,
            text="mixte",
            language="fr",
            speaker_id="speaker_00",
            attribution="unknown",
        ).validate(duration_s=20.0)
    AlignedUtterance(
        session_id="session_interface_demo",
        start_s=10.0,
        end_s=11.0,
        text="Je voudrais parler. D'accord.",
        language="fr",
        speaker_id=None,
        attribution="unknown",
        audio_available=False,
    ).validate(duration_s=20.0)
