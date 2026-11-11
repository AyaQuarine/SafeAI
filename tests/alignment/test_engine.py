from __future__ import annotations

from safeai.alignment.contracts import AlignedSession
from safeai.alignment.engine import AlignmentConfig, align_session
from safeai.diarization.contracts import DiarizationResult, SpeakerTurn
from safeai.transcription.contracts import TranscriptSegment, WordTiming


SESSION = "session_align_demo"
DURATION = 20.0


def _word(start: float, end: float, text: str) -> WordTiming:
    return WordTiming(start_s=start, end_s=end, text=text)


def _transcript(
    *,
    start: float,
    end: float,
    text: str,
    words: tuple[WordTiming, ...] = (),
    status: str = "completed",
    source_id: str = "seg_001",
) -> TranscriptSegment:
    return TranscriptSegment(
        session_id=SESSION,
        source_segment_id=source_id,
        start_s=start,
        end_s=end,
        text=text,
        language="fr",
        status=status,  # type: ignore[arg-type]
        words=words,
    )


def _turn(
    start: float,
    end: float,
    speaker_id: str | None,
    *,
    status: str = "completed",
) -> SpeakerTurn:
    return SpeakerTurn(
        session_id=SESSION,
        start_s=start,
        end_s=end,
        speaker_id=speaker_id,
        status="unknown" if speaker_id is None else status,  # type: ignore[arg-type]
    )


def _diarization(
    turns: tuple[SpeakerTurn, ...] = (),
    *,
    status: str = "completed",
    speakers: int | None = None,
    error: str | None = None,
) -> DiarizationResult:
    unique = {turn.speaker_id for turn in turns if turn.speaker_id is not None}
    if speakers is None:
        speakers = 0 if status == "empty" else len(unique)
    return DiarizationResult(
        session_id=SESSION,
        duration_s=DURATION,
        status=status,  # type: ignore[arg-type]
        turns=turns,
        speakers_estimated=speakers,
        error=error,
    )


def test_align_01_single_speaker_is_attributed():
    transcripts = [
        _transcript(
            start=1.0,
            end=2.4,
            text="Bonjour je vais bien",
            words=(
                _word(1.0, 1.5, "Bonjour"),
                _word(1.5, 1.7, "je"),
                _word(1.7, 2.0, "vais"),
                _word(2.0, 2.4, "bien"),
            ),
        )
    ]
    diarization = _diarization((_turn(1.0, 2.5, "speaker_00"),))

    session = align_session(
        transcripts,
        diarization,
        session_id=SESSION,
        duration_s=DURATION,
    )

    assert session.status == "completed"
    assert session.provenance == "alignment_engine"
    assert len(session.utterances) == 1
    utterance = session.utterances[0]
    assert utterance.speaker_id == "speaker_00"
    assert utterance.attribution == "attributed"
    assert utterance.overlap is False
    assert "Bonjour" in utterance.text and "bien" in utterance.text
    session.validate()


def test_align_02_silence_is_empty_not_invented():
    transcripts = [
        _transcript(start=0.0, end=1.0, text="", status="empty", words=()),
    ]
    diarization = _diarization((), status="empty", speakers=0)

    session = align_session(
        transcripts,
        diarization,
        session_id=SESSION,
        duration_s=DURATION,
    )

    assert session.status == "empty"
    assert session.utterances == ()
    assert session.error is None
    session.validate()


def test_align_03_two_speakers_are_associated():
    transcripts = [
        _transcript(
            start=5.0,
            end=7.2,
            text="Bonjour à tous",
            words=(
                _word(5.0, 5.6, "Bonjour"),
                _word(5.6, 5.8, "à"),
                _word(5.8, 7.2, "tous"),
            ),
            source_id="seg_a",
        ),
        _transcript(
            start=8.1,
            end=10.0,
            text="Je voudrais parler",
            words=(
                _word(8.1, 8.3, "Je"),
                _word(8.3, 8.9, "voudrais"),
                _word(8.9, 10.0, "parler"),
            ),
            source_id="seg_b",
        ),
    ]
    diarization = _diarization(
        (
            _turn(5.0, 7.5, "speaker_00"),
            _turn(8.0, 10.5, "speaker_01"),
        )
    )

    session = align_session(
        transcripts,
        diarization,
        session_id=SESSION,
        duration_s=DURATION,
    )

    assert session.status == "completed"
    speakers = [item.speaker_id for item in session.utterances]
    assert speakers == ["speaker_00", "speaker_01"]
    assert all(item.attribution == "attributed" for item in session.utterances)
    texts = [item.text for item in session.utterances]
    assert any("Bonjour" in text for text in texts)
    assert any("parler" in text for text in texts)
    session.validate()


def test_align_04_overlap_keeps_uncertainty():
    # Exemple superviseur : « bien » chevauche speaker_00 et speaker_01.
    transcripts = [
        _transcript(
            start=1.0,
            end=2.4,
            text="Bonjour je vais bien",
            words=(
                _word(1.0, 1.5, "Bonjour"),
                _word(1.5, 1.7, "je"),
                _word(1.7, 2.0, "vais"),
                _word(2.0, 2.4, "bien"),
            ),
        )
    ]
    diarization = _diarization(
        (
            _turn(1.0, 2.2, "speaker_00"),
            _turn(2.0, 2.5, "speaker_01"),
        )
    )

    session = align_session(
        transcripts,
        diarization,
        session_id=SESSION,
        duration_s=DURATION,
        config=AlignmentConfig(min_overlap_ratio=0.25, exclusive_ratio=0.60),
    )

    assert session.status == "completed"
    by_text = {item.text: item for item in session.utterances}
    # Les premiers mots restent attribués ; « bien » reste ambigu.
    assert any(
        item.attribution == "attributed" and item.speaker_id == "speaker_00"
        for item in session.utterances
    )
    overlapped = [item for item in session.utterances if item.attribution == "overlap"]
    assert len(overlapped) == 1
    assert overlapped[0].text.strip() == "bien"
    assert overlapped[0].speaker_id is None
    assert overlapped[0].overlap is True
    # Pas d'attribution unique forcée sur le chevauchement.
    assert not any(
        item.text.strip() == "bien" and item.attribution == "attributed"
        for item in session.utterances
    )
    # Aucun mot compté deux fois.
    word_tokens = " ".join(item.text for item in session.utterances).split()
    assert word_tokens.count("bien") == 1
    assert word_tokens.count("Bonjour") == 1
    session.validate()
    assert "bien" in by_text or any("bien" in item.text for item in session.utterances)


def test_align_05_model_failure_is_not_silence():
    transcripts = [
        _transcript(start=1.0, end=2.0, text="Bonjour", words=(_word(1.0, 2.0, "Bonjour"),))
    ]
    diarization = _diarization(
        (),
        status="failed",
        speakers=0,
        error="Modèle de diarisation indisponible",
    )

    session = align_session(
        transcripts,
        diarization,
        session_id=SESSION,
        duration_s=DURATION,
    )

    assert session.status == "failed"
    assert session.utterances == ()
    assert session.error
    assert "indisponible" in session.error or "failed" in session.error.lower()
    session.validate()

    failed_asr = [
        _transcript(start=1.0, end=2.0, text="", status="failed", words=()),
    ]
    ok_diarization = _diarization((_turn(1.0, 2.0, "speaker_00"),))
    asr_failed = align_session(
        failed_asr,
        ok_diarization,
        session_id=SESSION,
        duration_s=DURATION,
    )
    assert asr_failed.status == "failed"
    assert asr_failed.utterances == ()
    assert asr_failed.status != "empty"
    asr_failed.validate()


def test_align_06_complete_session_is_coherent():
    transcripts = [
        _transcript(
            start=5.0,
            end=7.2,
            text="Bonjour à tous",
            words=(
                _word(5.0, 5.6, "Bonjour"),
                _word(5.6, 5.8, "à"),
                _word(5.8, 7.2, "tous"),
            ),
            source_id="seg_a",
        ),
        _transcript(
            start=8.1,
            end=11.0,
            text="Je voudrais parler D'accord",
            words=(
                _word(8.1, 8.3, "Je"),
                _word(8.3, 8.9, "voudrais"),
                _word(8.9, 10.0, "parler"),
                _word(10.0, 10.8, "D'accord"),
            ),
            source_id="seg_b",
        ),
    ]
    diarization = _diarization(
        (
            _turn(5.0, 7.5, "speaker_00"),
            _turn(8.0, 10.0, "speaker_01"),
            _turn(10.0, 10.8, "speaker_00"),
            _turn(10.2, 11.0, "speaker_01"),
        )
    )

    session = align_session(
        transcripts,
        diarization,
        session_id=SESSION,
        duration_s=DURATION,
        source="data/local_dev/r06_deux_locuteurs.wav",
        audio_available=True,
    )

    assert isinstance(session, AlignedSession)
    assert session.status == "completed"
    assert session.session_id == SESSION
    assert session.duration_s == DURATION
    assert session.provenance == "alignment_engine"
    assert session.source.endswith("r06_deux_locuteurs.wav")
    assert all(item.session_id == SESSION for item in session.utterances)
    assert all(0.0 <= item.start_s < item.end_s <= DURATION + 1e-6 for item in session.utterances)
    # Chronologie monotone non décroissante entre énoncés successifs.
    starts = [item.start_s for item in session.utterances]
    assert starts == sorted(starts)
    session.validate()
    dumped = session.to_dict()
    assert dumped["status"] == "completed"
    assert "ndarray" not in str(dumped)


def test_align_without_word_timings_uses_coarse_or_unknown():
    transcripts = [
        _transcript(start=5.0, end=7.0, text="Bonjour à tous", words=()),
    ]
    diarization = _diarization((_turn(5.0, 7.0, "speaker_00"),))
    session = align_session(
        transcripts,
        diarization,
        session_id=SESSION,
        duration_s=DURATION,
    )
    assert len(session.utterances) == 1
    assert session.utterances[0].speaker_id == "speaker_00"
    assert session.utterances[0].attribution == "attributed"

    ambiguous = align_session(
        [_transcript(start=5.0, end=7.0, text="Mixte", words=())],
        _diarization(
            (
                _turn(5.0, 6.5, "speaker_00"),
                _turn(5.5, 7.0, "speaker_01"),
            )
        ),
        session_id=SESSION,
        duration_s=DURATION,
    )
    assert ambiguous.utterances[0].attribution in {"overlap", "unknown"}
    assert ambiguous.utterances[0].speaker_id is None
