"""Moteur d'alignement Whisper ↔ Community-1.

Associe les mots (ou les segments) aux tours de locuteurs anonymes.
Un chevauchement reste une incertitude : aucune attribution unique forcée.
Ce module ne charge aucun modèle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from safeai.alignment.contracts import AlignedSession, AlignedUtterance, Attribution
from safeai.alignment.errors import InvalidAlignmentError
from safeai.diarization.contracts import DiarizationResult, SpeakerTurn
from safeai.transcription.contracts import TranscriptSegment

DEFAULT_PROVENANCE = "alignment_engine"


@dataclass(frozen=True)
class AlignmentConfig:
    """Seuils de recouvrement, testables, pas des certitudes inventées."""

    min_overlap_ratio: float = 0.25
    exclusive_ratio: float = 0.60


@dataclass(frozen=True)
class _WordDecision:
    start_s: float
    end_s: float
    text: str
    language: str
    speaker_id: str | None
    attribution: Attribution
    overlap: bool


def align_session(
    transcripts: Sequence[TranscriptSegment] | list[TranscriptSegment],
    diarization: DiarizationResult,
    *,
    session_id: str,
    duration_s: float,
    offset_s: float | None = None,
    language: str = "fr",
    provenance: str = DEFAULT_PROVENANCE,
    source: str | None = None,
    audio_available: bool = True,
    config: AlignmentConfig | None = None,
) -> AlignedSession:
    """Aligne transcriptions et diarisation vers une ``AlignedSession``.

    Indépendant des modèles : les entrées sont des contrats déjà produits.
    """
    if not session_id:
        raise InvalidAlignmentError("session_id manquant")
    if duration_s < 0:
        raise InvalidAlignmentError("duration_s invalide")
    cfg = config or AlignmentConfig()
    _validate_thresholds(cfg)
    items = list(transcripts)
    for segment in items:
        try:
            segment.validate()
        except ValueError as exc:
            raise InvalidAlignmentError(f"transcription invalide : {exc}") from exc
        if segment.session_id != session_id:
            raise InvalidAlignmentError(
                f"session incompatible : transcription {segment.session_id}"
            )
    try:
        diarization.validate()
    except ValueError as exc:
        raise InvalidAlignmentError(f"diarisation invalide : {exc}") from exc
    if diarization.session_id != session_id:
        raise InvalidAlignmentError(
            f"session incompatible : diarisation {diarization.session_id}"
        )

    resolved_offset = diarization.offset_s if offset_s is None else float(offset_s)
    if resolved_offset < 0:
        raise InvalidAlignmentError("offset_s invalide")

    if diarization.status == "failed":
        return _failed_session(
            session_id=session_id,
            duration_s=duration_s,
            offset_s=resolved_offset,
            language=language,
            provenance=provenance,
            source=source,
            error=diarization.error or "diarisation failed",
        )

    failed_transcripts = [item for item in items if item.status == "failed"]
    if failed_transcripts:
        return _failed_session(
            session_id=session_id,
            duration_s=duration_s,
            offset_s=resolved_offset,
            language=language,
            provenance=provenance,
            source=source,
            error="transcription failed",
        )

    usable = [item for item in items if item.status == "completed" and item.text.strip()]
    if not usable:
        return AlignedSession(
            session_id=session_id,
            duration_s=duration_s,
            utterances=(),
            offset_s=resolved_offset,
            language=language,
            provenance=provenance,
            source=source,
            status="empty",
        )

    known_turns = tuple(
        turn
        for turn in diarization.turns
        if turn.speaker_id is not None and turn.status != "unknown"
    )
    decisions: list[_WordDecision] = []
    for segment in usable:
        decisions.extend(
            _decisions_for_segment(segment, known_turns=known_turns, config=cfg)
        )

    utterances = _merge_decisions(
        decisions,
        session_id=session_id,
        audio_available=audio_available,
    )
    if not utterances:
        return AlignedSession(
            session_id=session_id,
            duration_s=duration_s,
            utterances=(),
            offset_s=resolved_offset,
            language=language,
            provenance=provenance,
            source=source,
            status="empty",
        )

    status = "partial" if diarization.status == "partial" else "completed"
    session = AlignedSession(
        session_id=session_id,
        duration_s=duration_s,
        utterances=tuple(utterances),
        offset_s=resolved_offset,
        language=language,
        provenance=provenance,
        source=source,
        status=status,
    )
    try:
        session.validate()
    except ValueError as exc:
        raise InvalidAlignmentError(str(exc)) from exc
    return session


def _validate_thresholds(config: AlignmentConfig) -> None:
    for name, value in (
        ("min_overlap_ratio", config.min_overlap_ratio),
        ("exclusive_ratio", config.exclusive_ratio),
    ):
        if not 0.0 <= value <= 1.0:
            raise InvalidAlignmentError(f"{name} doit rester dans [0, 1]")
    if config.exclusive_ratio < config.min_overlap_ratio:
        raise InvalidAlignmentError(
            "exclusive_ratio doit être >= min_overlap_ratio"
        )


def _failed_session(
    *,
    session_id: str,
    duration_s: float,
    offset_s: float,
    language: str,
    provenance: str,
    source: str | None,
    error: str,
) -> AlignedSession:
    session = AlignedSession(
        session_id=session_id,
        duration_s=duration_s,
        utterances=(),
        offset_s=offset_s,
        language=language,
        provenance=provenance,
        source=source,
        status="failed",
        error=error,
    )
    session.validate()
    return session


def _decisions_for_segment(
    segment: TranscriptSegment,
    *,
    known_turns: Sequence[SpeakerTurn],
    config: AlignmentConfig,
) -> list[_WordDecision]:
    language = segment.language
    if segment.words:
        return [
            _decide_interval(
                start_s=word.start_s,
                end_s=word.end_s,
                text=word.text,
                language=language,
                known_turns=known_turns,
                config=config,
            )
            for word in segment.words
            if word.text.strip()
        ]
    # Pas d'horodatage de mot : alignement grossier sur le segment, ou unknown.
    return [
        _decide_interval(
            start_s=segment.start_s,
            end_s=segment.end_s,
            text=segment.text,
            language=language,
            known_turns=known_turns,
            config=config,
        )
    ]


def _decide_interval(
    *,
    start_s: float,
    end_s: float,
    text: str,
    language: str,
    known_turns: Sequence[SpeakerTurn],
    config: AlignmentConfig,
) -> _WordDecision:
    duration = end_s - start_s
    if duration <= 0:
        return _WordDecision(
            start_s=start_s,
            end_s=end_s,
            text=text,
            language=language,
            speaker_id=None,
            attribution="unknown",
            overlap=False,
        )
    covers: list[tuple[str, float]] = []
    for turn in known_turns:
        ratio = _overlap_ratio(start_s, end_s, turn.start_s, turn.end_s)
        if ratio >= config.min_overlap_ratio and turn.speaker_id is not None:
            covers.append((turn.speaker_id, ratio))
    covers.sort(key=lambda item: item[1], reverse=True)

    if not covers:
        return _WordDecision(
            start_s=start_s,
            end_s=end_s,
            text=text,
            language=language,
            speaker_id=None,
            attribution="unknown",
            overlap=False,
        )

    best_speaker, best_ratio = covers[0]
    others = [item for item in covers[1:] if item[0] != best_speaker]
    ambiguous = [
        item for item in others if item[1] >= config.min_overlap_ratio
    ]
    if ambiguous:
        # Plusieurs voix distinctes sur le même intervalle : ne pas forcer.
        return _WordDecision(
            start_s=start_s,
            end_s=end_s,
            text=text,
            language=language,
            speaker_id=None,
            attribution="overlap",
            overlap=True,
        )
    # Un seul candidat insuffisant (< exclusive_ratio) reste unknown.
    if best_ratio >= config.exclusive_ratio:
        return _WordDecision(
            start_s=start_s,
            end_s=end_s,
            text=text,
            language=language,
            speaker_id=best_speaker,
            attribution="attributed",
            overlap=False,
        )
    return _WordDecision(
        start_s=start_s,
        end_s=end_s,
        text=text,
        language=language,
        speaker_id=None,
        attribution="unknown",
        overlap=False,
    )


def _overlap_ratio(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    start = max(left_start, right_start)
    end = min(left_end, right_end)
    if end <= start:
        return 0.0
    duration = left_end - left_start
    if duration <= 0:
        return 0.0
    return (end - start) / duration


def _merge_decisions(
    decisions: Iterable[_WordDecision],
    *,
    session_id: str,
    audio_available: bool,
) -> list[AlignedUtterance]:
    """Fusionne les mots consecutifs de même attribution / locuteur.

    Un mot n'est ni perdu ni compté deux fois.
    """
    merged: list[AlignedUtterance] = []
    pending: _WordDecision | None = None
    for decision in decisions:
        if pending is None:
            pending = decision
            continue
        same = (
            pending.attribution == decision.attribution
            and pending.speaker_id == decision.speaker_id
            and pending.overlap == decision.overlap
            and pending.language == decision.language
            and abs(pending.end_s - decision.start_s) <= 0.05
        )
        if same:
            pending = _WordDecision(
                start_s=pending.start_s,
                end_s=decision.end_s,
                text=_join_text(pending.text, decision.text),
                language=pending.language,
                speaker_id=pending.speaker_id,
                attribution=pending.attribution,
                overlap=pending.overlap,
            )
            continue
        merged.append(
            _utterance_from_decision(
                pending,
                session_id=session_id,
                audio_available=audio_available,
            )
        )
        pending = decision
    if pending is not None:
        merged.append(
            _utterance_from_decision(
                pending,
                session_id=session_id,
                audio_available=audio_available,
            )
        )
    return merged


def _utterance_from_decision(
    decision: _WordDecision,
    *,
    session_id: str,
    audio_available: bool,
) -> AlignedUtterance:
    return AlignedUtterance(
        session_id=session_id,
        start_s=decision.start_s,
        end_s=decision.end_s,
        text=decision.text.strip(),
        language=decision.language,
        speaker_id=decision.speaker_id,
        attribution=decision.attribution,
        audio_available=audio_available,
        overlap=decision.overlap,
    )


def _join_text(left: str, right: str) -> str:
    left_part = left.rstrip()
    right_part = right.lstrip()
    if not left_part:
        return right_part
    if not right_part:
        return left_part
    if left_part[-1] in "([{'\"«" or right_part[0] in ".,;:!?)]}%»":
        return f"{left_part}{right_part}"
    return f"{left_part} {right_part}"
