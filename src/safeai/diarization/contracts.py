"""Contrats de sortie du pilier diarisation.

Les identifiants sont anonymes et limités à une session. Une attribution
incertaine reste ``speaker_id=None`` : le module n'invente pas de nom.

Convention temporelle unique :
``duration_s`` est la durée du fichier traité, ``offset_s`` la position de
début dans la session, ``start_s`` / ``end_s`` les temps absolus de session.
Un extrait de 10 s commençant à 120 s accepte un tour 122–125 s.

``overlap`` signale un chevauchement vocal : deux identifiants distincts et
connus se recouvrent. Une intersection du même locuteur, ou d'un tour
``unknown``, n'est pas un chevauchement vocal confirmé.

La diarisation lit l'audio continu du cahier n° 01, pas seulement les
``SpeechSegment`` de Silero.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Literal

SPEAKER_ID_PATTERN = re.compile(r"^speaker_\d+$")
TurnStatus = Literal["completed", "unknown"]
ResultStatus = Literal["completed", "partial", "empty", "failed"]


def _finite_time(value: float, *, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} doit être un temps fini")


@dataclass(frozen=True)
class SpeakerTurn:
    """Intervalle de parole attribué à une voix anonyme, ou à personne."""

    session_id: str
    start_s: float
    end_s: float
    speaker_id: str | None
    status: TurnStatus = "completed"
    overlap: bool = False
    confidence: float | None = None

    def validate(self, *, duration_s: float | None = None, offset_s: float = 0.0) -> None:
        if not self.session_id:
            raise ValueError("session_id manquant")
        _finite_time(self.start_s, name="start_s")
        _finite_time(self.end_s, name="end_s")
        _finite_time(offset_s, name="offset_s")
        if offset_s < 0:
            raise ValueError("offset_s doit être >= 0")
        if self.start_s < 0:
            raise ValueError("start_s doit être >= 0")
        if self.end_s <= self.start_s:
            raise ValueError("end_s doit être > start_s")
        if duration_s is not None:
            _finite_time(duration_s, name="duration_s")
            if duration_s < 0:
                raise ValueError("duration_s doit être >= 0")
            window_end = offset_s + duration_s
            if self.start_s < offset_s - 1e-3:
                raise ValueError("le tour commence avant l'extrait")
            if self.end_s > window_end + 1e-3:
                raise ValueError("le tour dépasse la durée audio")
        if self.speaker_id is not None and SPEAKER_ID_PATTERN.fullmatch(self.speaker_id) is None:
            raise ValueError("speaker_id doit être anonyme (speaker_N) ou null")
        if self.status not in {"completed", "unknown"}:
            raise ValueError(f"statut de tour inconnu : {self.status}")
        if self.speaker_id is None and self.status != "unknown":
            raise ValueError("un tour sans locuteur doit rester unknown")
        if self.speaker_id is not None and self.status == "unknown":
            raise ValueError("un tour unknown ne porte pas d'identifiant")
        if self.confidence is not None:
            _finite_time(self.confidence, name="confidence")
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError("confidence doit rester dans [0, 1] ou être absente")

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "speaker_id": self.speaker_id,
            "overlap": self.overlap,
            "confidence": self.confidence,
            "status": self.status,
        }


def _intervals_intersect(left: SpeakerTurn, right: SpeakerTurn) -> bool:
    return left.start_s < right.end_s and right.start_s < left.end_s


def _distinct_known_speakers(left: SpeakerTurn, right: SpeakerTurn) -> bool:
    if left.speaker_id is None or right.speaker_id is None:
        return False
    return left.speaker_id != right.speaker_id


def mark_overlaps(turns: tuple[SpeakerTurn, ...] | list[SpeakerTurn]) -> tuple[SpeakerTurn, ...]:
    """Marquer le chevauchement vocal, sans fusionner les tours.

    Deux tours du même locuteur qui s'intersectent restent ``overlap=False``.
    Un tour ``unknown`` qui s'intersecte avec une voix connue reste incertain.
    """
    items = list(turns)
    flags = [False] * len(items)
    for index, left in enumerate(items):
        flags[index] = any(
            _intervals_intersect(left, right) and _distinct_known_speakers(left, right)
            for other, right in enumerate(items)
            if other != index
        )
    return tuple(replace(turn, overlap=flag) for turn, flag in zip(items, flags, strict=True))


@dataclass(frozen=True)
class DiarizationResult:
    """Résultat d'une session de diarisation, sans échantillons audio."""

    session_id: str
    duration_s: float
    status: ResultStatus
    turns: tuple[SpeakerTurn, ...] = ()
    speakers_estimated: int = 0
    model_id: str | None = None
    model_version: str | None = None
    config_hash: str | None = None
    offset_s: float = 0.0
    processing_seconds: float | None = None
    real_time_factor: float | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "turns", mark_overlaps(self.turns))

    @property
    def unknown_turns(self) -> tuple[SpeakerTurn, ...]:
        return tuple(turn for turn in self.turns if turn.speaker_id is None)

    @property
    def overlapping_turns(self) -> tuple[SpeakerTurn, ...]:
        return tuple(turn for turn in self.turns if turn.overlap)

    def validate(self) -> None:
        if not self.session_id:
            raise ValueError("session_id manquant")
        if self.status not in {"completed", "partial", "empty", "failed"}:
            raise ValueError(f"statut de diarisation inconnu : {self.status}")
        _finite_time(self.duration_s, name="duration_s")
        if self.duration_s < 0:
            raise ValueError("duration_s doit être >= 0")
        _finite_time(self.offset_s, name="offset_s")
        if self.offset_s < 0:
            raise ValueError("offset_s doit être >= 0")
        if self.speakers_estimated < 0:
            raise ValueError("speakers_estimated doit être >= 0")
        if self.processing_seconds is not None:
            _finite_time(self.processing_seconds, name="processing_seconds")
            if self.processing_seconds < 0:
                raise ValueError("processing_seconds doit être >= 0")
        if self.real_time_factor is not None:
            _finite_time(self.real_time_factor, name="real_time_factor")
            if self.real_time_factor < 0:
                raise ValueError("real_time_factor doit être >= 0")

        for turn in self.turns:
            if turn.session_id != self.session_id:
                raise ValueError("un tour n'appartient pas à la session")
            turn.validate(duration_s=self.duration_s, offset_s=self.offset_s)

        unique_speakers = {turn.speaker_id for turn in self.turns if turn.speaker_id is not None}
        if self.status == "empty":
            if self.turns:
                raise ValueError("un résultat empty ne contient pas de tour")
            if self.speakers_estimated != 0:
                raise ValueError("un silence n'estime aucun locuteur")
        if self.status == "failed":
            if not self.error:
                raise ValueError("un échec doit porter un message")
        elif self.error:
            raise ValueError("un résultat non failed n'emporte pas d'erreur")
        if self.status == "completed" and unique_speakers and self.speakers_estimated != len(unique_speakers):
            raise ValueError("speakers_estimated ne correspond pas aux identifiants distincts")

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "duration_s": self.duration_s,
            "status": self.status,
            "turns": [turn.to_dict() for turn in self.turns],
            "speakers_estimated": self.speakers_estimated,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "config_hash": self.config_hash,
            "offset_s": self.offset_s,
            "processing_seconds": self.processing_seconds,
            "real_time_factor": self.real_time_factor,
            "unknown_count": len(self.unknown_turns),
            "overlap_count": len(self.overlapping_turns),
            "error": self.error,
        }

    def to_rttm(self, file_id: str = "session") -> str:
        """Export RTTM sans audio. Les tours unknown n'y figurent pas."""
        lines: list[str] = []
        for turn in self.turns:
            if turn.speaker_id is None:
                continue
            duration = turn.end_s - turn.start_s
            lines.append(
                f"SPEAKER {file_id} 1 {turn.start_s:.3f} {duration:.3f} "
                f"<NA> <NA> {turn.speaker_id} <NA> <NA>"
            )
        return "\n".join(lines)
