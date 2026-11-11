"""Contrat commun voix–texte pour les analyses textuelle et vocale.

Une ligne de temps unique : locuteur anonyme, texte, horodatage de session.
Whisper ne fournit pas le locuteur. La diarisation ne fournit pas le texte.
Cet alignement les relie, ou laisse ``speaker_id`` vide en cas d'incertitude.
Ce n'est pas encore le moteur d'alignement : ce sont les données consommées.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from safeai.alignment.errors import InvalidAlignmentError
from safeai.diarization.contracts import SPEAKER_ID_PATTERN, SpeakerTurn, _finite_time

Attribution = Literal["attributed", "unknown", "overlap"]
SessionStatus = Literal["completed", "empty", "failed", "partial"]

_UTTERANCE_REQUIRED = ("session_id", "start_s", "end_s", "attribution")
_SESSION_REQUIRED = ("session_id", "duration_s", "utterances")


@dataclass(frozen=True)
class AlignedUtterance:
    """Un intervalle de session : texte + locuteur, ou incertitude explicite."""

    session_id: str
    start_s: float
    end_s: float
    text: str
    language: str
    speaker_id: str | None
    attribution: Attribution
    audio_available: bool = True
    overlap: bool = False

    def validate(self, *, duration_s: float | None = None, offset_s: float = 0.0) -> None:
        if not self.session_id:
            raise ValueError("session_id manquant")
        if not self.language:
            raise ValueError("langue manquante")
        if self.attribution not in {"attributed", "unknown", "overlap"}:
            raise ValueError(f"attribution inconnue : {self.attribution}")
        if not isinstance(self.audio_available, bool):
            raise ValueError("audio_available doit être un booléen")
        if not isinstance(self.overlap, bool):
            raise ValueError("overlap doit être un booléen")
        SpeakerTurn(
            session_id=self.session_id,
            start_s=self.start_s,
            end_s=self.end_s,
            speaker_id=self.speaker_id,
            status="unknown" if self.speaker_id is None else "completed",
            overlap=self.overlap,
        ).validate(duration_s=duration_s, offset_s=offset_s)
        if self.attribution == "attributed":
            if self.speaker_id is None:
                raise ValueError("une attribution certaine exige speaker_id")
            if self.overlap:
                raise ValueError("un chevauchement n'est pas une attribution unique")
            if not self.text.strip():
                raise ValueError("texte manquant")
        if self.attribution == "unknown" and self.speaker_id is not None:
            raise ValueError("unknown ne porte pas de locuteur inventé")
        if self.attribution == "overlap":
            if not self.overlap:
                raise ValueError("overlap exige overlap=True")
            if self.speaker_id is not None and SPEAKER_ID_PATTERN.fullmatch(self.speaker_id) is None:
                raise ValueError("speaker_id anonyme requis")

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "text": self.text,
            "language": self.language,
            "speaker_id": self.speaker_id,
            "attribution": self.attribution,
            "audio_available": self.audio_available,
            "overlap": self.overlap,
        }


@dataclass(frozen=True)
class AlignedSession:
    """Session alignée, sans échantillons audio dans le JSON."""

    session_id: str
    duration_s: float
    utterances: tuple[AlignedUtterance, ...]
    offset_s: float = 0.0
    language: str = "fr"
    provenance: str = "pipeline"
    source: str | None = None
    status: SessionStatus = "completed"
    error: str | None = None
    diagnostics: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.session_id:
            raise ValueError("session_id manquant")
        if self.status not in {"completed", "empty", "failed", "partial"}:
            raise ValueError(f"statut de session inconnu : {self.status}")
        _finite_time(self.duration_s, name="duration_s")
        _finite_time(self.offset_s, name="offset_s")
        if self.duration_s < 0 or self.offset_s < 0:
            raise ValueError("durée ou offset invalide")
        if self.status == "empty":
            if self.utterances:
                raise ValueError("un résultat empty ne contient pas d'énoncé")
            if self.error:
                raise ValueError("un silence n'emporte pas d'erreur")
        elif self.status == "failed":
            if not self.error:
                raise ValueError("un échec doit porter un message")
        elif self.error:
            raise ValueError("un résultat non failed n'emporte pas d'erreur")
        if self.status == "completed" and not self.utterances:
            raise ValueError("aucun énoncé")
        if self.status == "partial" and not self.utterances:
            raise ValueError("un résultat partial exige au moins un énoncé")
        for item in self.utterances:
            if item.session_id != self.session_id:
                raise ValueError("un énoncé n'appartient pas à la session")
            item.validate(duration_s=self.duration_s, offset_s=self.offset_s)

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "duration_s": self.duration_s,
            "offset_s": self.offset_s,
            "language": self.language,
            "status": self.status,
            "provenance": self.provenance,
            "source": self.source,
            "error": self.error,
            "diagnostics": list(self.diagnostics),
            "utterances": [item.to_dict() for item in self.utterances],
        }


def _require_bool(value: object, *, name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise InvalidAlignmentError(f"{name} doit être un booléen")
    return value


def _utterance_from_dict(
    item: dict[str, object],
    *,
    fallback_language: str,
    index: int,
) -> AlignedUtterance:
    missing = [key for key in _UTTERANCE_REQUIRED if key not in item]
    if missing:
        raise InvalidAlignmentError(
            f"clé obligatoire absente dans l'énoncé {index} : {missing[0]}"
        )
    speaker_id = item.get("speaker_id")
    if speaker_id is not None and not isinstance(speaker_id, str):
        raise InvalidAlignmentError(f"speaker_id de l'énoncé {index} doit être une chaîne ou null")
    try:
        return AlignedUtterance(
            session_id=str(item["session_id"]),
            start_s=float(item["start_s"]),
            end_s=float(item["end_s"]),
            text=str(item.get("text", "")),
            language=str(item.get("language", fallback_language)),
            speaker_id=speaker_id,
            attribution=item["attribution"],  # type: ignore[arg-type]
            audio_available=_require_bool(
                item.get("audio_available"),
                name="audio_available",
                default=True,
            ),
            overlap=_require_bool(item.get("overlap"), name="overlap", default=False),
        )
    except (TypeError, ValueError) as exc:
        raise InvalidAlignmentError(f"énoncé {index} illisible : {exc}") from exc


def session_from_dict(payload: dict[str, object]) -> AlignedSession:
    if not isinstance(payload, dict):
        raise InvalidAlignmentError("JSON de session invalide")
    missing = [key for key in _SESSION_REQUIRED if key not in payload]
    if missing:
        raise InvalidAlignmentError(f"clé obligatoire absente : {missing[0]}")
    raw_items = payload.get("utterances")
    if not isinstance(raw_items, list):
        raise InvalidAlignmentError("utterances manquant")
    language = str(payload.get("language") or "fr")
    utterances: list[AlignedUtterance] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise InvalidAlignmentError(f"énoncé {index} mal formé")
        utterances.append(
            _utterance_from_dict(item, fallback_language=language, index=index)
        )
    raw_status = payload.get("status")
    if raw_status is None:
        status: SessionStatus = "empty" if not utterances else "completed"
    else:
        status = str(raw_status)  # type: ignore[assignment]
    try:
        session = AlignedSession(
            session_id=str(payload["session_id"]),
            duration_s=float(payload["duration_s"]),
            utterances=tuple(utterances),
            offset_s=float(payload.get("offset_s") or 0.0),
            language=language,
            provenance=str(payload.get("provenance") or "pipeline"),
            source=str(payload["source"]) if payload.get("source") else None,
            status=status,
            error=str(payload["error"]) if payload.get("error") else None,
            diagnostics=tuple(
                str(item) for item in (payload.get("diagnostics") or ())
            ),
        )
        session.validate()
    except (TypeError, ValueError) as exc:
        raise InvalidAlignmentError(str(exc)) from exc
    return session


def load_aligned_session(path: str | Path) -> AlignedSession:
    file = Path(path)
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise InvalidAlignmentError(f"Lecture impossible : {file}") from exc
    if not isinstance(payload, dict):
        raise InvalidAlignmentError("JSON de session invalide")
    return session_from_dict(payload)
