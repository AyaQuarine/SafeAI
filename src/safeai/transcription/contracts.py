"""Contrat de sortie du pilier transcription.

Whisper ne devine pas le locuteur : speaker_id reste None.
Les temps sont ceux de la session, pas ceux internes au segment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class WordTiming:
    start_s: float
    end_s: float
    text: str

    def to_dict(self) -> dict[str, object]:
        return {"start_s": self.start_s, "end_s": self.end_s, "text": self.text}


@dataclass(frozen=True)
class TranscriptSegment:
    session_id: str
    source_segment_id: str
    start_s: float
    end_s: float
    text: str
    language: str
    status: Literal["completed", "empty", "failed"]
    speaker_id: None = None
    words: tuple[WordTiming, ...] = ()

    def validate(self) -> None:
        if self.speaker_id is not None:
            raise ValueError("speaker_id doit rester null")
        if self.status not in {"completed", "empty", "failed"}:
            raise ValueError(f"statut de transcription inconnu : {self.status}")
        if not self.session_id or not self.source_segment_id:
            raise ValueError("identifiants de transcription manquants")
        if not self.language:
            raise ValueError("langue manquante")
        if not math.isfinite(self.start_s) or not math.isfinite(self.end_s) or self.end_s <= self.start_s:
            raise ValueError("horodatage de transcription invalide")
        if self.status == "completed" and not self.text.strip():
            raise ValueError("transcription complétée sans texte")
        if self.status == "empty" and self.text.strip():
            raise ValueError("transcription vide avec du texte")
        for word in self.words:
            if not math.isfinite(word.start_s) or not math.isfinite(word.end_s) or word.end_s <= word.start_s:
                raise ValueError("horodatage de mot invalide")
            if word.start_s < self.start_s - 1e-3 or word.end_s > self.end_s + 1e-3:
                raise ValueError("mot hors du segment")

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "source_segment_id": self.source_segment_id,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "text": self.text,
            "language": self.language,
            "status": self.status,
            "speaker_id": self.speaker_id,
            "words": [word.to_dict() for word in self.words],
        }
