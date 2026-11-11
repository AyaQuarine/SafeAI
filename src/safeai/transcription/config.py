"""Réglages de transcription. Le modèle reste local."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptionConfig:
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "fr"
    task: str = "transcribe"
    beam_size: int = 5
    vad_filter: bool = False
    word_timestamps: bool = True
