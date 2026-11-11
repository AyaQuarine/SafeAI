"""Diarisation locale des locuteurs — pilier 3, sans identité personnelle.

Le module lit l'audio continu de l'acquisition. Il n'identifie personne
par son nom et n'attend pas les SpeechSegment de Silero.
"""

from safeai.diarization.adapters import PyannoteAdapter
from safeai.diarization.config import DiarizationConfig, load_config
from safeai.diarization.contracts import DiarizationResult, SpeakerTurn
from safeai.diarization.errors import (
    ConfigurationError,
    DiarizationError,
    InvalidDiarizationError,
    ModelInferenceError,
    ModelUnavailableError,
)
from safeai.diarization.service import DiarizationService

__all__ = [
    "ConfigurationError",
    "DiarizationConfig",
    "DiarizationError",
    "DiarizationResult",
    "DiarizationService",
    "InvalidDiarizationError",
    "ModelInferenceError",
    "ModelUnavailableError",
    "PyannoteAdapter",
    "SpeakerTurn",
    "load_config",
]
