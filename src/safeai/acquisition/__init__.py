"""Module d'acquisition audio, de qualité et de VAD — premier pilier SafeAI."""

from safeai.acquisition.contracts import (
    AudioChunk,
    AudioQualityEvent,
    MuteStatus,
    QualityStatus,
    SourceConfig,
    SourceType,
    SpeechActivity,
    SpeechSegment,
    StreamStatus,
)
from safeai.acquisition.errors import (
    AcquisitionError,
    ConsentRequiredError,
    InvalidAudioError,
    SourceUnavailableError,
)
from safeai.acquisition.service import AcquisitionService

__all__ = [
    "AcquisitionError",
    "AcquisitionService",
    "AudioChunk",
    "AudioQualityEvent",
    "ConsentRequiredError",
    "InvalidAudioError",
    "MuteStatus",
    "QualityStatus",
    "SourceConfig",
    "SourceType",
    "SourceUnavailableError",
    "SpeechActivity",
    "SpeechSegment",
    "StreamStatus",
]
