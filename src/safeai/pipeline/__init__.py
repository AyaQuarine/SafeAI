"""Pipeline commun hors ligne — acquisition, Whisper, Community-1, alignement."""

from safeai.pipeline.errors import PipelineConfigurationError, PipelineError
from safeai.pipeline.offline import (
    AcquisitionSnapshot,
    CommonBundle,
    CommonPipeline,
    DiarizationServiceAdapter,
    LocalAcquisition,
    SoundFileAcquisition,
    SpeechInterval,
    WhisperFileTranscriber,
)

__all__ = [
    "AcquisitionSnapshot",
    "CommonBundle",
    "CommonPipeline",
    "DiarizationServiceAdapter",
    "LocalAcquisition",
    "PipelineConfigurationError",
    "PipelineError",
    "SoundFileAcquisition",
    "SpeechInterval",
    "WhisperFileTranscriber",
]
