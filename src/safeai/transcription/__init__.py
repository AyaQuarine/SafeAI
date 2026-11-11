"""Transcription locale des SpeechSegment — pilier 2, sans attribution de locuteur."""

from safeai.transcription.config import TranscriptionConfig
from safeai.transcription.contracts import TranscriptSegment, WordTiming
from safeai.transcription.whisper import WhisperTranscriber

__all__ = [
    "TranscriptSegment",
    "TranscriptionConfig",
    "WhisperTranscriber",
    "WordTiming",
]
