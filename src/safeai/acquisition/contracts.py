"""Contrats de données du module d'acquisition.

Ces structures sont le format interne stable consommé par la transcription,
la diarisation et l'analyse acoustique. Une sortie invalide est rejetée
ici plutôt que d'être propagée silencieusement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np

SourceType = Literal["microphone", "file", "synthetic"]
TARGET_SAMPLE_RATE = 16000
TIMESTAMP_TOLERANCE_S = 0.001


class StreamStatus(str, Enum):
    """Disponibilité technique du flux — distincte du silence comportemental."""

    ACTIVE = "ACTIVE"
    INTERRUPTED = "INTERRUPTED"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class MuteStatus(str, Enum):
    """État de coupure matérielle/logicielle du micro, s'il est observable."""

    MUTED = "MUTED"
    UNMUTED = "UNMUTED"
    UNKNOWN = "UNKNOWN"


class SpeechActivity(str, Enum):
    """Présence de parole détectée, sans interprétation du contenu."""

    SPEECH = "SPEECH"
    NON_SPEECH = "NON_SPEECH"
    UNKNOWN = "UNKNOWN"


class QualityStatus(str, Enum):
    """Exploitabilité technique du signal, distincte du niveau de risque."""

    OK = "OK"
    DEGRADED = "DEGRADED"
    UNUSABLE = "UNUSABLE"
    UNKNOWN = "UNKNOWN"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_session_id(when: datetime | None = None) -> str:
    stamp = (when or utc_now()).strftime("%Y%m%dT%H%M%SZ")
    return f"session_{stamp}_{uuid4().hex[:8]}"


@dataclass(frozen=True)
class SessionClock:
    """Horloge de session : temps relatif monotone + origine civile."""

    session_id: str
    started_at_utc: datetime
    monotonic_origin: float


@dataclass(frozen=True)
class SourceMetadata:
    source_type: SourceType
    original_sample_rate: int
    original_channels: int
    duration_s: float | None = None
    device_name: str | None = None
    device_id: str | None = None
    file_path: str | None = None
    downmix: Literal["average", "none"] = "none"


@dataclass
class SourceConfig:
    source_type: SourceType
    device_id: str | None = None
    file_path: Path | None = None
    target_sample_rate: int = TARGET_SAMPLE_RATE
    channels: int = 1
    chunk_duration_s: float = 1.0
    recording_consent: bool = False
    session_id: str | None = None

    def __post_init__(self) -> None:
        if self.file_path is not None and not isinstance(self.file_path, Path):
            self.file_path = Path(self.file_path)
        if self.chunk_duration_s <= 0:
            raise ValueError("chunk_duration_s must be > 0")
        if self.target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be > 0")
        if self.channels < 1:
            raise ValueError("channels must be >= 1")


@dataclass(frozen=True)
class AudioChunk:
    """Bloc audio normalisé, horodaté, destiné aux modules suivants."""

    session_id: str
    source_type: SourceType
    start_s: float
    end_s: float
    sample_rate: int
    samples: np.ndarray
    channels: int = 1
    stream_status: StreamStatus = StreamStatus.ACTIVE
    mute_status: MuteStatus = MuteStatus.UNKNOWN
    samples_dropped: int = 0

    def validate(self) -> None:
        if self.sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(f"Expected {TARGET_SAMPLE_RATE} Hz, got {self.sample_rate}")
        if self.channels != 1 or self.samples.ndim != 1:
            raise ValueError("Expected mono audio")
        if self.samples.dtype != np.float32:
            raise ValueError("Expected float32 samples")
        if not np.all(np.isfinite(self.samples)):
            raise ValueError("Invalid audio samples")
        if self.end_s <= self.start_s:
            raise ValueError("Invalid timestamps")
        duration = len(self.samples) / self.sample_rate
        if abs((self.end_s - self.start_s) - duration) > TIMESTAMP_TOLERANCE_S:
            raise ValueError("Timestamp/sample mismatch")
        if len(self.samples) == 0:
            raise ValueError("Empty audio chunk")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class SpeechSegment:
    """Intervalle détecté comme contenant de la parole."""

    session_id: str
    start_s: float
    end_s: float
    speech_detected: bool
    vad_mean_probability: float
    audio_available: bool
    samples: np.ndarray | None = None
    sample_rate: int = TARGET_SAMPLE_RATE

    def validate(self) -> None:
        if self.end_s <= self.start_s:
            raise ValueError("Invalid speech segment timestamps")
        if not 0.0 <= self.vad_mean_probability <= 1.0:
            raise ValueError("VAD probability must be in [0, 1]")
        if self.sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(f"Expected {TARGET_SAMPLE_RATE} Hz, got {self.sample_rate}")
        if self.audio_available:
            if self.samples is None or len(self.samples) == 0:
                raise ValueError("audio_available is true but no samples were provided")
            if self.samples.ndim != 1:
                raise ValueError("Expected mono speech samples")
            if self.samples.dtype != np.float32:
                raise ValueError("Expected float32 speech samples")
            if not np.all(np.isfinite(self.samples)):
                raise ValueError("Invalid speech samples")
            duration = len(self.samples) / self.sample_rate
            if abs((self.end_s - self.start_s) - duration) > TIMESTAMP_TOLERANCE_S:
                raise ValueError("Timestamp/sample mismatch")
        elif self.samples is not None:
            raise ValueError("audio_available is false but samples were provided")


@dataclass(frozen=True)
class AudioQualityEvent:
    """Événement technique de qualité ou de disponibilité — pas un niveau de risque."""

    session_id: str
    start_s: float
    end_s: float
    stream_status: StreamStatus
    mute_status: MuteStatus
    speech_activity: SpeechActivity
    quality_status: QualityStatus
    rms: float
    peak: float
    clipping_ratio: float
    noise_estimate: float
    silence_duration_s: float
    buffer_overflow: bool
    reason: str
    samples_dropped: int = 0
    speech_active_at_end: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "stream_status": self.stream_status.value,
            "mute_status": self.mute_status.value,
            "speech_activity": self.speech_activity.value,
            "speech_active_at_end": self.speech_active_at_end,
            "quality_status": self.quality_status.value,
            "rms": self.rms,
            "peak": self.peak,
            "clipping_ratio": self.clipping_ratio,
            "noise_estimate": self.noise_estimate,
            "silence_duration_s": self.silence_duration_s,
            "buffer_overflow": self.buffer_overflow,
            "samples_dropped": self.samples_dropped,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class VADFrame:
    start_s: float
    end_s: float
    probability: float


@dataclass
class AcquisitionStatus:
    running: bool
    session_id: str | None
    source_type: SourceType | None
    stream_status: StreamStatus
    mute_status: MuteStatus
    last_speech_activity: SpeechActivity
    last_quality_status: QualityStatus
    chunks_published: int
    speech_segments_published: int
    quality_events_published: int
    samples_dropped: int
    started_at_utc: datetime | None
    elapsed_s: float
    target_sample_rate: int
    backend_vad: str
    publisher_drops: int = 0
    speech_present_in_window: bool = False
    speech_active_at_end: bool = False
    incidents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "running": self.running,
            "session_id": self.session_id,
            "source_type": self.source_type,
            "stream_status": self.stream_status.value,
            "mute_status": self.mute_status.value,
            "last_speech_activity": self.last_speech_activity.value,
            "last_quality_status": self.last_quality_status.value,
            "chunks_published": self.chunks_published,
            "speech_segments_published": self.speech_segments_published,
            "quality_events_published": self.quality_events_published,
            "samples_dropped": self.samples_dropped,
            "started_at_utc": self.started_at_utc.isoformat() if self.started_at_utc else None,
            "elapsed_s": self.elapsed_s,
            "target_sample_rate": self.target_sample_rate,
            "backend_vad": self.backend_vad,
            "publisher_drops": self.publisher_drops,
            "speech_present_in_window": self.speech_present_in_window,
            "speech_active_at_end": self.speech_active_at_end,
            "incidents": list(self.incidents),
        }


@dataclass(frozen=True)
class TimelineRow:
    """Ligne de diagnostic : parole ou silence, avec qualité associée."""

    start_s: float
    end_s: float
    kind: Literal["speech", "silence"]
    quality_status: QualityStatus
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "kind": self.kind,
            "label": "Parole" if self.kind == "speech" else "Silence",
            "quality_status": self.quality_status.value,
            "reason": self.reason,
        }
