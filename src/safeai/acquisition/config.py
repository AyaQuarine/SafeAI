"""Chargement de la configuration d'acquisition."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from safeai.acquisition.contracts import TARGET_SAMPLE_RATE
from safeai.acquisition.errors import ConfigurationError

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "acquisition.yaml"


@dataclass(frozen=True)
class VADSettings:
    backend: Literal["silero", "energy"] = "silero"
    threshold: float = 0.5
    min_speech_duration_s: float = 0.25
    min_silence_duration_s: float = 0.40
    speech_pad_s: float = 0.15
    window_samples: int = 512
    max_speech_segment_s: float = 12.0
    segment_overlap_s: float = 1.0


@dataclass(frozen=True)
class QualitySettings:
    clipping_amplitude: float = 0.99
    clipping_ratio_degraded: float = 0.01
    clipping_ratio_unusable: float = 0.05
    rms_silence: float = 0.005
    noise_degraded: float = 0.04
    peak_intense: float = 0.85


@dataclass(frozen=True)
class QueueSettings:
    audio_maxsize: int = 64
    speech_maxsize: int = 64
    quality_maxsize: int = 128


@dataclass(frozen=True)
class PublisherSettings:
    drop_oldest_on_full: bool = True


@dataclass(frozen=True)
class AcquisitionConfig:
    target_sample_rate: int = TARGET_SAMPLE_RATE
    channels: int = 1
    chunk_duration_s: float = 1.0
    vad: VADSettings = field(default_factory=VADSettings)
    quality: QualitySettings = field(default_factory=QualitySettings)
    queues: QueueSettings = field(default_factory=QueueSettings)
    publisher: PublisherSettings = field(default_factory=PublisherSettings)

    def __post_init__(self) -> None:
        if self.target_sample_rate != TARGET_SAMPLE_RATE:
            raise ConfigurationError(
                f"V1 n'accepte que {TARGET_SAMPLE_RATE} Hz en format interne."
            )
        if self.channels != 1:
            raise ConfigurationError("V1 n'accepte que le mono en format interne.")
        if self.chunk_duration_s < 1.0 or self.chunk_duration_s > 2.0:
            raise ConfigurationError("chunk_duration_s doit être compris entre 1 et 2 secondes.")


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigurationError(f"La section '{name}' doit être un objet YAML.")
    return value


def load_config(path: str | Path | None = None) -> AcquisitionConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        return AcquisitionConfig()

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigurationError("Le fichier de configuration doit contenir un objet YAML.")

    return AcquisitionConfig(
        target_sample_rate=int(raw.get("target_sample_rate", TARGET_SAMPLE_RATE)),
        channels=int(raw.get("channels", 1)),
        chunk_duration_s=float(raw.get("chunk_duration_s", 1.0)),
        vad=VADSettings(**_section(raw, "vad")),
        quality=QualitySettings(**_section(raw, "quality")),
        queues=QueueSettings(**_section(raw, "queues")),
        publisher=PublisherSettings(**_section(raw, "publisher")),
    )
