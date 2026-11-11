"""Chargement de la configuration de diarisation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from safeai.diarization.errors import ConfigurationError

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "diarization.yaml"


@dataclass(frozen=True)
class DiarizationConfig:
    """Paramètres du backend. Aucun modèle n'est chargé ici."""

    backend: Literal["pyannote"] = "pyannote"
    model_id: str = "pyannote/speaker-diarization-community-1"
    model_version: str | None = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
    min_speakers: int | None = None
    max_speakers: int | None = None
    exclusive: bool = False
    offset_s: float = 0.0

    def __post_init__(self) -> None:
        if self.backend != "pyannote":
            raise ConfigurationError(f"Backend de diarisation inconnu : {self.backend}")
        if not self.model_id.strip():
            raise ConfigurationError("model_id est requis.")
        if self.min_speakers is not None and self.min_speakers < 1:
            raise ConfigurationError("min_speakers doit être >= 1 lorsqu'il est fixé.")
        if self.max_speakers is not None and self.max_speakers < 1:
            raise ConfigurationError("max_speakers doit être >= 1 lorsqu'il est fixé.")
        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers > self.max_speakers
        ):
            raise ConfigurationError("min_speakers ne peut pas dépasser max_speakers.")
        if self.offset_s < 0:
            raise ConfigurationError("offset_s doit être >= 0.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigurationError(f"La section '{name}' doit être un objet YAML.")
    return value


def load_config(path: str | Path | None = None) -> DiarizationConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        return DiarizationConfig()

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigurationError("Le fichier de configuration doit contenir un objet YAML.")

    fields = {**_section(raw, "diarization"), **raw}
    known = {key: fields[key] for key in DiarizationConfig.__dataclass_fields__ if key in fields}
    return DiarizationConfig(**known)
