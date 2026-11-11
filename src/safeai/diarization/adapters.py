"""Adaptation d'un backend de diarisation vers SpeakerTurn.

Le modèle n'est chargé qu'à la demande. Un échec de téléchargement ou
d'import n'est jamais transformé en résultat vide.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

from safeai.acquisition.contracts import TARGET_SAMPLE_RATE
from safeai.acquisition.errors import InvalidAudioError
from safeai.acquisition.normalizer import AudioNormalizer
from safeai.diarization.config import DiarizationConfig
from safeai.diarization.contracts import DiarizationResult, SpeakerTurn
from safeai.diarization.errors import (
    InvalidDiarizationError,
    ModelInferenceError,
    ModelUnavailableError,
)

MODEL_UNAVAILABLE_MESSAGE = (
    "Modèle de diarisation indisponible. Installez pyannote.audio, acceptez "
    "les conditions de pyannote/speaker-diarization-community-1, et fournissez "
    "un jeton Hugging Face via HF_TOKEN. Le jeton ne doit figurer ni dans Git "
    "ni dans les rapports."
)


def anonymize_labels(labels: Iterable[str]) -> dict[str, str]:
    """Convertit les étiquettes natives en speaker_00, speaker_01, …"""
    mapping: dict[str, str] = {}
    for label in labels:
        if label not in mapping:
            mapping[label] = f"speaker_{len(mapping):02d}"
    return mapping


def turns_from_native_segments(
    session_id: str,
    segments: Iterable[tuple[float, float, str]],
    *,
    offset_s: float = 0.0,
) -> tuple[SpeakerTurn, ...]:
    items = [(float(start), float(end), str(label)) for start, end, label in segments]
    mapping = anonymize_labels(label for _, _, label in items)
    turns: list[SpeakerTurn] = []
    for start_s, end_s, label in items:
        turns.append(
            SpeakerTurn(
                session_id=session_id,
                start_s=start_s + offset_s,
                end_s=end_s + offset_s,
                speaker_id=mapping[label],
                status="completed",
            )
        )
    return tuple(turns)


def _huggingface_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    try:
        from huggingface_hub.utils import get_token
    except Exception:
        return None
    try:
        return get_token()
    except Exception:
        return None


class PyannoteAdapter:
    """Charge Community-1 à la demande et produit des SpeakerTurn."""

    def __init__(self, config: DiarizationConfig | None = None) -> None:
        self.config = config or DiarizationConfig()
        self._pipeline = None

    def load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise ModelUnavailableError(MODEL_UNAVAILABLE_MESSAGE) from exc
        token = _huggingface_token()
        try:
            kwargs: dict[str, object] = {}
            if token:
                kwargs["token"] = token
            if self.config.model_version:
                kwargs["revision"] = self.config.model_version
            self._pipeline = Pipeline.from_pretrained(self.config.model_id, **kwargs)
        except Exception as exc:
            raise ModelUnavailableError(MODEL_UNAVAILABLE_MESSAGE) from exc
        if self._pipeline is None:
            raise ModelUnavailableError(MODEL_UNAVAILABLE_MESSAGE)

    def diarize(
        self,
        path: str | Path,
        *,
        session_id: str,
        offset_s: float | None = None,
    ) -> DiarizationResult:
        wav = Path(path)
        if not wav.is_file():
            raise InvalidDiarizationError(f"Fichier audio introuvable : {wav}")
        samples, sample_rate = _load_waveform(wav)
        duration_s = len(samples) / sample_rate
        started = time.perf_counter()
        self.load()
        annotation = self._run_pipeline(samples, sample_rate)
        native = list(_iter_native_turns(annotation))
        elapsed = time.perf_counter() - started
        applied_offset = self.config.offset_s if offset_s is None else offset_s
        turns = turns_from_native_segments(session_id, native, offset_s=applied_offset)
        speakers = {turn.speaker_id for turn in turns if turn.speaker_id is not None}
        status: str = "empty" if not turns else "completed"
        result = DiarizationResult(
            session_id=session_id,
            duration_s=duration_s,
            status=status,  # type: ignore[arg-type]
            turns=turns,
            speakers_estimated=len(speakers),
            model_id=self.config.model_id,
            model_version=self.config.model_version,
            config_hash=self.config.fingerprint(),
            offset_s=applied_offset,
            processing_seconds=elapsed,
            real_time_factor=elapsed / duration_s if duration_s > 0 else None,
        )
        result.validate()
        return result

    def _run_pipeline(self, samples, sample_rate: int):
        assert self._pipeline is not None
        options: dict[str, object] = {}
        if self.config.min_speakers is not None:
            options["min_speakers"] = self.config.min_speakers
        if self.config.max_speakers is not None:
            options["max_speakers"] = self.config.max_speakers
        try:
            import torch

            waveform = torch.from_numpy(samples).unsqueeze(0)
        except ImportError:
            import numpy as np

            waveform = np.ascontiguousarray(samples, dtype=np.float32).reshape(1, -1)
        try:
            output = self._pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                **options,
            )
        except Exception as exc:
            raise ModelInferenceError(
                f"L'inférence de diarisation a échoué : {exc}"
            ) from exc
        if self.config.exclusive:
            annotation = getattr(output, "exclusive_speaker_diarization", None)
        else:
            annotation = getattr(output, "speaker_diarization", output)
        if annotation is None:
            raise ModelInferenceError("La sortie exclusive du modèle est absente.")
        return annotation


def _load_waveform(path: Path) -> tuple[object, int]:
    """Lit un WAV et le ramène explicitement à 16 kHz mono float32."""
    import numpy as np
    import soundfile as sf

    try:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise InvalidDiarizationError(f"Lecture impossible : {path}") from exc
    if data.size == 0:
        raise InvalidDiarizationError(f"Fichier audio sans échantillons : {path}")
    try:
        converted = AudioNormalizer().convert(data, int(sample_rate))
    except InvalidAudioError as exc:
        raise InvalidDiarizationError(str(exc)) from exc
    samples = np.ascontiguousarray(converted, dtype=np.float32)
    return samples, TARGET_SAMPLE_RATE


def _iter_native_turns(annotation) -> Iterator[tuple[float, float, str]]:
    if hasattr(annotation, "itertracks"):
        for segment, _, label in annotation.itertracks(yield_label=True):
            yield float(segment.start), float(segment.end), str(label)
        return
    for item in annotation:
        if not isinstance(item, tuple):
            continue
        if len(item) == 2:
            turn, speaker = item
            yield float(turn.start), float(turn.end), str(speaker)
        elif len(item) >= 3:
            turn, _track, speaker = item[:3]
            yield float(turn.start), float(turn.end), str(speaker)
