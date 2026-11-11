"""Détection d'activité vocale — présence de parole, pas de danger."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from safeai.acquisition.config import VADSettings
from safeai.acquisition.contracts import TARGET_SAMPLE_RATE, AudioChunk, VADFrame
from safeai.acquisition.errors import ConfigurationError

LOGGER = logging.getLogger(__name__)


class VADDetector(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def process(self, chunk: AudioChunk) -> list[VADFrame]:
        ...


class EnergyVADDetector:
    """Détecteur énergétique de repli, déterministe, sans modèle neuronal.

    Il fournit une pseudo-probabilité à partir du RMS. Il n'identifie pas
    des mots. Il n'est utilisé que si `vad.backend: energy` est demandé
    explicitement — jamais comme repli silencieux de Silero.
    """

    name = "energy"

    def __init__(self, settings: VADSettings | None = None) -> None:
        self.settings = settings or VADSettings()
        self._tail = np.zeros(0, dtype=np.float32)
        self._tail_start_s = 0.0
        self._has_tail = False
        self._last_end_s: float | None = None

    def reset(self) -> None:
        self._tail = np.zeros(0, dtype=np.float32)
        self._tail_start_s = 0.0
        self._has_tail = False
        self._last_end_s = None

    def process(self, chunk: AudioChunk) -> list[VADFrame]:
        _discard_state_on_discontinuity(self, chunk)
        frames = _windowed_scores(
            chunk,
            tail=self._tail,
            tail_start_s=self._tail_start_s,
            has_tail=self._has_tail,
            window=self.settings.window_samples,
            scorer=self._score,
            store_tail=self._store_tail,
        )
        self._last_end_s = chunk.end_s
        return frames

    def _store_tail(self, tail: np.ndarray, tail_start_s: float) -> None:
        self._tail = tail
        self._tail_start_s = tail_start_s
        self._has_tail = tail.size > 0

    def _score(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        # 0.02 ≈ parole claire proche ; 0.004 ≈ plancher de silence.
        speech_ref = 0.02
        silence_ref = 0.004
        if rms <= silence_ref:
            return 0.02
        scaled = (rms - silence_ref) / (speech_ref - silence_ref)
        return float(np.clip(scaled, 0.0, 1.0))


class SileroVADDetector:
    """Inférence locale Silero VAD (ONNX), fenêtres de 512 échantillons à 16 kHz."""

    name = "silero"

    def __init__(self, settings: VADSettings | None = None) -> None:
        self.settings = settings or VADSettings()
        self._model = _load_silero()
        self._tail = np.zeros(0, dtype=np.float32)
        self._tail_start_s = 0.0
        self._has_tail = False
        self._last_end_s: float | None = None

    def reset(self) -> None:
        self._tail = np.zeros(0, dtype=np.float32)
        self._tail_start_s = 0.0
        self._has_tail = False
        self._last_end_s = None
        reset = getattr(self._model, "reset_states", None)
        if callable(reset):
            reset()

    def process(self, chunk: AudioChunk) -> list[VADFrame]:
        _discard_state_on_discontinuity(self, chunk)
        frames = _windowed_scores(
            chunk,
            tail=self._tail,
            tail_start_s=self._tail_start_s,
            has_tail=self._has_tail,
            window=self.settings.window_samples,
            scorer=self._score,
            store_tail=self._store_tail,
        )
        self._last_end_s = chunk.end_s
        return frames

    def _store_tail(self, tail: np.ndarray, tail_start_s: float) -> None:
        self._tail = tail
        self._tail_start_s = tail_start_s
        self._has_tail = tail.size > 0

    def _score(self, frame: np.ndarray) -> float:
        try:
            import torch

            tensor = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
            result = self._model(tensor, TARGET_SAMPLE_RATE)
            return float(np.clip(_as_float(result), 0.0, 1.0))
        except Exception:
            result = self._model(np.ascontiguousarray(frame, dtype=np.float32), TARGET_SAMPLE_RATE)
            return float(np.clip(_as_float(result), 0.0, 1.0))


def build_vad(settings: VADSettings | None = None) -> VADDetector:
    """Construit le VAD demandé. Aucun repli silencieux vers le détecteur énergétique.

    Si `backend=silero` et que le modèle est indisponible, une ConfigurationError
    est levée. Le VAD énergétique n'est utilisé que lorsque `backend=energy`.
    """
    settings = settings or VADSettings()
    if settings.backend == "energy":
        return EnergyVADDetector(settings)
    if settings.backend != "silero":
        raise ConfigurationError(f"Backend VAD inconnu : {settings.backend}")
    try:
        return SileroVADDetector(settings)
    except Exception as exc:
        raise ConfigurationError(
            "Silero VAD est demandé mais indisponible. "
            "Installez silero-vad et onnxruntime, ou définissez explicitement "
            "vad.backend: energy pour les tests."
        ) from exc


def _load_silero() -> object:
    from silero_vad import load_silero_vad

    return load_silero_vad(onnx=True)


def _as_float(value: object) -> float:
    item = getattr(value, "item", None)
    if callable(item):
        return float(item())
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def _discard_state_on_discontinuity(detector, chunk: AudioChunk) -> None:
    """Un trou confirmé réinitialise le modèle, même sans reste de fenêtre.

    Le dernier temps traité est la fin du bloc précédent, pas seulement
    l'existence d'un tampon résiduel. Sans cela, un bloc qui tombe juste
    sur une fenêtre laisse l'état Silero traverser la perte.
    """
    last_end = detector._last_end_s
    if last_end is None:
        return
    if abs(chunk.start_s - last_end) <= (1.0 / chunk.sample_rate):
        return
    detector._tail = np.zeros(0, dtype=np.float32)
    detector._tail_start_s = 0.0
    detector._has_tail = False
    reset = getattr(getattr(detector, "_model", None), "reset_states", None)
    if callable(reset):
        reset()


def _windowed_scores(
    chunk: AudioChunk,
    *,
    tail: np.ndarray,
    tail_start_s: float,
    has_tail: bool,
    window: int,
    scorer,
    store_tail,
) -> list[VADFrame]:
    if chunk.sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError("Le VAD V1 attend 16 kHz.")

    if has_tail:
        audio = np.concatenate([tail, chunk.samples])
        origin_s = tail_start_s
    else:
        audio = chunk.samples
        origin_s = chunk.start_s

    complete = (len(audio) // window) * window
    frames: list[VADFrame] = []
    for index in range(0, complete, window):
        frame = audio[index : index + window]
        start_s = origin_s + (index / chunk.sample_rate)
        end_s = start_s + (window / chunk.sample_rate)
        frames.append(
            VADFrame(
                start_s=start_s,
                end_s=end_s,
                probability=float(scorer(frame)),
            )
        )

    leftover = audio[complete:]
    leftover_start = origin_s + (complete / chunk.sample_rate)
    store_tail(np.ascontiguousarray(leftover, dtype=np.float32), leftover_start)
    return frames
