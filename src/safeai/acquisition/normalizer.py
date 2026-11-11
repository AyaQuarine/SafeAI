"""Normalisation vers le format interne SafeAI : 16 kHz, mono, float32."""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

from safeai.acquisition.contracts import TARGET_SAMPLE_RATE
from safeai.acquisition.errors import InvalidAudioError


class AudioNormalizer:
    """Convertit un signal hétérogène vers le format interne commun.

    Conversion mono : moyenne arithmétique des canaux. Ce choix est adapté
    à un fichier stéréo mixé, pas à deux micros distincts que la diarisation
    voudrait conserver. V1 documente et applique ce downmix ; la préservation
    multi-canal sera un contrat ultérieur.
    """

    def __init__(self, target_sample_rate: int = TARGET_SAMPLE_RATE) -> None:
        if target_sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(f"V1 impose {TARGET_SAMPLE_RATE} Hz.")
        self.target_sample_rate = target_sample_rate

    def to_mono(self, samples: np.ndarray) -> np.ndarray:
        array = np.asarray(samples)
        if array.size == 0:
            raise InvalidAudioError("Signal audio vide.")
        if array.ndim == 1:
            return np.ascontiguousarray(array, dtype=np.float32)
        if array.ndim != 2:
            raise InvalidAudioError(f"Rang audio non supporté : {array.ndim}")
        if array.shape[1] == 1:
            return np.ascontiguousarray(array[:, 0], dtype=np.float32)
        return np.ascontiguousarray(np.mean(array, axis=1), dtype=np.float32)

    def resample(self, samples: np.ndarray, source_rate: int) -> np.ndarray:
        if source_rate <= 0:
            raise InvalidAudioError(f"Fréquence d'échantillonnage invalide : {source_rate}")
        mono = self.to_mono(samples)
        if source_rate == self.target_sample_rate:
            return np.ascontiguousarray(mono, dtype=np.float32)

        up_down = gcd(source_rate, self.target_sample_rate)
        up = self.target_sample_rate // up_down
        down = source_rate // up_down
        resampled = resample_poly(mono, up, down)
        return np.ascontiguousarray(resampled, dtype=np.float32)

    def convert(self, samples: np.ndarray, source_rate: int) -> np.ndarray:
        converted = self.resample(samples, source_rate)
        if converted.size == 0:
            raise InvalidAudioError("Le signal normalisé est vide.")
        if not np.all(np.isfinite(converted)):
            raise InvalidAudioError("Le signal contient des échantillons non finis.")
        return converted

    def expected_output_length(self, n_input: int, source_rate: int) -> int:
        return int(round(n_input * self.target_sample_rate / source_rate))
