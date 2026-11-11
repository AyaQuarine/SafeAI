"""Signaux synthétiques déterministes — tests et démonstration, pas une source superviseur."""

from __future__ import annotations

import numpy as np

from safeai.acquisition.contracts import TARGET_SAMPLE_RATE


def time_axis(duration_s: float, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    n = int(round(duration_s * sample_rate))
    return np.linspace(0.0, duration_s, n, endpoint=False, dtype=np.float64)


def silence(duration_s: float, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(round(duration_s * sample_rate)), dtype=np.float32)


def speech_like(
    duration_s: float,
    sample_rate: int = TARGET_SAMPLE_RATE,
    f0: float = 160.0,
    amplitude: float = 0.28,
) -> np.ndarray:
    """Voix synthétique approximative : fondamentale + formants + modulation.

    Suffisante pour un VAD énergétique. Silero, entraîné sur de la parole
    réelle, peut à juste titre la rejeter. Ce n'est pas de la parole réelle
    et ne doit pas être stockée comme donnée personnelle.
    """
    t = time_axis(duration_s, sample_rate)
    glottal = (
        np.sin(2 * np.pi * f0 * t)
        + 0.45 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.20 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    formant = (
        np.sin(2 * np.pi * 700 * t) * 0.35
        + np.sin(2 * np.pi * 1200 * t) * 0.18
    )
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 3.5 * t)
    voiced = (glottal * 0.7 + formant) * envelope
    return np.ascontiguousarray(amplitude * voiced / np.max(np.abs(voiced) + 1e-12), dtype=np.float32)


def background_noise(
    duration_s: float,
    sample_rate: int = TARGET_SAMPLE_RATE,
    amplitude: float = 0.08,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * sample_rate))
    return np.ascontiguousarray(rng.normal(0.0, amplitude, n), dtype=np.float32)


def clipped_signal(
    duration_s: float,
    sample_rate: int = TARGET_SAMPLE_RATE,
    gain: float = 8.0,
) -> np.ndarray:
    raw = speech_like(duration_s, sample_rate, amplitude=0.4) * gain
    return np.ascontiguousarray(np.clip(raw, -1.0, 1.0), dtype=np.float32)


def intense_clean(
    duration_s: float,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Signal fort sans saturation — SC07 : l'intensité n'est pas une anomalie."""
    return speech_like(duration_s, sample_rate, amplitude=0.88)


def demo_meeting_signal(sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Scénario de diagnostic : silence, parole, silence, parole bruitée, saturation."""
    parts = [
        silence(1.4, sample_rate),
        speech_like(2.8, sample_rate),
        silence(1.8, sample_rate),
        np.clip(speech_like(2.5, sample_rate) + background_noise(2.5, sample_rate, 0.09, seed=3), -1.0, 1.0),
        silence(0.6, sample_rate),
        clipped_signal(1.2, sample_rate),
        silence(0.7, sample_rate),
    ]
    return np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)
