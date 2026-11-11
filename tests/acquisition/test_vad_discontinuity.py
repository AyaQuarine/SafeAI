"""Silero doit oublier son état à chaque trou, même sans reste de fenêtre."""

from __future__ import annotations

import numpy as np

from safeai.acquisition.contracts import AudioChunk
from safeai.acquisition.config import VADSettings
from safeai.acquisition.vad import SileroVADDetector


class _Model:
    def __init__(self) -> None:
        self.resets = 0

    def reset_states(self) -> None:
        self.resets += 1

    def __call__(self, audio, sample_rate: int) -> float:
        return 0.01


def _chunk(start_s: float, n: int) -> AudioChunk:
    return AudioChunk(
        session_id="gap",
        source_type="synthetic",
        start_s=start_s,
        end_s=start_s + n / 16000,
        sample_rate=16000,
        samples=np.zeros(n, dtype=np.float32),
    )


def test_silero_resets_when_gap_has_no_residual_tail() -> None:
    model = _Model()
    detector = SileroVADDetector.__new__(SileroVADDetector)
    detector.settings = VADSettings(window_samples=512)
    detector._model = model
    detector._tail = np.zeros(0, dtype=np.float32)
    detector._tail_start_s = 0.0
    detector._has_tail = False
    detector._last_end_s = None

    detector.process(_chunk(0.0, 512))
    assert detector._has_tail is False
    assert model.resets == 0

    detector.process(_chunk(1.0, 512))
    assert model.resets == 1
