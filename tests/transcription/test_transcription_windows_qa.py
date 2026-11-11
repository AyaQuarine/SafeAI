"""Tests d'audit SafeAI : limites connues de la methode C.

Ces tests expriment des exigences recommandees et peuvent echouer
sur la version de windows.py du 02/11/2026. Ne pas detendre leurs
assertions sans une decision explicite sur le contrat attendu.
"""
import numpy as np
import pytest

from safeai.transcription.contracts import TranscriptSegment, WordTiming
from safeai.transcription.windows import (
    _suppress_repeated_tokens, pause_windows, transcribe_windows, AudioWindow
)


def test_repetitions_reelles_sont_preservees():
    original = TranscriptSegment(
        session_id="s", source_segment_id="w1", start_s=1.0,
        end_s=1.5, text="Non non", language="fr", status="completed",
        words=(WordTiming(1.0, 1.12, "Non"), WordTiming(1.20, 1.35, "non")),
    )
    original.validate()
    result = _suppress_repeated_tokens([original])
    assert [w.text.lower() for seg in result for w in seg.words] == ["non", "non"]


def test_pause_windows_honore_plafond_temporel():
    windows = pause_windows([(0.0, 16.0)], max_window_s=12.0, max_pause_s=0.6)
    assert all(window.end_s - window.start_s <= 12.0 for window in windows)


def test_window_transcription_ne_traite_pas_du_48k_comme_16k():
    class FakeTranscriber:
        def transcribe_samples(self, audio, **kwargs):
            # L'adaptateur doit refuser ou resampler avant de deleguer.
            assert len(audio) == 16000  # 1 seconde normalisee a 16 kHz
            return []

    samples = np.zeros(48000, dtype=np.float32)
    transcribe_windows(
        FakeTranscriber(), samples,
        [AudioWindow(0.0, 1.0, 0.0, 1.0)],
        session_id="s", sample_rate=48000,
    )
