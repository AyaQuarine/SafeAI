"""Régressions ciblées des seuils qualité SafeAI, sur signal synthétique.

Ajouter ce fichier dans tests/acquisition/ puis exécuter :
    python -m pytest -q tests/acquisition/test_acquisition_quality_thresholds.py

Ces tests vérifient les branches de QualityMonitor ; ils ne remplacent pas
une recette microphone ni l'inspection du passage R07 de 110 à 135 s.
"""

import numpy as np
import pytest

from safeai.acquisition.config import QualitySettings
from safeai.acquisition.contracts import (
    AudioChunk,
    QualityStatus,
    SpeechActivity,
)
from safeai.acquisition.quality import QualityMonitor

RATE = 16_000


def _chunk(samples: np.ndarray) -> AudioChunk:
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    chunk = AudioChunk(
        session_id="quality-regression",
        source_type="synthetic",
        start_s=0.0,
        end_s=len(samples) / RATE,
        sample_rate=RATE,
        samples=samples,
    )
    chunk.validate()
    return chunk


def test_signal_intense_sans_clipping_reste_ok():
    t = np.arange(RATE, dtype=np.float32) / RATE
    clean = 0.92 * np.sin(2 * np.pi * 440 * t)
    event = QualityMonitor().observe(_chunk(clean), SpeechActivity.SPEECH)
    assert event.quality_status is QualityStatus.OK
    assert event.reason == "intense_but_clean"
    assert event.clipping_ratio == 0


@pytest.mark.parametrize(
    "clipped_samples,expected",
    [(320, QualityStatus.DEGRADED), (1600, QualityStatus.UNUSABLE)],
)
def test_clipping_synthetique_declenche_la_bonne_qualite(clipped_samples, expected):
    samples = np.full(RATE, 0.2, dtype=np.float32)
    samples[:clipped_samples] = 1.0
    event = QualityMonitor().observe(_chunk(samples), SpeechActivity.SPEECH)
    assert event.quality_status is expected
    assert event.reason == "clipping"
    assert event.clipping_ratio == pytest.approx(clipped_samples / RATE)


def test_bruit_synthetique_declenche_degraded_seulement_au_dela_du_seuil():
    settings = QualitySettings(noise_degraded=0.04)
    rng = np.random.default_rng(2026)
    noise = rng.normal(size=RATE).astype(np.float32)
    noise /= np.sqrt(np.mean(noise.astype(np.float64) ** 2))

    low = QualityMonitor(settings).observe(
        _chunk(0.015 * noise), SpeechActivity.NON_SPEECH
    )
    high = QualityMonitor(settings).observe(
        _chunk(0.080 * noise), SpeechActivity.NON_SPEECH
    )

    assert low.quality_status is QualityStatus.OK
    assert high.quality_status is QualityStatus.DEGRADED
    assert high.reason == "background_noise"
    assert low.noise_estimate < settings.noise_degraded
    assert high.noise_estimate > settings.noise_degraded
