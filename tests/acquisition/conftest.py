from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from safeai.acquisition.config import AcquisitionConfig, VADSettings
from safeai.acquisition.contracts import TARGET_SAMPLE_RATE
from safeai.acquisition.signals import background_noise, intense_clean, silence, speech_like
from safeai.acquisition.vad import EnergyVADDetector


@pytest.fixture
def energy_config() -> AcquisitionConfig:
    return AcquisitionConfig(vad=VADSettings(backend="energy"))


@pytest.fixture
def energy_vad(energy_config: AcquisitionConfig) -> EnergyVADDetector:
    return EnergyVADDetector(energy_config.vad)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> Path:
    sf.write(str(path), samples, sample_rate, subtype="FLOAT")
    return path


@pytest.fixture
def wav_mono_16k(tmp_path: Path) -> Path:
    audio = speech_like(2.0, TARGET_SAMPLE_RATE)
    return write_wav(tmp_path / "mono16k.wav", audio, TARGET_SAMPLE_RATE)


@pytest.fixture
def wav_48k(tmp_path: Path) -> Path:
    audio = speech_like(2.0, 48_000)
    return write_wav(tmp_path / "speech48k.wav", audio, 48_000)


@pytest.fixture
def wav_silent(tmp_path: Path) -> Path:
    return write_wav(tmp_path / "silent.wav", silence(2.0), TARGET_SAMPLE_RATE)


@pytest.fixture
def wav_stereo_44100(tmp_path: Path) -> Path:
    left = speech_like(1.5, 44_100, f0=140)
    right = speech_like(1.5, 44_100, f0=190)
    stereo = np.stack([left, right], axis=1)
    return write_wav(tmp_path / "stereo44100.wav", stereo, 44_100)


@pytest.fixture
def wav_noisy(tmp_path: Path) -> Path:
    room = background_noise(0.6, amplitude=0.10, seed=11)
    mixed = np.clip(speech_like(1.8) + background_noise(1.8, amplitude=0.10, seed=3), -1.0, 1.0)
    return write_wav(tmp_path / "noisy.wav", np.concatenate([room, mixed]), TARGET_SAMPLE_RATE)


@pytest.fixture
def wav_clipped(tmp_path: Path) -> Path:
    from safeai.acquisition.signals import clipped_signal

    return write_wav(tmp_path / "clipped.wav", clipped_signal(1.5), TARGET_SAMPLE_RATE)


@pytest.fixture
def wav_intense(tmp_path: Path) -> Path:
    return write_wav(tmp_path / "intense.wav", intense_clean(1.5), TARGET_SAMPLE_RATE)
