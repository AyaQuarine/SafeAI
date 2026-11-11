"""Étape A — FileSource + AudioNormalizer + AudioChunk.

Les trois tests demandés par le cahier des charges n° 01, section 12.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from safeai.acquisition.contracts import TARGET_SAMPLE_RATE, new_session_id
from safeai.acquisition.errors import InvalidAudioError
from safeai.acquisition.normalizer import AudioNormalizer
from safeai.acquisition.signals import silence, speech_like
from safeai.acquisition.sources import FileSource
from tests.acquisition.conftest import write_wav


def test_mono_16khz_is_read_without_unnecessary_transform(wav_mono_16k: Path) -> None:
    original, rate = sf.read(str(wav_mono_16k), dtype="float32")
    assert rate == TARGET_SAMPLE_RATE

    source = FileSource(wav_mono_16k, session_id=new_session_id(), chunk_duration_s=1.0)
    chunks = list(source.read_chunks())

    assert source.metadata.original_sample_rate == TARGET_SAMPLE_RATE
    assert source.metadata.original_channels == 1
    assert source.metadata.downmix == "none"
    assert chunks
    reconstructed = np.concatenate([chunk.samples for chunk in chunks])
    assert reconstructed.dtype == np.float32
    np.testing.assert_allclose(reconstructed, original, atol=1e-7)
    assert abs((len(reconstructed) / TARGET_SAMPLE_RATE) - (len(original) / rate)) < 1e-9
    for chunk in chunks:
        chunk.validate()
    assert chunks[0].start_s == pytest.approx(0.0)
    assert chunks[-1].end_s == pytest.approx(len(original) / rate, abs=0.001)


def test_48khz_is_converted_to_16khz_with_preserved_duration(wav_48k: Path) -> None:
    original, rate = sf.read(str(wav_48k), dtype="float32")
    original_duration = len(original) / rate

    source = FileSource(wav_48k, session_id=new_session_id(), chunk_duration_s=1.0)
    chunks = list(source.read_chunks())

    assert source.metadata.original_sample_rate == 48_000
    reconstructed = np.concatenate([chunk.samples for chunk in chunks])
    output_duration = len(reconstructed) / TARGET_SAMPLE_RATE
    assert reconstructed.dtype == np.float32
    assert reconstructed.ndim == 1
    assert abs(output_duration - original_duration) < 0.02
    for chunk in chunks:
        chunk.validate()
        assert chunk.sample_rate == TARGET_SAMPLE_RATE
    previous = 0.0
    for chunk in chunks:
        assert chunk.start_s == pytest.approx(previous, abs=0.001)
        previous = chunk.end_s


def test_silent_and_invalid_files_are_handled_without_crash(wav_silent: Path, tmp_path: Path) -> None:
    source = FileSource(wav_silent, session_id=new_session_id(), chunk_duration_s=1.0)
    chunks = list(source.read_chunks())
    assert chunks
    for chunk in chunks:
        chunk.validate()
        assert float(np.max(np.abs(chunk.samples))) < 1e-6

    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(InvalidAudioError):
        FileSource(empty, session_id=new_session_id())

    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"this is not audio")
    with pytest.raises(InvalidAudioError):
        FileSource(corrupt, session_id=new_session_id())


def test_normalizer_averages_stereo_channels() -> None:
    stereo = np.stack(
        [np.ones(1600, dtype=np.float32), np.full(1600, 0.5, dtype=np.float32)],
        axis=1,
    )
    mono = AudioNormalizer().to_mono(stereo)
    np.testing.assert_allclose(mono, 0.75, atol=1e-6)


def test_normalizer_rejects_empty_and_nonfinite() -> None:
    normalizer = AudioNormalizer()
    with pytest.raises(InvalidAudioError):
        normalizer.convert(np.zeros(0, dtype=np.float32), TARGET_SAMPLE_RATE)
    with pytest.raises(InvalidAudioError):
        normalizer.convert(np.array([np.nan], dtype=np.float32), TARGET_SAMPLE_RATE)


def test_identical_rate_returns_float32_contiguous(tmp_path: Path) -> None:
    audio = silence(0.5)
    path = write_wav(tmp_path / "half.wav", audio, TARGET_SAMPLE_RATE)
    source = FileSource(path, session_id="session_unit")
    chunk = next(source.read_chunks())
    assert chunk.samples.flags["C_CONTIGUOUS"]
    assert chunk.samples.dtype == np.float32
    assert speech_like(0.1).dtype == np.float32
