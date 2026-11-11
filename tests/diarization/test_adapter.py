from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from safeai.acquisition.contracts import TARGET_SAMPLE_RATE
from safeai.diarization.adapters import (
    MODEL_UNAVAILABLE_MESSAGE,
    PyannoteAdapter,
    _load_waveform,
    anonymize_labels,
    turns_from_native_segments,
)
from safeai.diarization.config import DiarizationConfig
from safeai.diarization.errors import (
    InvalidDiarizationError,
    ModelInferenceError,
    ModelUnavailableError,
)
from safeai.diarization.service import DiarizationService


def test_native_labels_become_anonymous_ids():
    mapping = anonymize_labels(["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"])
    assert mapping == {"SPEAKER_00": "speaker_00", "SPEAKER_01": "speaker_01"}


def test_offset_is_added_once_to_native_times():
    turns = turns_from_native_segments(
        "session_001",
        [(2.0, 5.0, "SPEAKER_00")],
        offset_s=120.0,
    )
    assert turns[0].start_s == 122.0
    assert turns[0].end_s == 125.0
    assert turns[0].speaker_id == "speaker_00"


def test_missing_model_is_an_explicit_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("pyannote"):
            raise ImportError("pyannote.audio")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    adapter = PyannoteAdapter(DiarizationConfig())
    with pytest.raises(ModelUnavailableError, match="indisponible"):
        adapter.load()
    assert MODEL_UNAVAILABLE_MESSAGE in str(ModelUnavailableError(MODEL_UNAVAILABLE_MESSAGE))


def test_adapter_does_not_force_a_speaker_count():
    import inspect

    source = inspect.getsource(PyannoteAdapter._run_pipeline)
    assert "num_speakers" not in source


def test_missing_wav_is_invalid():
    adapter = PyannoteAdapter(DiarizationConfig())
    with pytest.raises(InvalidDiarizationError, match="introuvable"):
        adapter.diarize(Path("data/local_dev/absent.wav"), session_id="session_missing")


def test_stereo_48k_is_normalized_to_16k_mono(tmp_path: Path):
    stereo = np.stack(
        [np.ones(48_000, dtype=np.float32), np.zeros(48_000, dtype=np.float32)],
        axis=1,
    )
    wav = tmp_path / "stereo48k.wav"
    sf.write(str(wav), stereo, 48_000)
    samples, rate = _load_waveform(wav)
    assert rate == TARGET_SAMPLE_RATE
    assert samples.ndim == 1
    assert samples.dtype == np.float32
    assert abs(len(samples) / rate - 1.0) < 0.02
    assert samples.mean() == pytest.approx(0.5, abs=0.02)


def test_inference_failure_is_not_a_missing_model(tmp_path: Path):
    wav = tmp_path / "tone.wav"
    sf.write(str(wav), np.zeros(16_000, dtype=np.float32), 16_000)

    class _Boom:
        def __call__(self, *_args, **_kwargs):
            raise RuntimeError("mémoire insuffisante")

    adapter = PyannoteAdapter(DiarizationConfig())
    adapter._pipeline = _Boom()
    with pytest.raises(ModelInferenceError, match="inférence"):
        adapter.diarize(wav, session_id="session_oom")


def test_load_passes_the_pinned_revision(monkeypatch):
    import sys
    import types

    captured: dict[str, object] = {}
    fake_audio = types.ModuleType("pyannote.audio")

    class _Pipeline:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            captured["model_id"] = model_id
            captured["kwargs"] = kwargs
            return object()

    fake_audio.Pipeline = _Pipeline
    monkeypatch.setitem(sys.modules, "pyannote", types.ModuleType("pyannote"))
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)
    adapter = PyannoteAdapter(
        DiarizationConfig(model_version="3533c8cf8e369892e6b79ff1bf80f7b0286a54ee")
    )
    adapter.load()
    assert captured["model_id"] == "pyannote/speaker-diarization-community-1"
    assert captured["kwargs"]["revision"] == "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"


def test_service_uses_injected_adapter(tmp_path: Path):
    wav = tmp_path / "tone.wav"
    sf.write(str(wav), np.zeros(16_000, dtype=np.float32), 16_000)

    class _Fake:
        def diarize(self, path, *, session_id, offset_s):
            from safeai.diarization.contracts import DiarizationResult

            result = DiarizationResult(
                session_id=session_id,
                duration_s=1.0,
                status="empty",
                offset_s=offset_s or 0.0,
            )
            result.validate()
            return result

    service = DiarizationService(adapter=_Fake())
    result = service.run_file(wav, session_id="session_fake")
    assert result.status == "empty"
    assert result.speakers_estimated == 0
    assert result.to_rttm() == ""
