from __future__ import annotations

from pathlib import Path

import pytest

from safeai.acquisition.cli import main, serialize_result
from safeai.acquisition.config import AcquisitionConfig, VADSettings
from safeai.acquisition.service import downsample_envelope, run_file
from safeai.acquisition.vad import EnergyVADDetector


def test_inspect_cli_on_file(wav_mono_16k: Path, capsys) -> None:
    code = main(["inspect", "--file", str(wav_mono_16k), "--energy-vad"])
    assert code == 0
    output = capsys.readouterr().out
    assert "Output rate" in output
    assert "16000 Hz" in output
    assert "Timestamp errors   : 0" in output
    assert "Signal status      : ACTIVE" in output


def test_demo_cli(capsys) -> None:
    code = main(["demo"])
    assert code == 0
    output = capsys.readouterr().out
    assert "synthetic-demo" in output
    assert "Segments émis" in output


def test_devices_cli_does_not_crash(capsys) -> None:
    assert main(["devices"]) == 0
    capsys.readouterr()


def test_status_exposes_publisher_drops() -> None:
    import numpy as np

    from safeai.acquisition.contracts import AudioChunk
    from safeai.acquisition.service import AcquisitionService

    service = AcquisitionService(config=AcquisitionConfig(vad=VADSettings(backend="energy")))
    service.subscribe_audio()
    for index in range(70):
        service._publisher.publish_audio(
            AudioChunk(
                session_id="qa",
                source_type="synthetic",
                start_s=float(index),
                end_s=float(index + 1),
                sample_rate=16000,
                samples=np.zeros(16000, dtype=np.float32),
            )
        )
    service._note_publisher_drops()
    status = service.get_status()
    assert status.publisher_drops == 6
    assert any("abandonne" in item for item in status.incidents)


def test_record_is_only_for_the_microphone(tmp_path) -> None:
    from safeai.acquisition.cli import main

    code = main(
        [
            "inspect",
            "--file",
            str(tmp_path / "absent.wav"),
            "--record",
            str(tmp_path / "r01.wav"),
            "--energy-vad",
        ]
    )
    assert code == 2


def test_serialize_result_omits_raw_samples(wav_mono_16k: Path) -> None:
    config = AcquisitionConfig(vad=VADSettings(backend="energy"))
    result = run_file(wav_mono_16k, config=config, vad=EnergyVADDetector(config.vad))
    payload = serialize_result(result, "test.wav")
    dumped = str(payload)
    assert "ndarray" not in dumped
    assert payload["timestamp_errors"] == 0
    envelope = downsample_envelope(result.chunks)
    assert envelope["times"]
    assert len(envelope["min"]) == len(envelope["max"])


def test_unavailable_microphone_does_not_create_a_wav(tmp_path) -> None:
    import threading

    from safeai.acquisition.cli import _capture_microphone
    from safeai.acquisition.errors import SourceUnavailableError
    from safeai.acquisition.service import AcquisitionService

    config = AcquisitionConfig(vad=VADSettings(backend="energy"))
    service = AcquisitionService(config=config, vad=EnergyVADDetector(config.vad))
    path = tmp_path / "r01.wav"
    with pytest.raises(SourceUnavailableError):
        _capture_microphone(
            service,
            device_id="999999",
            duration_s=1.0,
            chunk_duration_s=1.0,
            record_path=path,
        )
    assert not path.exists()
    assert not any(thread.name == "safeai-record" and thread.is_alive() for thread in threading.enumerate())


def test_report_keeps_published_count_when_history_is_bounded() -> None:
    import numpy as np

    from safeai.acquisition.config import QueueSettings
    from safeai.acquisition.contracts import AudioChunk, SourceMetadata
    from safeai.acquisition.service import AcquisitionService

    class _GapSource:
        session_id = "gap"
        metadata = SourceMetadata(
            source_type="synthetic",
            original_sample_rate=16000,
            original_channels=1,
            duration_s=11.0,
        )

        def read_chunks(self):
            for index in range(10):
                start = float(index if index == 0 else index + 1)
                samples = np.zeros(16000, dtype=np.float32)
                yield AudioChunk(
                    session_id="gap",
                    source_type="synthetic",
                    start_s=start,
                    end_s=start + 1.0,
                    sample_rate=16000,
                    samples=samples,
                )

        def close(self) -> None:
            return None

    config = AcquisitionConfig(
        vad=VADSettings(backend="energy"),
        queues=QueueSettings(audio_maxsize=4),
    )
    service = AcquisitionService(config=config, vad=EnergyVADDetector(config.vad))
    result = service.run_source(_GapSource())
    assert result.chunks_published == 10
    assert len(result.chunks) == 4
    assert result.timestamp_errors == 0
    assert result.timestamp_errors_total == 1
    payload = serialize_result(result, "gap")
    assert payload["chunks_published"] == 10
    assert payload["chunks_retained"] == 4
    assert payload["chunks_generated"] == 10
    assert payload["timestamp_errors"] == 1


def test_simulated_microphone_recording_keeps_every_received_sample(tmp_path) -> None:
    """Le dernier bloc est publié pendant l'arrêt : le WAV doit tout de même le contenir."""
    import threading
    import time

    import numpy as np
    import soundfile as sf

    from safeai.acquisition.cli import _capture_microphone
    from safeai.acquisition.service import AcquisitionService

    hop = 8000
    tone = np.full(hop, 0.25, dtype=np.float32)

    class _Stream:
        def __init__(self) -> None:
            self._callback = None
            self.active = False

        def bind(self, *, callback):
            self._callback = callback
            return self

        def start(self) -> None:
            self.active = True
            block = tone.reshape(-1, 1)

            def emit() -> None:
                time.sleep(0.05)
                self._callback(block, hop, None, None)

            threading.Thread(target=emit, daemon=True).start()

        def stop(self) -> None:
            self.active = False

        def close(self) -> None:
            self.stop()

    class _SlowVad:
        name = "energy"

        def __init__(self) -> None:
            self._inner = EnergyVADDetector(VADSettings(backend="energy"))

        def reset(self) -> None:
            self._inner.reset()

        def process(self, chunk):
            time.sleep(0.6)
            return self._inner.process(chunk)

    stream = _Stream()
    config = AcquisitionConfig(vad=VADSettings(backend="energy"))
    service = AcquisitionService(config=config, vad=_SlowVad(), stream_factory=stream.bind)
    path = tmp_path / "capture.wav"
    result = _capture_microphone(
        service,
        device_id="1",
        duration_s=0.15,
        chunk_duration_s=0.5,
        record_path=path,
    )
    assert result.chunks_published == 1
    assert not any(thread.name == "safeai-record" and thread.is_alive() for thread in threading.enumerate())
    recorded, rate = sf.read(str(path), dtype="float32", always_2d=False)
    expected = np.concatenate([chunk.samples for chunk in result.chunks])
    assert rate == 16000
    assert recorded.shape == expected.shape
    assert np.allclose(recorded, expected)
    assert abs(len(recorded) / rate - len(expected) / rate) < 1e-9


def test_simulated_capture_writes_the_partial_last_block(tmp_path) -> None:
    """2,5 s capturées par blocs d'une seconde doivent tenir en entier dans le WAV."""
    import numpy as np
    import soundfile as sf

    from safeai.acquisition.cli import _capture_microphone
    from safeai.acquisition.service import AcquisitionService

    total = 40_000
    tone = np.full(total, 0.2, dtype=np.float32)

    class _Stream:
        def __init__(self) -> None:
            self._callback = None
            self.active = False

        def bind(self, *, callback):
            self._callback = callback
            return self

        def start(self) -> None:
            self.active = True
            self._callback(tone.reshape(-1, 1), total, None, None)

        def stop(self) -> None:
            self.active = False

        def close(self) -> None:
            self.stop()

    stream = _Stream()
    config = AcquisitionConfig(vad=VADSettings(backend="energy"))
    service = AcquisitionService(
        config=config,
        vad=EnergyVADDetector(config.vad),
        stream_factory=stream.bind,
    )
    path = tmp_path / "partial.wav"
    result = _capture_microphone(
        service,
        device_id="1",
        duration_s=0.6,
        chunk_duration_s=1.0,
        record_path=path,
    )
    recorded, rate = sf.read(str(path), dtype="float32", always_2d=False)
    assert rate == 16000
    assert len(recorded) == total
    assert np.allclose(recorded, tone)
    assert result.chunks_published == 3
    assert result.chunks[-1].end_s == pytest.approx(2.5)
    assert result.timestamp_errors_total == 0
    assert abs(len(recorded) / rate - 2.5) < 1e-9


def test_injected_stream_stop_is_an_incident_not_silence(tmp_path) -> None:
    """Un flux qui meurt après quelques blocs n'est pas prolongé par du silence."""
    import threading

    import numpy as np
    import soundfile as sf

    from safeai.acquisition.cli import _capture_microphone
    from safeai.acquisition.contracts import QualityStatus, StreamStatus
    from safeai.acquisition.service import AcquisitionService
    from safeai.acquisition.signals import speech_like

    audio = speech_like(2.0)

    class _Stream:
        def __init__(self) -> None:
            self._callback = None
            self.active = False

        def bind(self, *, callback):
            self._callback = callback
            return self

        def start(self) -> None:
            self.active = True
            self._callback(audio.reshape(-1, 1), len(audio), None, None)
            self.active = False

        def stop(self) -> None:
            self.active = False

        def close(self) -> None:
            self.stop()

    stream = _Stream()
    config = AcquisitionConfig(vad=VADSettings(backend="energy"))
    service = AcquisitionService(
        config=config,
        vad=EnergyVADDetector(config.vad),
        stream_factory=stream.bind,
    )
    path = tmp_path / "interrupted.wav"
    result = _capture_microphone(
        service,
        device_id="1",
        duration_s=1.2,
        chunk_duration_s=1.0,
        record_path=path,
    )

    received = np.concatenate([chunk.samples for chunk in result.chunks])
    assert result.chunks_published == 2
    assert len(received) == len(audio)
    assert np.allclose(received, audio)
    assert result.chunks[-1].end_s == pytest.approx(2.0)
    assert result.timestamp_errors_total == 0
    assert result.publisher_drops == 0
    assert result.incidents
    assert any("interrompu" in incident for incident in result.incidents)
    incident = result.quality_events[-1]
    assert incident.stream_status is StreamStatus.UNAVAILABLE
    assert incident.quality_status is QualityStatus.UNUSABLE
    assert incident.reason == "technical_incident"
    assert incident.start_s == pytest.approx(2.0)
    assert incident.end_s == pytest.approx(2.0)
    assert result.timeline
    assert max(row.end_s for row in result.timeline) == pytest.approx(2.0)
    assert all(row.end_s <= 2.0 + 1e-6 for row in result.timeline)

    recorded, rate = sf.read(str(path), dtype="float32", always_2d=False)
    assert rate == 16000
    assert len(recorded) == len(audio)
    assert np.allclose(recorded, audio)
    status = service.get_status()
    assert status.running is False
    assert status.stream_status is StreamStatus.UNAVAILABLE
    assert not any(thread.name == "safeai-record" and thread.is_alive() for thread in threading.enumerate())
