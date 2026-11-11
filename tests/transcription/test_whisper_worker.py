"""Acquisition vers WhisperWorker : abonnement, mémoire bornée, texte pendant la capture."""

from __future__ import annotations

import threading

import numpy as np
import pytest
import soundfile as sf

from safeai.acquisition.config import AcquisitionConfig, QueueSettings, VADSettings
from safeai.acquisition.contracts import new_session_id
from safeai.acquisition.service import AcquisitionService
from safeai.acquisition.signals import silence, speech_like
from safeai.acquisition.sources import LiveReplaySource, SyntheticSource
from safeai.acquisition.vad import EnergyVADDetector
from safeai.transcription.contracts import TranscriptSegment
from safeai.transcription.worker import WhisperWorker


def _config(**queues: int) -> AcquisitionConfig:
    return AcquisitionConfig(
        vad=VADSettings(backend="energy"),
        queues=QueueSettings(
            audio_maxsize=queues.get("audio_maxsize", 64),
            speech_maxsize=queues.get("speech_maxsize", 64),
            quality_maxsize=queues.get("quality_maxsize", 128),
        ),
    )


def _service(config: AcquisitionConfig | None = None) -> AcquisitionService:
    chosen = config or _config()
    return AcquisitionService(config=chosen, vad=EnergyVADDetector(chosen.vad))


class ScriptedTranscriber:
    def __init__(self, text: str = "bonjour", gate: threading.Event | None = None) -> None:
        self.text = text
        self.gate = gate
        self.seen_chunks: list[int] = []
        self.service: AcquisitionService | None = None
        self.entered = threading.Event()

    def transcribe_speech(self, segment, source_segment_id: str):
        self.entered.set()
        if self.service is not None:
            self.seen_chunks.append(self.service._chunks_published)
        if self.gate is not None:
            assert self.gate.wait(timeout=2.0)
        return [
            TranscriptSegment(
                session_id=segment.session_id,
                source_segment_id=source_segment_id,
                start_s=segment.start_s,
                end_s=segment.end_s,
                text=self.text,
                language="fr",
                status="completed",
            )
        ]


class FailingTranscriber:
    def transcribe_speech(self, segment, source_segment_id: str):
        raise RuntimeError("modèle indisponible")


def test_subscriber_receives_every_chunk_of_a_replay(tmp_path) -> None:
    path = tmp_path / "three.wav"
    sf.write(str(path), np.zeros(48000, dtype=np.float32), 16000)
    service = _service()
    audio = service.subscribe_audio()
    state = {"t": 0.0}
    source = LiveReplaySource(
        path,
        session_id="replay",
        chunk_duration_s=1.0,
        clock=lambda: state["t"],
        sleeper=lambda delay: None,
    )
    assert service._publisher.audio.subscriber_count == 1
    result = service.run_source(source)
    assert service._publisher.audio.subscriber_count == 1
    assert result.chunks_published == 3
    assert result.publisher_drops == 0
    received = []
    while not audio.empty():
        received.append(audio.get_nowait())
    assert [chunk.end_s for chunk in received] == [1.0, 2.0, 3.0]


def test_live_history_stays_bounded_while_the_count_grows() -> None:
    config = _config(audio_maxsize=4)
    samples = np.concatenate([silence(1.0) for _ in range(10)])
    result = _service(config).run_source(
        SyntheticSource(samples, session_id=new_session_id(), chunk_duration_s=1.0)
    )
    assert result.chunks_published == 10
    assert len(result.chunks) == 4
    assert result.chunks[0].start_s == pytest.approx(6.0)


def test_short_speech_is_published_while_capture_continues() -> None:
    service = _service()
    speech = service.subscribe_speech_segments()
    transcriber = ScriptedTranscriber("bonjour")
    transcriber.service = service
    worker = WhisperWorker(transcriber, speech, clock=lambda: 0.0)
    worker.start()
    samples = np.concatenate([speech_like(1.0), silence(3.0)])
    try:
        result = service.run_source(SyntheticSource(samples, session_id=new_session_id()))
    finally:
        worker.finish()
    assert result.speech_segments_published >= 1
    assert worker.received_count >= 1
    assert worker.published[-1].segment.text == "bonjour"
    assert worker.published[-1].segment.status == "completed"
    assert transcriber.seen_chunks[0] < result.chunks_published


def test_continuous_conversation_publishes_each_closed_segment() -> None:
    service = _service()
    speech = service.subscribe_speech_segments()
    worker = WhisperWorker(ScriptedTranscriber("suite"), speech, clock=lambda: 0.0)
    worker.start()
    pieces = [item for _ in range(3) for item in (speech_like(1.0), silence(1.0))]
    try:
        result = service.run_source(SyntheticSource(np.concatenate(pieces), session_id=new_session_id()))
    finally:
        worker.finish()
    assert result.speech_segments_published >= 2
    assert worker.received_count == result.speech_segments_published
    assert len(worker.context) == worker.received_count


def test_prolonged_silence_publishes_no_text() -> None:
    service = _service()
    speech = service.subscribe_speech_segments()
    worker = WhisperWorker(ScriptedTranscriber("bonjour"), speech, clock=lambda: 0.0)
    worker.start()
    try:
        result = service.run_source(SyntheticSource(silence(6.0), session_id=new_session_id()))
    finally:
        worker.finish()
    assert result.chunks_published == 6
    assert result.speech_segments_published == 0
    assert worker.received_count == 0
    assert list(worker.published) == []


def test_slow_consumer_reports_queue_saturation() -> None:
    config = _config(speech_maxsize=1)
    service = _service(config)
    speech = service.subscribe_speech_segments()
    gate = threading.Event()
    transcriber = ScriptedTranscriber("lent", gate=gate)
    worker = WhisperWorker(transcriber, speech, clock=lambda: 0.0)
    worker.start()
    pieces = [item for _ in range(6) for item in (speech_like(1.0), silence(1.0))]
    try:
        result = service.run_source(SyntheticSource(np.concatenate(pieces), session_id=new_session_id()))
        assert transcriber.entered.wait(timeout=2.0)
        assert result.publisher_drops > 0
        assert any("abandonne" in incident for incident in result.incidents)
        assert result.chunks_published == 12
    finally:
        gate.set()
        worker.finish()


def test_transcriber_error_becomes_failed_and_capture_finishes() -> None:
    service = _service()
    speech = service.subscribe_speech_segments()
    worker = WhisperWorker(FailingTranscriber(), speech, clock=lambda: 0.0)
    worker.start()
    try:
        result = service.run_source(
            SyntheticSource(np.concatenate([speech_like(1.0), silence(1.0)]), session_id=new_session_id())
        )
    finally:
        worker.finish()
    assert result.chunks_published == 2
    assert worker.errors >= 1
    assert worker.published[-1].segment.status == "failed"
    worker.published[-1].segment.validate()


def test_default_clock_stays_on_one_timeline() -> None:
    from safeai.acquisition.contracts import SpeechSegment

    speech = SpeechSegment(
        session_id="s",
        start_s=0.0,
        end_s=1.0,
        speech_detected=True,
        vad_mean_probability=0.9,
        audio_available=True,
        samples=np.zeros(16000, dtype=np.float32),
    )
    worker = WhisperWorker(ScriptedTranscriber("bonjour"), __import__("queue").Queue())
    worker._transcribe(speech)
    assert abs(worker.published[-1].lag_s) < 5.0


def test_lag_subtracts_session_origin_and_audio_end() -> None:
    import queue

    from safeai.acquisition.contracts import SpeechSegment

    speech = SpeechSegment(
        session_id="s",
        start_s=0.0,
        end_s=1.0,
        speech_detected=True,
        vad_mean_probability=0.9,
        audio_available=True,
        samples=np.zeros(16000, dtype=np.float32),
    )
    worker = WhisperWorker(
        ScriptedTranscriber("bonjour"),
        queue.Queue(),
        clock=lambda: 1000.0,
        session_origin_s=990.0,
    )
    heard = worker.subscribe_transcripts()
    worker._transcribe(speech)
    assert worker.published[-1].lag_s == pytest.approx(9.0)
    assert heard.get_nowait().segment.text == "bonjour"
