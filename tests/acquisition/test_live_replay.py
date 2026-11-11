"""Rejeu au rythme réel : les blocs arrivent à l'heure de l'audio, pas d'un coup."""

import numpy as np
import soundfile as sf

from safeai.acquisition.service import AcquisitionService
from safeai.acquisition.sources import LiveReplaySource
from safeai.acquisition.vad import EnergyVADDetector


def _clock():
    state = {"t": 0.0}

    def now() -> float:
        return state["t"]

    def sleep(delay: float) -> None:
        state["t"] += delay

    return state, now, sleep


def test_live_replay_waits_until_each_chunk_would_have_ended(tmp_path) -> None:
    path = tmp_path / "two_seconds.wav"
    sf.write(str(path), np.zeros(32000, dtype=np.float32), 16000)
    state, now, sleep = _clock()
    source = LiveReplaySource(path, session_id="replay", chunk_duration_s=1.0, clock=now, sleeper=sleep)
    chunks = list(source.read_chunks())
    assert [chunk.end_s for chunk in chunks] == [1.0, 2.0]
    assert state["t"] == 2.0
    assert max(abs(lag) for lag in source.emission_lag_s) < 1e-9


def test_live_replay_reports_lag_when_the_consumer_falls_behind(tmp_path) -> None:
    path = tmp_path / "two_seconds.wav"
    sf.write(str(path), np.zeros(32000, dtype=np.float32), 16000)
    state, now, sleep = _clock()
    source = LiveReplaySource(path, session_id="replay", chunk_duration_s=1.0, clock=now, sleeper=sleep)
    iterator = source.read_chunks()
    next(iterator)
    state["t"] += 1.5
    next(iterator)
    assert source.emission_lag_s[-1] == 0.5


def test_live_replay_feeds_the_acquisition_pipeline(tmp_path) -> None:
    path = tmp_path / "two_seconds.wav"
    sf.write(str(path), np.zeros(32000, dtype=np.float32), 16000)
    _state, now, sleep = _clock()
    source = LiveReplaySource(path, session_id="replay", chunk_duration_s=1.0, clock=now, sleeper=sleep)
    result = AcquisitionService(vad=EnergyVADDetector()).run_source(source)
    assert len(result.chunks) == 2
    assert result.timestamp_errors == 0
    assert result.chunks[-1].end_s == 2.0
    assert not any("trou" in incident for incident in result.incidents)
