"""Sources audio : fichier, microphone, synthétique — même contrat de sortie."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol

import numpy as np

from safeai.acquisition.contracts import (
    TARGET_SAMPLE_RATE,
    AudioChunk,
    MuteStatus,
    SourceConfig,
    SourceMetadata,
    StreamStatus,
)
from safeai.acquisition.errors import InvalidAudioError, SourceUnavailableError
from safeai.acquisition.normalizer import AudioNormalizer

LOGGER = logging.getLogger(__name__)

MuteProbe = Callable[[str | None], MuteStatus]


class AudioSource(Protocol):
    metadata: SourceMetadata

    def read_chunks(self) -> Iterator[AudioChunk]:
        ...

    def close(self) -> None:
        ...


class SampleClock:
    """Compte les échantillons de sortie pour des horodatages continus."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.n = 0

    def stamp(self, n_samples: int) -> tuple[float, float]:
        start_s = self.n / self.sample_rate
        self.n += n_samples
        end_s = self.n / self.sample_rate
        return start_s, end_s


def _chunk_array(
    samples: np.ndarray,
    *,
    session_id: str,
    source_type: str,
    sample_rate: int,
    chunk_duration_s: float,
    stream_status: StreamStatus = StreamStatus.ACTIVE,
    mute_status: MuteStatus = MuteStatus.UNKNOWN,
    samples_dropped: int = 0,
) -> Iterator[AudioChunk]:
    hop = max(1, int(round(chunk_duration_s * sample_rate)))
    clock = SampleClock(sample_rate)
    for index in range(0, len(samples), hop):
        piece = np.ascontiguousarray(samples[index : index + hop], dtype=np.float32)
        if piece.size == 0:
            continue
        start_s, end_s = clock.stamp(len(piece))
        chunk = AudioChunk(
            session_id=session_id,
            source_type=source_type,  # type: ignore[arg-type]
            start_s=start_s,
            end_s=end_s,
            sample_rate=sample_rate,
            samples=piece,
            channels=1,
            stream_status=stream_status,
            mute_status=mute_status,
            samples_dropped=samples_dropped,
        )
        chunk.validate()
        yield chunk


class FileSource:
    """Lecture d'un fichier WAV (FLAC accepté si le backend SoundFile le permet)."""

    def __init__(
        self,
        path: str | Path,
        session_id: str,
        normalizer: AudioNormalizer | None = None,
        chunk_duration_s: float = 1.0,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> None:
        self.path = Path(path)
        self.session_id = session_id
        self.normalizer = normalizer or AudioNormalizer(target_sample_rate)
        self.chunk_duration_s = chunk_duration_s
        self.target_sample_rate = target_sample_rate
        self.metadata = self._probe()

    def _probe(self) -> SourceMetadata:
        if not self.path.is_file():
            raise InvalidAudioError(f"Fichier introuvable : {self.path}")
        if self.path.stat().st_size == 0:
            raise InvalidAudioError(f"Fichier vide : {self.path}")

        try:
            import soundfile as sf

            info = sf.info(str(self.path))
        except Exception as exc:
            raise InvalidAudioError(f"Fichier audio illisible : {self.path}") from exc

        if info.frames <= 0:
            raise InvalidAudioError(f"Fichier audio sans échantillons : {self.path}")

        original_channels = int(info.channels)
        return SourceMetadata(
            source_type="file",
            original_sample_rate=int(info.samplerate),
            original_channels=original_channels,
            duration_s=float(info.duration),
            file_path=str(self.path),
            downmix="average" if original_channels > 1 else "none",
        )

    def read_chunks(self) -> Iterator[AudioChunk]:
        """Lit le fichier par blocs. Le signal entier ne reste pas en mémoire."""
        import soundfile as sf

        block = max(1, int(round(self.chunk_duration_s * self.metadata.original_sample_rate)))
        clock = SampleClock(self.target_sample_rate)
        try:
            with sf.SoundFile(str(self.path)) as handle:
                source_rate = int(handle.samplerate)
                while True:
                    data = handle.read(block, dtype="float32", always_2d=True)
                    if data.shape[0] == 0:
                        break
                    converted = self.normalizer.convert(data, source_rate)
                    if converted.size == 0:
                        continue
                    start_s, end_s = clock.stamp(len(converted))
                    chunk = AudioChunk(
                        session_id=self.session_id,
                        source_type="file",
                        start_s=start_s,
                        end_s=end_s,
                        sample_rate=self.target_sample_rate,
                        samples=converted,
                        channels=1,
                        stream_status=StreamStatus.ACTIVE,
                        mute_status=MuteStatus.UNKNOWN,
                        samples_dropped=0,
                    )
                    chunk.validate()
                    yield chunk
        except InvalidAudioError:
            raise
        except Exception as exc:
            raise InvalidAudioError(f"Lecture impossible : {self.path}") from exc

    def close(self) -> None:
        return


class LiveReplaySource:
    """Relit un fichier en livrant chaque bloc à l'instant où il se terminerait en direct.

    Cinq minutes d'audio occupent cinq minutes d'horloge. Le fichier peut être
    un extrait court ou une session continue : le rythme ne crée pas la réunion.
    """

    def __init__(
        self,
        path: str | Path,
        session_id: str,
        normalizer: AudioNormalizer | None = None,
        chunk_duration_s: float = 1.0,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self._file = FileSource(
            path,
            session_id,
            normalizer=normalizer,
            chunk_duration_s=chunk_duration_s,
            target_sample_rate=target_sample_rate,
        )
        self.session_id = session_id
        self.metadata = self._file.metadata
        self._clock = clock or time.perf_counter
        self._sleeper = sleeper
        self._stop = threading.Event()
        self.emission_lag_s: deque[float] = deque(maxlen=4096)
        self.origin_s: float | None = None

    def request_stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self._stop.set()
        self._file.close()

    def read_chunks(self) -> Iterator[AudioChunk]:
        origin = self._clock()
        self.origin_s = origin
        for chunk in self._file.read_chunks():
            if self._stop.is_set():
                break
            remaining = (origin + chunk.end_s) - self._clock()
            if remaining > 0 and self._wait(remaining):
                break
            self.emission_lag_s.append(self._clock() - origin - chunk.end_s)
            yield chunk

    def monotonic(self) -> float:
        return self._clock()

    def session_origin_s(self) -> float:
        if self.origin_s is None:
            return self._clock()
        return self.origin_s

    def elapsed(self) -> float:
        if self.origin_s is None:
            return 0.0
        return self._clock() - self.origin_s

    def _wait(self, seconds: float) -> bool:
        if self._sleeper is not None:
            self._sleeper(seconds)
            return self._stop.is_set()
        return self._stop.wait(seconds)


class SyntheticSource:
    """Source déterministe pour les tests automatiques — pas une source superviseur."""

    def __init__(
        self,
        samples: np.ndarray,
        session_id: str,
        source_rate: int = TARGET_SAMPLE_RATE,
        normalizer: AudioNormalizer | None = None,
        chunk_duration_s: float = 1.0,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        mute_status: MuteStatus = MuteStatus.UNKNOWN,
        stream_status: StreamStatus = StreamStatus.ACTIVE,
    ) -> None:
        self.session_id = session_id
        self.normalizer = normalizer or AudioNormalizer(target_sample_rate)
        self.chunk_duration_s = chunk_duration_s
        self.target_sample_rate = target_sample_rate
        self.mute_status = mute_status
        self.stream_status = stream_status
        array = np.asarray(samples)
        original_channels = 1 if array.ndim == 1 else int(array.shape[-1])
        self._converted = self.normalizer.convert(array, source_rate)
        self.metadata = SourceMetadata(
            source_type="synthetic",
            original_sample_rate=source_rate,
            original_channels=original_channels,
            duration_s=len(self._converted) / target_sample_rate,
            downmix="average" if original_channels > 1 else "none",
        )

    def read_chunks(self) -> Iterator[AudioChunk]:
        yield from _chunk_array(
            self._converted,
            session_id=self.session_id,
            source_type="synthetic",
            sample_rate=self.target_sample_rate,
            chunk_duration_s=self.chunk_duration_s,
            stream_status=self.stream_status,
            mute_status=self.mute_status,
        )

    def close(self) -> None:
        return


def default_mute_probe(_device_id: str | None) -> MuteStatus:
    """sounddevice n'expose pas un état mute fiable : mieux vaut UNKNOWN que d'inventer."""

    return MuteStatus.UNKNOWN


class MicrophoneSource:
    """Capture continue via sounddevice.InputStream et une file bornée.

    Le callback se limite à copier les échantillons. Toute normalisation ou
    analyse a lieu hors du thread PortAudio. Une file pleine produit un
    incident de débordement, pas un blocage silencieux.
    """

    def __init__(
        self,
        session_id: str,
        device_id: str | int | None = None,
        normalizer: AudioNormalizer | None = None,
        chunk_duration_s: float = 1.0,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        mute_probe: MuteProbe | None = None,
        queue_maxsize: int = 32,
        stream_factory: Callable[..., object] | None = None,
    ) -> None:
        self.session_id = session_id
        self.device_id = None if device_id in (None, "") else str(device_id)
        self.normalizer = normalizer or AudioNormalizer(target_sample_rate)
        self.chunk_duration_s = chunk_duration_s
        self.target_sample_rate = target_sample_rate
        self.mute_probe = mute_probe or default_mute_probe
        self.queue_maxsize = queue_maxsize
        self._stream_factory = stream_factory
        self._stop = threading.Event()
        self._closed = False
        self.samples_dropped = 0
        self.last_stream_status = StreamStatus.UNKNOWN
        self._native_rate = target_sample_rate
        self._native_channels = 1
        self.device_name: str | None = None
        self._configure_device()
        self.metadata = SourceMetadata(
            source_type="microphone",
            original_sample_rate=self._native_rate,
            original_channels=self._native_channels,
            device_name=self.device_name,
            device_id=self.device_id,
            downmix="average" if self._native_channels > 1 else "none",
        )

    def _configure_device(self) -> None:
        if self._stream_factory is not None:
            self._native_rate = 16_000
            self._native_channels = 1
            self.device_name = "injected-stream"
            return
        try:
            import sounddevice as sd

            parsed = _parse_device(self.device_id)
            info = sd.query_devices(parsed, kind="input")
            self._native_rate = int(info["default_samplerate"])
            self._native_channels = max(1, int(info["max_input_channels"]))
            self.device_name = str(info.get("name") or f"device-{parsed}")
        except Exception as exc:
            raise SourceUnavailableError(
                f"Microphone indisponible (device={self.device_id})."
            ) from exc

    def read_chunks(self) -> Iterator[AudioChunk]:
        raw_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=self.queue_maxsize)
        overflow_flag = threading.Event()
        error_holder: list[BaseException] = []

        def callback(indata, frames, time_info, status) -> None:  # type: ignore[no-untyped-def]
            if self._stop.is_set():
                return
            if status:
                LOGGER.warning("Incident PortAudio : %s", status)
                if getattr(status, "input_overflow", False):
                    overflow_flag.set()
            try:
                raw_queue.put_nowait(np.ascontiguousarray(indata, dtype=np.float32).copy())
            except queue.Full:
                self.samples_dropped += int(frames)
                overflow_flag.set()

        stream = self._open_stream(callback)
        self.last_stream_status = StreamStatus.ACTIVE
        pending = np.zeros((0, self._native_channels), dtype=np.float32)
        native_hop = max(1, int(round(self.chunk_duration_s * self._native_rate)))
        clock = SampleClock(self.target_sample_rate)

        def emit(native: np.ndarray) -> AudioChunk:
            converted = self.normalizer.convert(native, self._native_rate)
            if converted.size == 0:
                raise InvalidAudioError("Le reste de capture normalisé est vide.")
            dropped = self.samples_dropped
            overflow = overflow_flag.is_set()
            if overflow:
                overflow_flag.clear()
            start_s, end_s = clock.stamp(len(converted))
            stream_status = StreamStatus.INTERRUPTED if overflow or dropped else StreamStatus.ACTIVE
            self.last_stream_status = stream_status
            chunk = AudioChunk(
                session_id=self.session_id,
                source_type="microphone",
                start_s=start_s,
                end_s=end_s,
                sample_rate=self.target_sample_rate,
                samples=converted,
                channels=1,
                stream_status=stream_status,
                mute_status=self.mute_probe(self.device_id),
                samples_dropped=dropped,
            )
            chunk.validate()
            self.samples_dropped = 0
            return chunk

        def consume(item: np.ndarray) -> Iterator[AudioChunk]:
            nonlocal pending
            if item.ndim == 1:
                item = item.reshape(-1, 1)
            pending = np.concatenate([pending, item], axis=0)
            while len(pending) >= native_hop:
                native = pending[:native_hop]
                pending = pending[native_hop:]
                yield emit(native)

        try:
            stream.start()
            while not self._stop.is_set():
                try:
                    item = raw_queue.get(timeout=0.25)
                except queue.Empty:
                    if not _stream_active(stream) and not self._stop.is_set():
                        self.last_stream_status = StreamStatus.UNAVAILABLE
                        raise SourceUnavailableError(
                            "Le flux microphone s'est interrompu (périphérique indisponible)."
                        )
                    continue
                if item is None:
                    break
                yield from consume(item)
            self._stop_stream(stream)
            while True:
                try:
                    item = raw_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    continue
                yield from consume(item)
            if len(pending) > 0:
                tail = pending
                pending = np.zeros((0, self._native_channels), dtype=np.float32)
                try:
                    tail_chunk = emit(tail)
                except InvalidAudioError:
                    tail_chunk = None
                if tail_chunk is not None and tail_chunk.samples.size > 0:
                    yield tail_chunk
        except SourceUnavailableError:
            raise
        except Exception as exc:
            error_holder.append(exc)
            self.last_stream_status = StreamStatus.UNAVAILABLE
            raise SourceUnavailableError("Interruption technique du microphone.") from exc
        finally:
            self._stop_stream(stream)
            self.close()

        if error_holder:
            raise SourceUnavailableError("Interruption technique du microphone.") from error_holder[0]

    def request_stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self._stop.set()
        self._closed = True

    def _open_stream(self, callback: Callable[..., None]) -> object:
        if self._stream_factory is not None:
            return self._stream_factory(callback=callback)
        import sounddevice as sd

        return sd.InputStream(
            device=_parse_device(self.device_id),
            channels=self._native_channels,
            samplerate=self._native_rate,
            dtype="float32",
            callback=callback,
        )

    def _stop_stream(self, stream: object) -> None:
        for action in ("stop", "close"):
            method = getattr(stream, action, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    LOGGER.debug("Fermeture de flux : %s a échoué", action, exc_info=True)


def _stream_active(stream: object) -> bool:
    active = getattr(stream, "active", None)
    if active is None:
        return True
    return bool(active)


def _parse_device(device_id: str | int | None) -> int | None:
    if device_id is None or device_id == "":
        return None
    return int(device_id)


def list_input_devices() -> list[dict[str, object]]:
    try:
        import sounddevice as sd
    except Exception as exc:
        LOGGER.warning("sounddevice indisponible : %s", exc)
        return []

    try:
        devices = sd.query_devices()
    except Exception as exc:
        LOGGER.warning("Impossible d'interroger les périphériques audio : %s", exc)
        return []

    result: list[dict[str, object]] = []
    for index, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) <= 0:
            continue
        result.append(
            {
                "id": str(index),
                "name": str(device.get("name", f"device-{index}")),
                "max_input_channels": int(device.get("max_input_channels", 0)),
                "default_samplerate": float(device.get("default_samplerate", 0.0)),
            }
        )
    return result
