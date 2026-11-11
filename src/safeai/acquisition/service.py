"""Service d'acquisition : orchestration locale, non bloquante, sans réseau."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field

from safeai.acquisition.config import AcquisitionConfig, load_config
from safeai.acquisition.contracts import (
    TARGET_SAMPLE_RATE,
    AcquisitionStatus,
    AudioChunk,
    AudioQualityEvent,
    MuteStatus,
    QualityStatus,
    SessionClock,
    SourceConfig,
    SourceMetadata,
    SourceType,
    SpeechActivity,
    SpeechSegment,
    StreamStatus,
    TimelineRow,
    new_session_id,
    utc_now,
)
from safeai.acquisition.errors import ConsentRequiredError, InvalidAudioError, SourceUnavailableError
from safeai.acquisition.normalizer import AudioNormalizer
from safeai.acquisition.publisher import AcquisitionPublisher
from safeai.acquisition.quality import QualityMonitor
from safeai.acquisition.segmenter import AudioSegmenter
from safeai.acquisition.sources import AudioSource, FileSource, MicrophoneSource, SyntheticSource
from safeai.acquisition.vad import VADDetector, build_vad

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    metadata: SourceMetadata
    session: SessionClock
    chunks: list[AudioChunk] = field(default_factory=list)
    speech_segments: list[SpeechSegment] = field(default_factory=list)
    quality_events: list[AudioQualityEvent] = field(default_factory=list)
    timeline: list[TimelineRow] = field(default_factory=list)
    incidents: list[str] = field(default_factory=list)
    processing_latency_s: list[float] = field(default_factory=list)
    vad_processing_s: list[float] = field(default_factory=list)
    capture_to_publish_s: list[float] = field(default_factory=list)
    vad_backend: str = "energy"
    publisher_drops: int = 0
    chunks_published: int = 0
    speech_segments_published: int = 0
    timestamp_errors_total: int = 0

    @property
    def speech_alignment_errors(self) -> int:
        errors = 0
        for segment in self.speech_segments:
            try:
                segment.validate()
            except ValueError:
                errors += 1
        return errors

    @property
    def timestamp_errors(self) -> int:
        errors = 0
        previous_end = 0.0
        for chunk in self.chunks:
            try:
                chunk.validate()
            except ValueError:
                errors += 1
            if self.chunks and chunk is not self.chunks[0] and abs(chunk.start_s - previous_end) > 0.001:
                errors += 1
            previous_end = chunk.end_s
        return errors


class AcquisitionService:
    """API cible V1 : start / stop / subscribe_* / get_status."""

    def __init__(
        self,
        config: AcquisitionConfig | None = None,
        vad: VADDetector | None = None,
        mute_probe=None,
        stream_factory=None,
    ) -> None:
        self.config = config or load_config()
        self._external_vad = vad
        self._mute_probe = mute_probe
        self._stream_factory = stream_factory
        self._normalizer = AudioNormalizer(self.config.target_sample_rate)
        self._quality = QualityMonitor(self.config.quality)
        self._segmenter = AudioSegmenter(self.config.vad, self.config.target_sample_rate)
        self._publisher = AcquisitionPublisher(self.config.queues, self.config.publisher)
        self._vad: VADDetector | None = None
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._source: AudioSource | None = None
        self._stop = threading.Event()
        self._running = False
        self._session: SessionClock | None = None
        self._source_type: SourceType | None = None
        self._metadata: SourceMetadata | None = None
        self._stream_status = StreamStatus.UNKNOWN
        self._mute_status = MuteStatus.UNKNOWN
        self._speech_activity = SpeechActivity.UNKNOWN
        self._quality_status = QualityStatus.UNKNOWN
        self._chunks_published = 0
        self._timestamp_errors = 0
        self._previous_chunk_end: float | None = None
        self._speech_published = 0
        self._quality_published = 0
        self._samples_dropped = 0
        self._incidents: deque[str] = deque(maxlen=128)
        self._collected_chunks: deque[AudioChunk] = deque()
        self._collected_speech: deque[SpeechSegment] = deque()
        self._collected_quality: deque[AudioQualityEvent] = deque()
        self._latencies: deque[float] = deque()
        self._error: BaseException | None = None
        self._collect_maxlen: int | None = None
        self._publisher_drops_seen = 0
        self._speech_present = False
        self._speech_active_at_end = False
        self._vad_times: deque[float] = deque()
        self._publish_times: deque[float] = deque()

    def start(self, source_config: SourceConfig) -> SessionClock:
        with self._lock:
            if self._running:
                raise RuntimeError("Une session d'acquisition est déjà en cours.")
            self._validate_consent(source_config)
            session_id = source_config.session_id or new_session_id()
            session = SessionClock(
                session_id=session_id,
                started_at_utc=utc_now(),
                monotonic_origin=time.monotonic(),
            )
            bound = None if source_config.source_type in {"file", "synthetic"} else 180
            self._reset_runtime(session, source_config.source_type, collect_maxlen=bound)
            self._vad = self._external_vad or build_vad(self.config.vad)
            self._vad.reset()
            self._quality.reset()
            self._segmenter.reset()
            self._publisher.reset()
            self._source = self._build_source(source_config, session_id)
            self._metadata = self._source.metadata
            self._stop.clear()
            self._running = True
            self._stream_status = StreamStatus.ACTIVE
            self._worker = threading.Thread(
                target=self._run_loop,
                name=f"safeai-acq-{session_id}",
                daemon=True,
            )
            self._worker.start()
            return session

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            source = self._source
            worker = self._worker
        if source is not None and hasattr(source, "request_stop"):
            source.request_stop()
        if source is not None:
            source.close()
        if worker is not None and worker.is_alive() and threading.current_thread() is not worker:
            worker.join(timeout=5.0)
        with self._lock:
            self._running = False
            self._worker = None
            self._source = None

    def subscribe_audio(self) -> queue.Queue[AudioChunk]:
        return self._publisher.subscribe_audio()

    def subscribe_speech_segments(self) -> queue.Queue[SpeechSegment]:
        return self._publisher.subscribe_speech_segments()

    def subscribe_quality_events(self) -> queue.Queue[AudioQualityEvent]:
        return self._publisher.subscribe_quality_events()

    def get_status(self) -> AcquisitionStatus:
        with self._lock:
            elapsed = 0.0
            if self._session is not None:
                elapsed = max(0.0, time.monotonic() - self._session.monotonic_origin)
            return AcquisitionStatus(
                running=self._running,
                session_id=self._session.session_id if self._session else None,
                source_type=self._source_type,
                stream_status=self._stream_status,
                mute_status=self._mute_status,
                last_speech_activity=self._speech_activity,
                last_quality_status=self._quality_status,
                chunks_published=self._chunks_published,
                speech_segments_published=self._speech_published,
                quality_events_published=self._quality_published,
                samples_dropped=self._samples_dropped,
                started_at_utc=self._session.started_at_utc if self._session else None,
                elapsed_s=elapsed,
                target_sample_rate=self.config.target_sample_rate,
                backend_vad=self._vad.name if self._vad is not None else self.config.vad.backend,
                publisher_drops=self._publisher.dropped_total,
                speech_present_in_window=self._speech_present,
                speech_active_at_end=self._speech_active_at_end,
                incidents=list(self._incidents),
            )

    def wait_until_idle(self, timeout_s: float = 30.0) -> None:
        worker = self._worker
        if worker is None:
            return
        worker.join(timeout=timeout_s)
        if self._error is not None:
            raise self._error

    def collect_result(self) -> PipelineResult:
        if self._session is None or self._metadata is None:
            raise RuntimeError("Aucune session n'a été démarrée.")
        timeline = build_timeline(self._collected_quality, self._collected_speech)
        return PipelineResult(
            metadata=self._metadata,
            session=self._session,
            chunks=list(self._collected_chunks),
            speech_segments=list(self._collected_speech),
            quality_events=list(self._collected_quality),
            timeline=timeline,
            incidents=list(self._incidents),
            processing_latency_s=list(self._latencies),
            vad_processing_s=list(self._vad_times),
            capture_to_publish_s=list(self._publish_times),
            vad_backend=self._vad.name if self._vad is not None else self.config.vad.backend,
            publisher_drops=self._publisher.dropped_total,
            chunks_published=self._chunks_published,
            speech_segments_published=self._speech_published,
            timestamp_errors_total=self._timestamp_errors,
        )

    def run_source(self, source: AudioSource, *, retain_history: bool = False) -> PipelineResult:
        """Exécute une source déjà construite, y compris un rejeu au rythme réel.

        L'historique audio conservé en mémoire est borné par la file d'acquisition.
        ``retain_history`` garde toute la session pour une mesure, pas pour le direct.
        Les abonnements pris avant cet appel restent en place.
        """
        with self._lock:
            if self._running:
                raise RuntimeError("Une session d'acquisition est déjà en cours.")
            self._stop.clear()
            session_id = getattr(source, "session_id", None) or new_session_id()
            session = SessionClock(
                session_id=session_id,
                started_at_utc=utc_now(),
                monotonic_origin=time.monotonic(),
            )
            bound = None if retain_history else self.config.queues.audio_maxsize
            self._reset_runtime(session, source.metadata.source_type, collect_maxlen=bound)
            self._vad = self._external_vad or build_vad(self.config.vad)
            self._vad.reset()
            self._quality.reset()
            self._segmenter.reset()
            self._publisher.reset()
            self._source = source
            self._metadata = source.metadata
            self._running = True
            self._stream_status = StreamStatus.ACTIVE
        try:
            self._run_loop()
            if self._error is not None:
                raise self._error
            return self.collect_result()
        finally:
            self.stop()

    def run_to_completion(self, source_config: SourceConfig) -> PipelineResult:
        self.start(source_config)
        try:
            duration_hint = None
            if self._metadata and self._metadata.duration_s:
                duration_hint = self._metadata.duration_s + 5.0
            self.wait_until_idle(timeout_s=duration_hint or 60.0)
            return self.collect_result()
        finally:
            self.stop()

    def _run_loop(self) -> None:
        assert self._source is not None
        last_end = 0.0
        try:
            for chunk in self._source.read_chunks():
                started = time.perf_counter()
                self._process_chunk(chunk)
                self._latencies.append(time.perf_counter() - started)
                last_end = chunk.end_s
            leftover = self._segmenter.flush(last_end)
            for segment in leftover:
                self._publish_speech(segment)
        except (InvalidAudioError, SourceUnavailableError) as exc:
            LOGGER.error("Incident d'acquisition : %s", exc)
            self._register_incident(str(exc), StreamStatus.UNAVAILABLE)
            self._error = exc
        except Exception as exc:
            LOGGER.exception("Erreur non contrôlée d'acquisition")
            self._register_incident(f"erreur_interne: {exc}", StreamStatus.UNAVAILABLE)
            self._error = exc
        finally:
            with self._lock:
                self._running = False

    def _process_chunk(self, chunk: AudioChunk) -> None:
        assert self._vad is not None
        received = time.perf_counter()
        frames = self._vad.process(chunk)
        vad_dt = time.perf_counter() - received
        verdict = self._segmenter.ingest(chunk, frames)
        activity, _mean_p, segments = verdict
        self._publish_timeline_gaps(chunk)
        event = self._quality.observe(
            chunk,
            activity,
            speech_active_at_end=verdict.speech_active_at_end,
        )
        self._publisher.publish_audio(chunk)
        self._publisher.publish_quality(event)
        publish_dt = time.perf_counter() - received
        with self._lock:
            self._count_timestamp(chunk)
            self._collected_chunks.append(chunk)
            self._collected_quality.append(event)
            self._chunks_published += 1
            self._quality_published += 1
            self._stream_status = chunk.stream_status
            self._mute_status = chunk.mute_status
            self._speech_activity = activity
            self._speech_present = verdict.speech_present_in_window
            self._speech_active_at_end = verdict.speech_active_at_end
            self._quality_status = event.quality_status
            self._samples_dropped += chunk.samples_dropped
            self._vad_times.append(vad_dt)
            self._publish_times.append(publish_dt)
            if chunk.samples_dropped:
                self._incidents.append(
                    f"{chunk.start_s:.3f}-{chunk.end_s:.3f}: perte de {chunk.samples_dropped} échantillons"
                )
            self._note_publisher_drops()
        for segment in segments:
            self._publish_speech(segment)

    def _count_timestamp(self, chunk: AudioChunk) -> None:
        """Compte les anomalies sur chaque bloc, même s'il quitte ensuite l'historique borné."""
        try:
            chunk.validate()
        except ValueError:
            self._timestamp_errors += 1
        if self._previous_chunk_end is not None and abs(chunk.start_s - self._previous_chunk_end) > 0.001:
            self._timestamp_errors += 1
        self._previous_chunk_end = chunk.end_s

    def _publish_timeline_gaps(self, chunk: AudioChunk) -> None:
        """Un trou entre deux blocs est une perte, pas une seconde de silence."""
        for gap in self._segmenter.take_gaps():
            event = AudioQualityEvent(
                session_id=chunk.session_id,
                start_s=gap.start_s,
                end_s=gap.end_s,
                stream_status=StreamStatus.INTERRUPTED,
                mute_status=chunk.mute_status,
                speech_activity=SpeechActivity.UNKNOWN,
                quality_status=QualityStatus.DEGRADED,
                rms=0.0,
                peak=0.0,
                clipping_ratio=0.0,
                noise_estimate=0.0,
                silence_duration_s=self._quality.silence_duration_s,
                buffer_overflow=True,
                reason="timeline_gap",
                samples_dropped=gap.samples_missing,
            )
            self._publisher.publish_quality(event)
            self._quality.break_silence_run()
            with self._lock:
                self._collected_quality.append(event)
                self._quality_published += 1
                self._samples_dropped += gap.samples_missing
                self._stream_status = StreamStatus.INTERRUPTED
                self._quality_status = QualityStatus.DEGRADED
                self._incidents.append(
                    f"{gap.start_s:.3f}-{gap.end_s:.3f}: perte de {gap.samples_missing} "
                    "échantillons (trou de flux, pas un silence)"
                )

    def _publish_speech(self, segment: SpeechSegment) -> None:
        self._publisher.publish_speech(segment)
        with self._lock:
            self._collected_speech.append(segment)
            self._speech_published += 1
            self._note_publisher_drops()

    def _register_incident(self, message: str, status: StreamStatus) -> None:
        with self._lock:
            self._incidents.append(message)
            self._stream_status = status
            self._quality_status = QualityStatus.UNUSABLE
            if self._session is None:
                return
            start_s = 0.0
            if self._collected_chunks:
                start_s = self._collected_chunks[-1].end_s
            event = AudioQualityEvent(
                session_id=self._session.session_id,
                start_s=start_s,
                end_s=start_s,
                stream_status=status,
                mute_status=self._mute_status,
                speech_activity=SpeechActivity.UNKNOWN,
                quality_status=QualityStatus.UNUSABLE,
                rms=0.0,
                peak=0.0,
                clipping_ratio=0.0,
                noise_estimate=0.0,
                silence_duration_s=0.0,
                buffer_overflow=False,
                reason="technical_incident",
            )
            self._collected_quality.append(event)
            self._quality_published += 1
        self._publisher.publish_quality(event)

    def _build_source(self, source_config: SourceConfig, session_id: str) -> AudioSource:
        chunk_s = source_config.chunk_duration_s or self.config.chunk_duration_s
        if source_config.source_type == "file":
            if source_config.file_path is None:
                raise InvalidAudioError("file_path est requis pour une source fichier.")
            return FileSource(
                path=source_config.file_path,
                session_id=session_id,
                normalizer=self._normalizer,
                chunk_duration_s=chunk_s,
                target_sample_rate=self.config.target_sample_rate,
            )
        if source_config.source_type == "synthetic":
            raise InvalidAudioError("La source synthétique se construit via SyntheticSource, pas via SourceConfig fichier.")
        return MicrophoneSource(
            session_id=session_id,
            device_id=source_config.device_id,
            normalizer=self._normalizer,
            chunk_duration_s=chunk_s,
            target_sample_rate=self.config.target_sample_rate,
            mute_probe=self._mute_probe,
            queue_maxsize=self.config.queues.audio_maxsize,
            stream_factory=self._stream_factory,
        )

    def _validate_consent(self, source_config: SourceConfig) -> None:
        if source_config.source_type == "microphone" and not source_config.recording_consent:
            raise ConsentRequiredError(
                "Le consentement d'enregistrement est obligatoire avant une capture microphone. "
                "Ce drapeau interne ne remplace pas la preuve d'accord écrit prévue par le guide."
            )

    def _reset_runtime(
        self,
        session: SessionClock,
        source_type: SourceType,
        collect_maxlen: int | None = None,
    ) -> None:
        self._session = session
        self._source_type = source_type
        self._metadata = None
        self._stream_status = StreamStatus.UNKNOWN
        self._mute_status = MuteStatus.UNKNOWN
        self._speech_activity = SpeechActivity.UNKNOWN
        self._quality_status = QualityStatus.UNKNOWN
        self._chunks_published = 0
        self._timestamp_errors = 0
        self._previous_chunk_end = None
        self._speech_published = 0
        self._quality_published = 0
        self._samples_dropped = 0
        self._incidents = deque(maxlen=128)
        self._collect_maxlen = collect_maxlen
        self._collected_chunks = deque(maxlen=collect_maxlen)
        self._collected_speech = deque(maxlen=collect_maxlen)
        self._collected_quality = deque(maxlen=collect_maxlen)
        self._latencies = deque(maxlen=collect_maxlen)
        self._vad_times = deque(maxlen=collect_maxlen)
        self._publish_times = deque(maxlen=collect_maxlen)
        self._publisher_drops_seen = 0
        self._speech_present = False
        self._speech_active_at_end = False
        self._error = None

    def _note_publisher_drops(self) -> None:
        current = self._publisher.dropped_total
        if current <= self._publisher_drops_seen:
            return
        delta = current - self._publisher_drops_seen
        self._publisher_drops_seen = current
        self._incidents.append(
            "files d'abonnes: "
            f"{delta} evenement(s) abandonne(s) "
            f"(audio={self._publisher.audio.dropped}, "
            f"speech={self._publisher.speech.dropped}, "
            f"quality={self._publisher.quality.dropped})"
        )


def run_file(
    path,
    config: AcquisitionConfig | None = None,
    vad: VADDetector | None = None,
    recording_consent: bool = True,
    session_id: str | None = None,
) -> PipelineResult:
    service = AcquisitionService(config=config, vad=vad)
    return service.run_to_completion(
        SourceConfig(
            source_type="file",
            file_path=path,
            recording_consent=recording_consent,
            chunk_duration_s=(config.chunk_duration_s if config else 1.0),
            session_id=session_id,
        )
    )


def run_synthetic(
    source: SyntheticSource,
    config: AcquisitionConfig | None = None,
    vad: VADDetector | None = None,
) -> PipelineResult:
    service = AcquisitionService(config=config, vad=vad)
    return service.run_source(source)


def iter_chunks(source: AudioSource) -> Iterator[AudioChunk]:
    try:
        yield from source.read_chunks()
    finally:
        source.close()


def build_timeline(
    quality_events: list[AudioQualityEvent],
    speech_segments: list[SpeechSegment],
) -> list[TimelineRow]:
    if not quality_events and not speech_segments:
        return []
    end_s = 0.0
    if quality_events:
        end_s = max(end_s, quality_events[-1].end_s)
    if speech_segments:
        end_s = max(end_s, speech_segments[-1].end_s)

    rows: list[TimelineRow] = []
    cursor = 0.0
    for segment in sorted(speech_segments, key=lambda item: item.start_s):
        if segment.start_s > cursor + 0.01:
            rows.append(
                _row_from_quality(cursor, segment.start_s, "silence", quality_events)
            )
        rows.append(_row_from_quality(segment.start_s, segment.end_s, "speech", quality_events))
        cursor = max(cursor, segment.end_s)
    if end_s > cursor + 0.01:
        rows.append(_row_from_quality(cursor, end_s, "silence", quality_events))
    if not rows and quality_events:
        for event in quality_events:
            kind = "speech" if event.speech_activity is SpeechActivity.SPEECH else "silence"
            rows.append(
                TimelineRow(
                    start_s=event.start_s,
                    end_s=event.end_s,
                    kind=kind,
                    quality_status=event.quality_status,
                    reason=event.reason,
                )
            )
    return rows


def _row_from_quality(
    start_s: float,
    end_s: float,
    kind: str,
    events: list[AudioQualityEvent],
) -> TimelineRow:
    overlapping = [event for event in events if event.end_s > start_s and event.start_s < end_s]
    if not overlapping:
        return TimelineRow(
            start_s=start_s,
            end_s=end_s,
            kind=kind,  # type: ignore[arg-type]
            quality_status=QualityStatus.UNKNOWN,
            reason="insufficient_information",
        )
    rank = {QualityStatus.UNUSABLE: 3, QualityStatus.DEGRADED: 2, QualityStatus.OK: 1, QualityStatus.UNKNOWN: 0}
    worst = max(overlapping, key=lambda event: rank[event.quality_status])
    return TimelineRow(
        start_s=start_s,
        end_s=end_s,
        kind=kind,  # type: ignore[arg-type]
        quality_status=worst.quality_status,
        reason=worst.reason,
    )


def downsample_envelope(chunks: list[AudioChunk], max_points: int = 1600) -> dict[str, list[float]]:
    if not chunks:
        return {"times": [], "min": [], "max": []}
    samples = []
    for chunk in chunks:
        samples.append(chunk.samples)
    import numpy as np

    audio = np.concatenate(samples)
    sample_rate = chunks[0].sample_rate or TARGET_SAMPLE_RATE
    hop = max(1, len(audio) // max_points)
    times: list[float] = []
    mins: list[float] = []
    maxs: list[float] = []
    for index in range(0, len(audio), hop):
        window = audio[index : index + hop]
        times.append(index / sample_rate)
        mins.append(float(window.min()))
        maxs.append(float(window.max()))
    return {"times": times, "min": mins, "max": maxs}
