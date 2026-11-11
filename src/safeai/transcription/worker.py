"""Worker de transcription, hors du thread d'acquisition.

Il consomme les SpeechSegment déjà publiés. Si Whisper est lent, la capture
continue. Une file pleine est comptée, pas ignorée en silence.

Le texte déjà publié est conservé dans une fenêtre bornée. Il n'est pas
renvoyé au modèle : les références d'écoute ne sont pas encore corrigées,
et un prompt de contexte changerait la stratégie mesurée.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from safeai.acquisition.contracts import SpeechSegment
from safeai.acquisition.publisher import BoundedFanout
from safeai.transcription.contracts import TranscriptSegment
from safeai.transcription.windows import _join_word_text, _same_acoustic_interval

LOGGER = logging.getLogger(__name__)

Clock = Callable[[], float]


@dataclass(frozen=True)
class PublishedTranscript:
    """Texte publié et les deux temps qui définissent sa latence.

    ``ready_s`` et ``session_origin_s`` sont sur la même horloge monotone.
    La fin audio est un décalage dans la session, pas une valeur de
    ``time.perf_counter`` brute.
    """

    segment: TranscriptSegment
    audio_end_s: float
    ready_s: float
    session_origin_s: float
    compute_s: float

    @property
    def lag_s(self) -> float:
        return self.ready_s - (self.session_origin_s + self.audio_end_s)


class WhisperWorker:
    def __init__(
        self,
        transcriber,
        speech_queue: queue.Queue[SpeechSegment],
        *,
        clock: Clock | None = None,
        session_origin_s: float | Callable[[], float] | None = None,
        context_size: int = 32,
        transcript_queue_size: int = 64,
    ) -> None:
        self._transcriber = transcriber
        self._queue = speech_queue
        if clock is None and session_origin_s is None:
            monotonic = time.perf_counter
            origin = monotonic()
            self._publication_clock = monotonic
            self._session_origin: Callable[[], float] = lambda: origin
        elif session_origin_s is None:
            assert clock is not None
            self._publication_clock = clock
            self._session_origin = lambda: 0.0
        else:
            self._publication_clock = clock or time.perf_counter
            if callable(session_origin_s):
                self._session_origin = session_origin_s
            else:
                captured = float(session_origin_s)
                self._session_origin = lambda: captured
        self._stop = threading.Event()
        self._busy = threading.Event()
        self._thread: threading.Thread | None = None
        self._context: deque[str] = deque(maxlen=context_size)
        self.published: deque[PublishedTranscript] = deque(maxlen=context_size)
        self.lag_s: deque[float] = deque(maxlen=256)
        self._transcripts: BoundedFanout[PublishedTranscript] = BoundedFanout(transcript_queue_size)
        self._taken_intervals: deque[tuple[float, float]] = deque(maxlen=512)
        self.received_count = 0
        self.errors = 0

    @property
    def context(self) -> tuple[str, ...]:
        return tuple(self._context)

    def subscribe_transcripts(self) -> queue.Queue[PublishedTranscript]:
        return self._transcripts.subscribe()

    @property
    def transcript_drops(self) -> int:
        return self._transcripts.dropped

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="safeai-whisper", daemon=True)
        self._thread.start()

    def finish(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._queue.empty() and not self._busy.is_set():
                time.sleep(0.02)
                if self._queue.empty() and not self._busy.is_set():
                    break
            else:
                time.sleep(0.01)
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=1.0)

    def _run(self) -> None:
        while not (self._stop.is_set() and self._queue.empty()):
            try:
                segment = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            self._busy.set()
            try:
                self._transcribe(segment)
            finally:
                self._busy.clear()

    def _transcribe(self, segment: SpeechSegment) -> None:
        source_id = f"speech_{self.received_count + 1:03d}"
        started = self._publication_clock()
        try:
            produced = self._transcriber.transcribe_speech(segment, source_id)
        except Exception:
            LOGGER.exception("Transcription en échec sur %s", source_id)
            self.errors += 1
            produced = [self._failed_segment(segment, source_id)]
        compute_s = self._publication_clock() - started
        ready_s = self._publication_clock()
        origin_s = float(self._session_origin())
        if not produced:
            produced = [self._empty_segment(segment, source_id)]
        for piece in produced:
            accepted = self._accept_piece(piece, segment, source_id)
            if accepted is None:
                continue
            if accepted.status == "completed":
                accepted = self._consolidate(accepted)
            if accepted is None:
                continue
            item = PublishedTranscript(
                segment=accepted,
                audio_end_s=segment.end_s,
                ready_s=ready_s,
                session_origin_s=origin_s,
                compute_s=compute_s,
            )
            self.published.append(item)
            self.lag_s.append(item.lag_s)
            self._transcripts.publish(item)
            if accepted.status == "completed" and accepted.text.strip():
                self._context.append(accepted.text.strip())
        self.received_count += 1

    def _accept_piece(
        self,
        piece: TranscriptSegment,
        segment: SpeechSegment,
        source_id: str,
    ) -> TranscriptSegment | None:
        try:
            piece.validate()
            return piece
        except Exception:
            LOGGER.exception("Sortie de transcription invalide sur %s", source_id)
            self.errors += 1
        failed = self._failed_segment(segment, source_id)
        try:
            failed.validate()
        except Exception:
            LOGGER.exception("Échec de transcription non publiable sur %s", source_id)
            return None
        return failed

    def _consolidate(self, piece: TranscriptSegment) -> TranscriptSegment | None:
        """Un intervalle acoustique déjà publié ne ressort pas une seconde fois.

        Le recouvrement audio des segments longs reste en place. Seul le texte
        est consolidé : un mot dont le temps est déjà couvert est omis, un mot
        nouveau après la frontière est gardé entier.
        """
        if not piece.words:
            if self._interval_taken(piece.start_s, piece.end_s):
                return None
            self._taken_intervals.append((piece.start_s, piece.end_s))
            return piece
        kept = [word for word in piece.words if not self._interval_taken(word.start_s, word.end_s)]
        if not kept:
            return None
        for word in kept:
            self._taken_intervals.append((word.start_s, word.end_s))
        rebuilt = TranscriptSegment(
            session_id=piece.session_id,
            source_segment_id=piece.source_segment_id,
            start_s=kept[0].start_s,
            end_s=kept[-1].end_s,
            text=_join_word_text(kept),
            language=piece.language,
            status="completed",
            speaker_id=None,
            words=tuple(kept),
        )
        rebuilt.validate()
        return rebuilt

    def _interval_taken(self, start_s: float, end_s: float) -> bool:
        return any(
            _same_acoustic_interval(taken_start, taken_end, start_s, end_s)
            for taken_start, taken_end in self._taken_intervals
        )

    def _failed_segment(self, segment: SpeechSegment, source_id: str) -> TranscriptSegment:
        return TranscriptSegment(
            session_id=segment.session_id,
            source_segment_id=source_id,
            start_s=segment.start_s,
            end_s=segment.end_s,
            text="",
            language="fr",
            status="failed",
        )

    def _empty_segment(self, segment: SpeechSegment, source_id: str) -> TranscriptSegment:
        return TranscriptSegment(
            session_id=segment.session_id,
            source_segment_id=source_id,
            start_s=segment.start_s,
            end_s=segment.end_s,
            text="",
            language="fr",
            status="empty",
        )
