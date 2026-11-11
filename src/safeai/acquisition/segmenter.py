"""Segmentation : flux continu préservé + intervalles de parole avec marges."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from safeai.acquisition.config import VADSettings
from safeai.acquisition.contracts import (
    TARGET_SAMPLE_RATE,
    AudioChunk,
    SpeechActivity,
    SpeechSegment,
    VADFrame,
)


@dataclass
class _OpenSpeech:
    session_id: str
    start_s: float
    first_speech_s: float
    last_speech_s: float
    probabilities: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class AudioGap:
    """Trou dans la ligne de temps. Ce n'est pas du silence capté."""

    start_s: float
    end_s: float
    samples_missing: int


@dataclass
class SegmenterResult:
    """Résultat d'une fenêtre. Se dépaquette en (activity, mean_p, segments).

    `activity` reflète speech_present_in_window, pas speech_active_at_end :
    une parole déjà terminée dans la fenêtre reste de la parole.
    """

    activity: SpeechActivity
    mean_probability: float
    segments: list[SpeechSegment]
    speech_present_in_window: bool
    speech_active_at_end: bool

    def __iter__(self):
        yield self.activity
        yield self.mean_probability
        yield self.segments


class _TimedAudioBuffer:
    """Tampon horodaté par morceaux réels.

    Un trou entre deux blocs est une perte : il n'est pas comblé par des zéros.
    Un zéro inséré serait relu plus tard comme un silence capté.
    """

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.pieces: list[tuple[float, np.ndarray]] = []
        self._gaps: list[AudioGap] = []

    def reset(self) -> None:
        self.pieces.clear()
        self._gaps.clear()

    @property
    def origin_s(self) -> float | None:
        if not self.pieces:
            return None
        return self.pieces[0][0]

    @property
    def end_s(self) -> float:
        if not self.pieces:
            return 0.0
        start, samples = self.pieces[-1]
        return start + len(samples) / self.sample_rate

    def append(self, start_s: float, samples: np.ndarray) -> AudioGap | None:
        piece = np.ascontiguousarray(samples, dtype=np.float32)
        if piece.size == 0:
            return None
        if not self.pieces:
            self.pieces.append((start_s, piece.copy()))
            return None
        last_start, last = self.pieces[-1]
        expected = last_start + len(last) / self.sample_rate
        gap_n = int(round((start_s - expected) * self.sample_rate))
        if gap_n > 0:
            gap = AudioGap(
                start_s=expected,
                end_s=expected + gap_n / self.sample_rate,
                samples_missing=gap_n,
            )
            self._gaps.append(gap)
            self.pieces.append((start_s, piece.copy()))
            return gap
        if gap_n < 0:
            skip = min(-gap_n, len(piece))
            piece = piece[skip:]
            if piece.size == 0:
                return None
            start_s = start_s + skip / self.sample_rate
        self.pieces[-1] = (last_start, np.concatenate([last, piece]))
        return None

    def drain_gaps(self) -> list[AudioGap]:
        found = self._gaps
        self._gaps = []
        return found

    def slice(self, start_s: float, end_s: float) -> tuple[np.ndarray, float, float]:
        """Extrait l'audio réel. S'arrête au premier trou, sans le combler."""
        if end_s <= start_s or not self.pieces:
            return np.zeros(0, dtype=np.float32), start_s, start_s
        parts: list[np.ndarray] = []
        actual_start: float | None = None
        cursor = start_s
        for piece_start, piece in self.pieces:
            piece_end = piece_start + len(piece) / self.sample_rate
            if piece_end <= start_s or piece_start >= end_s:
                continue
            if actual_start is not None and piece_start > cursor + (0.5 / self.sample_rate):
                break
            take_from = max(start_s, piece_start)
            take_to = min(end_s, piece_end)
            a = int(round((take_from - piece_start) * self.sample_rate))
            b = int(round((take_to - piece_start) * self.sample_rate))
            a = min(max(a, 0), len(piece))
            b = min(max(b, a), len(piece))
            if b <= a:
                continue
            if actual_start is None:
                actual_start = piece_start + a / self.sample_rate
            parts.append(piece[a:b])
            cursor = piece_start + b / self.sample_rate
        if not parts or actual_start is None:
            return np.zeros(0, dtype=np.float32), start_s, start_s
        sliced = np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)
        actual_end = actual_start + len(sliced) / self.sample_rate
        return sliced, actual_start, actual_end

    def trim_before(self, t_s: float) -> None:
        kept: list[tuple[float, np.ndarray]] = []
        for start, samples in self.pieces:
            end = start + len(samples) / self.sample_rate
            if end <= t_s:
                continue
            if start >= t_s:
                kept.append((start, samples))
                continue
            index = int(round((t_s - start) * self.sample_rate))
            index = min(max(index, 0), len(samples))
            if index >= len(samples):
                continue
            kept.append(
                (
                    start + index / self.sample_rate,
                    np.ascontiguousarray(samples[index:], dtype=np.float32),
                )
            )
        self.pieces = [(start, samples) for start, samples in kept if samples.size]


class AudioSegmenter:
    """Construit des SpeechSegment à partir des frames VAD.

    Les échantillons vivent dans un tampon horodaté. Un intervalle n'est
    extrait qu'une fois ses bornes connues, y compris s'il s'ouvre et se
    ferme dans le même bloc. Les silences restent dans le flux AudioChunk.
    """

    def __init__(self, settings: VADSettings | None = None, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
        self.settings = settings or VADSettings()
        self.sample_rate = sample_rate
        self._open: _OpenSpeech | None = None
        self._buffer = _TimedAudioBuffer(sample_rate)
        self._max_preroll_s = self.settings.speech_pad_s + 0.05

    def reset(self) -> None:
        self._open = None
        self._buffer.reset()

    def take_gaps(self) -> list[AudioGap]:
        """Pertes de flux depuis le dernier appel. Jamais comptées comme silence."""
        return self._buffer.drain_gaps()

    def ingest(self, chunk: AudioChunk, frames: list[VADFrame]) -> SegmenterResult:
        gap = self._buffer.append(chunk.start_s, chunk.samples)
        closed: list[SpeechSegment] = []
        if gap is not None and self._open is not None:
            segment = self._close(min(self._open.last_speech_s, gap.start_s))
            if segment is not None:
                closed.append(segment)
        speech_frames = 0
        prob_sum = 0.0
        closed_here: list[SpeechSegment] = []

        for frame in frames:
            prob_sum += frame.probability
            is_speech = frame.probability >= self.settings.threshold
            if is_speech:
                speech_frames += 1
            if self._open is None:
                if is_speech:
                    self._open_speech(frame, chunk.session_id)
            elif is_speech:
                self._open.last_speech_s = frame.end_s
                self._open.probabilities.append(frame.probability)
                segment = self._split_if_too_long(chunk.session_id)
                if segment is not None:
                    closed_here.append(segment)
            elif (frame.end_s - self._open.last_speech_s) >= self.settings.min_silence_duration_s:
                segment = self._close(self._open.last_speech_s)
                if segment is not None:
                    closed_here.append(segment)

        closed.extend(closed_here)
        self._trim_buffer(chunk.end_s)

        frame_s = self.settings.window_samples / self.sample_rate
        reliable_span = speech_frames * frame_s
        speech_present = (
            bool(closed_here)
            or self._open is not None
            or reliable_span >= self.settings.min_speech_duration_s
        )
        speech_active_at_end = self._open is not None
        mean_p = prob_sum / len(frames) if frames else 0.0
        activity = SpeechActivity.SPEECH if speech_present else SpeechActivity.NON_SPEECH
        return SegmenterResult(
            activity=activity,
            mean_probability=mean_p,
            segments=closed,
            speech_present_in_window=speech_present,
            speech_active_at_end=speech_active_at_end,
        )

    def flush(self, end_s: float) -> list[SpeechSegment]:
        if self._open is None:
            return []
        segment = self._close(min(self._open.last_speech_s, end_s))
        return [segment] if segment is not None else []

    def _open_speech(self, frame: VADFrame, session_id: str) -> None:
        start_s = max(0.0, frame.start_s - self.settings.speech_pad_s)
        self._open = _OpenSpeech(
            session_id=session_id,
            start_s=start_s,
            first_speech_s=frame.start_s,
            last_speech_s=frame.end_s,
            probabilities=[frame.probability],
        )

    def _close(self, end_limit_s: float) -> SpeechSegment | None:
        opened = self._open
        self._open = None
        if opened is None:
            return None
        if end_limit_s - opened.first_speech_s < self.settings.min_speech_duration_s:
            return None

        requested_end = end_limit_s + self.settings.speech_pad_s
        sliced, actual_start, actual_end = self._buffer.slice(opened.start_s, requested_end)
        if sliced.size == 0:
            return None

        mean_p = float(np.clip(np.mean(opened.probabilities), 0.0, 1.0))
        segment = SpeechSegment(
            session_id=opened.session_id,
            start_s=actual_start,
            end_s=actual_end,
            speech_detected=True,
            vad_mean_probability=mean_p,
            audio_available=True,
            samples=sliced,
            sample_rate=self.sample_rate,
        )
        segment.validate()
        return segment

    def _split_if_too_long(self, session_id: str) -> SpeechSegment | None:
        """Ferme un segment trop long et rouvre la suite, avec un recouvrement.

        Les échantillons émis quittent le tampon. Le recouvrement reste
        disponible pour le segment suivant, avec les mêmes horodatages.
        """
        opened = self._open
        if opened is None:
            return None
        limit = self.settings.max_speech_segment_s
        if limit <= 0 or opened.last_speech_s - opened.start_s < limit:
            return None
        overlap = min(max(0.0, self.settings.segment_overlap_s), limit / 2)
        cut = opened.start_s + limit
        sliced, actual_start, actual_end = self._buffer.slice(opened.start_s, cut)
        if sliced.size == 0 or actual_end - actual_start < self.settings.min_speech_duration_s:
            return None
        mean_p = float(np.clip(np.mean(opened.probabilities), 0.0, 1.0)) if opened.probabilities else 0.0
        segment = SpeechSegment(
            session_id=opened.session_id,
            start_s=actual_start,
            end_s=actual_end,
            speech_detected=True,
            vad_mean_probability=mean_p,
            audio_available=True,
            samples=sliced,
            sample_rate=self.sample_rate,
        )
        segment.validate()
        restart = max(actual_start, actual_end - overlap)
        self._open = _OpenSpeech(
            session_id=session_id,
            start_s=restart,
            first_speech_s=restart,
            last_speech_s=opened.last_speech_s,
            probabilities=opened.probabilities[-1:],
        )
        return segment

    def _trim_buffer(self, now_s: float) -> None:
        keep_from = now_s - self._max_preroll_s
        if self._open is not None:
            keep_from = min(keep_from, self._open.start_s)
        self._buffer.trim_before(max(0.0, keep_from))
