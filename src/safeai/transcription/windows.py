"""Fenêtres de transcription : contexte glissant ou pauses, puis déduplication.

Les durées 8–12 s sont une hypothèse. Les régions commises se suivent
sans trou et sans double publication.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from safeai.acquisition.contracts import TARGET_SAMPLE_RATE
from safeai.acquisition.normalizer import AudioNormalizer
from safeai.transcription.contracts import TranscriptSegment, WordTiming
from safeai.transcription.whisper import WhisperTranscriber

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioWindow:
    start_s: float
    end_s: float
    commit_start_s: float
    commit_end_s: float


def sliding_windows(duration_s: float, window_s: float, overlap_s: float) -> list[AudioWindow]:
    if duration_s <= 0:
        return []
    if window_s <= 0 or overlap_s < 0 or overlap_s >= window_s:
        raise ValueError("Fenêtre invalide : 0 <= chevauchement < durée de fenêtre.")
    windows: list[AudioWindow] = []
    start = 0.0
    while start < duration_s - 1e-6:
        end = min(duration_s, start + window_s)
        last = end >= duration_s - 1e-6
        commit_end = end if last else end - overlap_s
        if commit_end <= start:
            commit_end = end
        windows.append(AudioWindow(start, end, start, commit_end))
        if last:
            break
        start = end - overlap_s
    return windows


def pause_windows(
    regions: list[tuple[float, float]],
    *,
    max_window_s: float,
    max_pause_s: float,
) -> list[AudioWindow]:
    """Regroupe la parole tant que la pause interne reste courte.

    Un intervalle unique plus long que ``max_window_s`` est coupé en
    morceaux adjacents. Le plafond est une contrainte, pas une indication.
    """
    if max_window_s <= 0 or max_pause_s < 0:
        raise ValueError("Fenêtre ou pause maximale invalide.")
    ordered = sorted((start, end) for start, end in regions if end > start)
    if not ordered:
        return []
    groups: list[list[tuple[float, float]]] = [[ordered[0]]]
    for start, end in ordered[1:]:
        group = groups[-1]
        group_start = group[0][0]
        group_end = group[-1][1]
        pause = start - group_end
        if pause > max_pause_s or (end - group_start) > max_window_s:
            groups.append([(start, end)])
        else:
            group.append((start, end))
    windows: list[AudioWindow] = []
    for group in groups:
        windows.extend(_cap_span(group[0][0], group[-1][1], max_window_s))
    return windows


def _cap_span(start_s: float, end_s: float, max_window_s: float) -> list[AudioWindow]:
    windows: list[AudioWindow] = []
    cursor = start_s
    while end_s - cursor > 1e-9:
        piece_end = min(end_s, cursor + max_window_s)
        if piece_end <= cursor:
            break
        windows.append(AudioWindow(cursor, piece_end, cursor, piece_end))
        cursor = piece_end
    return windows


def covering_speech(windows: list[AudioWindow], regions: list[tuple[float, float]]) -> list[AudioWindow]:
    kept = []
    for window in windows:
        if any(window.start_s < end and window.end_s > start for start, end in regions):
            kept.append(window)
    return kept


def commit_transcripts(
    segments: list[TranscriptSegment],
    commit_start_s: float,
    commit_end_s: float,
) -> list[TranscriptSegment]:
    """Ne garde que le texte dont le début tombe dans la région commise."""
    kept: list[TranscriptSegment] = []
    for segment in segments:
        if segment.status != "completed":
            continue
        if segment.words:
            words = []
            clipped = False
            for word in segment.words:
                if not (commit_start_s <= word.start_s < commit_end_s):
                    clipped = True
                    continue
                end_s = min(word.end_s, commit_end_s)
                if end_s <= word.start_s:
                    clipped = True
                    continue
                if end_s < word.end_s:
                    clipped = True
                words.append(WordTiming(word.start_s, end_s, word.text))
            if not words:
                continue
            text = segment.text if not clipped else _join_word_text(words)
            rebuilt = TranscriptSegment(
                session_id=segment.session_id,
                source_segment_id=segment.source_segment_id,
                start_s=words[0].start_s,
                end_s=words[-1].end_s,
                text=text,
                language=segment.language,
                status="completed",
                speaker_id=None,
                words=tuple(words),
            )
            rebuilt.validate()
            kept.append(rebuilt)
            continue
        if not (commit_start_s <= segment.start_s < commit_end_s):
            continue
        end_s = min(segment.end_s, commit_end_s)
        if end_s <= segment.start_s:
            continue
        clipped = TranscriptSegment(
            session_id=segment.session_id,
            source_segment_id=segment.source_segment_id,
            start_s=segment.start_s,
            end_s=end_s,
            text=segment.text,
            language=segment.language,
            status="completed",
            speaker_id=None,
            words=(),
        )
        clipped.validate()
        kept.append(clipped)
    return kept


def _join_word_text(words: list[WordTiming]) -> str:
    parts: list[str] = []
    for word in words:
        token = word.text.strip()
        if not token:
            continue
        if parts and token[0] in {"'", "’"}:
            parts[-1] = parts[-1] + token
        else:
            parts.append(token)
    return " ".join(parts)


def _same_acoustic_interval(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    """Deux hypothèses décrivent le même son si leurs intervalles se recouvrent largement."""
    overlap = min(end_a, end_b) - max(start_a, start_b)
    shorter = min(end_a - start_a, end_b - start_b)
    if shorter <= 0:
        return False
    return overlap >= 0.5 * shorter


def _suppress_repeated_tokens(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Retire un mot seulement s'il répète une autre fenêtre sur le même intervalle.

    « Non non » dans la même hypothèse est une répétition réelle et reste.
    """
    published: list[tuple[str, str, float, float]] = []
    kept: list[TranscriptSegment] = []
    for segment in segments:
        if not segment.words:
            kept.append(segment)
            continue
        words: list[WordTiming] = []
        for word in segment.words:
            key = word.text.strip().lower()
            duplicate = any(
                source_id != segment.source_segment_id
                and previous_key == key
                and _same_acoustic_interval(previous_start, previous_end, word.start_s, word.end_s)
                for source_id, previous_key, previous_start, previous_end in published
            )
            if duplicate:
                continue
            words.append(word)
            published.append((segment.source_segment_id, key, word.start_s, word.end_s))
        if not words:
            continue
        if len(words) == len(segment.words):
            kept.append(segment)
            continue
        rebuilt = TranscriptSegment(
            session_id=segment.session_id,
            source_segment_id=segment.source_segment_id,
            start_s=words[0].start_s,
            end_s=words[-1].end_s,
            text=_join_word_text(words),
            language=segment.language,
            status="completed",
            speaker_id=None,
            words=tuple(words),
        )
        rebuilt.validate()
        kept.append(rebuilt)
    return kept


def transcribe_windows(
    transcriber: WhisperTranscriber,
    samples: np.ndarray,
    windows: list[AudioWindow],
    *,
    session_id: str,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> list[TranscriptSegment]:
    if sample_rate <= 0:
        raise ValueError(f"Fréquence d'échantillonnage invalide : {sample_rate}")
    if sample_rate != TARGET_SAMPLE_RATE:
        LOGGER.warning(
            "Audio à %s Hz ramené à %s Hz avant Whisper.",
            sample_rate,
            TARGET_SAMPLE_RATE,
        )
    normalizer = AudioNormalizer()
    produced: list[TranscriptSegment] = []
    for index, window in enumerate(windows, start=1):
        start = int(round(window.start_s * sample_rate))
        end = int(round(window.end_s * sample_rate))
        piece = np.asarray(samples[start:end], dtype=np.float32)
        if piece.size == 0:
            continue
        if sample_rate != TARGET_SAMPLE_RATE:
            piece = normalizer.resample(piece, sample_rate)
        if piece.size == 0:
            continue
        raw = transcriber.transcribe_samples(
            piece,
            session_id=session_id,
            source_segment_id=f"window_{index:03d}",
            offset_s=window.start_s,
            end_limit_s=window.end_s,
        )
        produced.extend(commit_transcripts(raw, window.commit_start_s, window.commit_end_s))
    return _suppress_repeated_tokens(produced)
