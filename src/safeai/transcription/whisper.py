"""faster-whisper sur des segments déjà consolidés.

Le VAD interne de Whisper est coupé : Silero a déjà découpé la parole.
Un temps local de 0,35 s dans un segment qui commence à 12,40 s devient 12,75 s.
L'audio est déjà en 16 kHz mono float32 : on ne le redécode pas via PyAV.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from safeai.acquisition.contracts import TARGET_SAMPLE_RATE, SpeechSegment
from safeai.acquisition.errors import ConfigurationError
from safeai.transcription.config import TranscriptionConfig
from safeai.transcription.contracts import TranscriptSegment, WordTiming

LOGGER = logging.getLogger(__name__)


class WhisperTranscriber:
    def __init__(self, config: TranscriptionConfig | None = None, model=None) -> None:
        self.config = config or TranscriptionConfig()
        self._model = model

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model(self.config)
        return self._model

    def transcribe_samples(
        self,
        samples: np.ndarray,
        *,
        session_id: str,
        source_segment_id: str,
        offset_s: float,
        end_limit_s: float | None = None,
    ) -> list[TranscriptSegment]:
        audio = np.asarray(samples)
        if audio.ndim != 1:
            raise ValueError("Whisper attend un audio mono")
        if audio.size == 0:
            return []
        if not np.all(np.isfinite(audio)):
            raise ValueError("Audio invalide")
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        available_end = offset_s + len(audio) / TARGET_SAMPLE_RATE
        limit = available_end if end_limit_s is None else min(float(end_limit_s), available_end)
        if limit <= offset_s:
            return []
        return self._transcribe_audio(
            audio,
            session_id=session_id,
            source_segment_id=source_segment_id,
            offset_s=offset_s,
            end_limit_s=limit,
        )

    def transcribe_speech(self, segment: SpeechSegment, source_segment_id: str) -> list[TranscriptSegment]:
        segment.validate()
        if segment.samples is None:
            return []
        return self.transcribe_samples(
            segment.samples,
            session_id=segment.session_id,
            source_segment_id=source_segment_id,
            offset_s=segment.start_s,
            end_limit_s=segment.end_s,
        )

    def transcribe_file(
        self,
        path: str | Path,
        *,
        session_id: str,
        source_segment_id: str = "file",
        offset_s: float = 0.0,
        duration_s: float | None = None,
    ) -> list[TranscriptSegment]:
        samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1).astype(np.float32)
        if rate != TARGET_SAMPLE_RATE:
            from safeai.acquisition.normalizer import AudioNormalizer

            audio = AudioNormalizer(TARGET_SAMPLE_RATE).convert(audio, rate)
        if duration_s is not None:
            if duration_s <= 0:
                raise ValueError("duration_s doit être positive")
            keep = min(len(audio), int(round(duration_s * TARGET_SAMPLE_RATE)))
            audio = audio[:keep]
        return self.transcribe_samples(
            audio,
            session_id=session_id,
            source_segment_id=source_segment_id,
            offset_s=offset_s,
            end_limit_s=offset_s + len(audio) / TARGET_SAMPLE_RATE,
        )

    def _transcribe_audio(
        self,
        audio: np.ndarray,
        *,
        session_id: str,
        source_segment_id: str,
        offset_s: float,
        end_limit_s: float,
    ) -> list[TranscriptSegment]:
        try:
            raw_segments, info = self.model.transcribe(
                audio,
                language=self.config.language,
                task=self.config.task,
                beam_size=self.config.beam_size,
                vad_filter=self.config.vad_filter,
                word_timestamps=self.config.word_timestamps,
            )
            raw_list = list(raw_segments)
        except ConfigurationError:
            raise
        except Exception:
            LOGGER.exception("La transcription a échoué")
            failed = TranscriptSegment(
                session_id=session_id,
                source_segment_id=source_segment_id,
                start_s=round(offset_s, 3),
                end_s=round(end_limit_s, 3),
                text="",
                language=self.config.language,
                status="failed",
                speaker_id=None,
            )
            failed.validate()
            return [failed]

        language = getattr(info, "language", None) or self.config.language
        produced: list[TranscriptSegment] = []
        for raw in raw_list:
            text = str(getattr(raw, "text", "") or "").strip()
            if not text:
                continue
            start_s, end_s = _bound_interval(
                offset_s + float(raw.start),
                offset_s + float(raw.end),
                low=offset_s,
                high=end_limit_s,
                kind="segment",
            ) or (None, None)
            if start_s is None or end_s is None:
                continue
            segment = TranscriptSegment(
                session_id=session_id,
                source_segment_id=source_segment_id,
                start_s=start_s,
                end_s=end_s,
                text=text,
                language=language,
                status="completed",
                speaker_id=None,
                words=_global_words(raw, offset_s, start_s, end_s),
            )
            segment.validate()
            produced.append(segment)
        if produced:
            return produced
        empty = TranscriptSegment(
            session_id=session_id,
            source_segment_id=source_segment_id,
            start_s=round(offset_s, 3),
            end_s=round(end_limit_s, 3),
            text="",
            language=language,
            status="empty",
            speaker_id=None,
        )
        empty.validate()
        return [empty]


def _accept_pcm_directly() -> None:
    """Les tests QA patchent ce nom. Il ne doit plus être appelé.

    faster-whisper accepte déjà un ndarray mono. Modifier
    `faster_whisper.transcribe.decode_audio` casserait les autres appels.
    """
    raise RuntimeError("Ne pas modifier faster_whisper.transcribe.decode_audio")


def _bound_interval(
    start_s: float,
    end_s: float,
    *,
    low: float,
    high: float,
    kind: str,
) -> tuple[float, float] | None:
    """Ramène un intervalle dans des bornes physiques. None s'il devient vide."""
    raw_start, raw_end = start_s, end_s
    start_s = min(high, max(low, start_s))
    end_s = min(high, max(low, end_s))
    start_s = min(high, max(low, round(start_s, 3)))
    end_s = min(high, max(low, round(end_s, 3)))
    if end_s <= start_s:
        if abs(raw_start - start_s) > 0.02 or abs(raw_end - end_s) > 0.02:
            LOGGER.warning(
                "Horodatage %s rejeté : %.3f–%.3f hors de %.3f–%.3f",
                kind, raw_start, raw_end, low, high,
            )
        return None
    if abs(raw_start - start_s) > 0.02 or abs(raw_end - end_s) > 0.02:
        LOGGER.warning(
            "Horodatage %s corrigé : %.3f–%.3f ramené à %.3f–%.3f",
            kind, raw_start, raw_end, start_s, end_s,
        )
    return start_s, end_s


def _global_words(
    raw,
    offset_s: float,
    segment_start_s: float,
    segment_end_s: float,
) -> tuple[WordTiming, ...]:
    timings = []
    for word in getattr(raw, "words", None) or []:
        text = str(word.word).strip()
        if not text:
            continue
        bounded = _bound_interval(
            offset_s + float(word.start),
            offset_s + float(word.end),
            low=segment_start_s,
            high=segment_end_s,
            kind="mot",
        )
        if bounded is None:
            continue
        start_s, end_s = bounded
        timings.append(WordTiming(start_s=start_s, end_s=end_s, text=text))
    return tuple(timings)


def _load_model(config: TranscriptionConfig):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ConfigurationError(
            "faster-whisper est requis pour la transcription. "
            "Installez-le avec : python -m pip install faster-whisper jiwer"
        ) from exc
    return WhisperModel(config.model_size, device=config.device, compute_type=config.compute_type)
