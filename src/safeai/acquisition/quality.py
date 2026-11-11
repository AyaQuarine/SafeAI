"""Contrôle de qualité technique — distinct du niveau de risque SafeAI."""

from __future__ import annotations

import numpy as np

from safeai.acquisition.config import QualitySettings
from safeai.acquisition.contracts import (
    AudioChunk,
    AudioQualityEvent,
    MuteStatus,
    QualityStatus,
    SpeechActivity,
    StreamStatus,
)


class QualityMonitor:
    """Calcule des indicateurs techniques par fenêtre, sans interpréter la scène.

    Un RMS faible n'est pas un micro coupé. Un micro coupé n'est affirmé
    que si une information système le confirme. Sinon mute_status reste UNKNOWN.
    Un signal intense et propre n'est pas une anomalie de qualité.
    """

    def __init__(self, settings: QualitySettings | None = None) -> None:
        self.settings = settings or QualitySettings()
        self._noise_estimate = 0.0
        self._noise_initialized = False
        self._silence_s = 0.0

    def reset(self) -> None:
        self._noise_estimate = 0.0
        self._noise_initialized = False
        self._silence_s = 0.0

    @property
    def silence_duration_s(self) -> float:
        return self._silence_s

    def break_silence_run(self) -> None:
        """Une perte de flux coupe le silence. Les deux côtés ne s'additionnent pas."""
        self._silence_s = 0.0

    def observe(
        self,
        chunk: AudioChunk,
        speech_activity: SpeechActivity = SpeechActivity.UNKNOWN,
        *,
        speech_active_at_end: bool | None = None,
    ) -> AudioQualityEvent:
        samples = chunk.samples
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        clipping_ratio = float(np.mean(np.abs(samples) >= self.settings.clipping_amplitude))
        roughness = _roughness(samples)
        active_at_end = (
            speech_activity is SpeechActivity.SPEECH
            if speech_active_at_end is None
            else speech_active_at_end
        )

        # Le suivi du silence dépend de speech_active_at_end, pas de la
        # simple présence de parole dans la fenêtre.
        if active_at_end:
            self._silence_s = 0.0
        elif speech_activity is SpeechActivity.NON_SPEECH:
            self._update_noise(rms, fast=True)
            self._silence_s += chunk.duration_s
        elif speech_activity is SpeechActivity.SPEECH:
            # La parole s'est arrêtée dans la fenêtre. La queue basse énergie
            # est du silence réel ; l'effacer sous-estime SC11.
            self._silence_s = _trailing_silence_s(
                samples, chunk.sample_rate, self.settings.rms_silence
            )
        elif rms < self.settings.rms_silence:
            self._update_noise(rms, fast=True)
            self._silence_s += chunk.duration_s
        else:
            self._silence_s = 0.0

        quality_status, reason = self._classify(
            chunk=chunk,
            rms=rms,
            peak=peak,
            clipping_ratio=clipping_ratio,
            roughness=roughness,
            speech_activity=speech_activity,
        )

        return AudioQualityEvent(
            session_id=chunk.session_id,
            start_s=chunk.start_s,
            end_s=chunk.end_s,
            stream_status=chunk.stream_status,
            mute_status=chunk.mute_status,
            speech_activity=speech_activity,
            speech_active_at_end=active_at_end,
            quality_status=quality_status,
            rms=rms,
            peak=peak,
            clipping_ratio=clipping_ratio,
            noise_estimate=self._noise_estimate,
            silence_duration_s=self._silence_s,
            buffer_overflow=chunk.samples_dropped > 0
            or chunk.stream_status is StreamStatus.INTERRUPTED,
            reason=reason,
            samples_dropped=chunk.samples_dropped,
        )

    def _update_noise(self, rms: float, fast: bool) -> None:
        if not self._noise_initialized:
            self._noise_estimate = rms
            self._noise_initialized = True
            return
        alpha = 0.25 if fast else 0.08
        self._noise_estimate = ((1.0 - alpha) * self._noise_estimate) + (alpha * rms)

    def _classify(
        self,
        chunk: AudioChunk,
        rms: float,
        peak: float,
        clipping_ratio: float,
        roughness: float,
        speech_activity: SpeechActivity,
    ) -> tuple[QualityStatus, str]:
        if chunk.stream_status is StreamStatus.UNAVAILABLE:
            return QualityStatus.UNUSABLE, "stream_unavailable"
        if chunk.mute_status is MuteStatus.MUTED:
            return QualityStatus.UNUSABLE, "microphone_muted"
        if chunk.samples_dropped > 0:
            return QualityStatus.DEGRADED, "samples_dropped"
        if chunk.stream_status is StreamStatus.INTERRUPTED:
            return QualityStatus.DEGRADED, "stream_interrupted"
        if clipping_ratio >= self.settings.clipping_ratio_unusable:
            return QualityStatus.UNUSABLE, "clipping"
        if clipping_ratio >= self.settings.clipping_ratio_degraded:
            return QualityStatus.DEGRADED, "clipping"

        relative_roughness = roughness / max(rms, 1e-8)
        broadband_noise = rms >= self.settings.noise_degraded and relative_roughness >= 0.55
        tracked_noise = self._noise_estimate >= self.settings.noise_degraded
        if broadband_noise or (
            tracked_noise
            and (
                speech_activity is not SpeechActivity.SPEECH
                or (rms / max(self._noise_estimate, 1e-8)) < 2.0
            )
        ):
            return QualityStatus.DEGRADED, "background_noise"

        # Un signal intense n'est pas, à lui seul, une anomalie de qualité (SC07).
        if peak >= self.settings.peak_intense and clipping_ratio < self.settings.clipping_ratio_degraded:
            return QualityStatus.OK, "intense_but_clean"
        if speech_activity is SpeechActivity.NON_SPEECH:
            return QualityStatus.OK, "silence"
        return QualityStatus.OK, "ok"


def _trailing_silence_s(samples: np.ndarray, sample_rate: int, rms_silence: float) -> float:
    """Durée basse énergie en fin de fenêtre, mesurée par pas de 20 ms."""
    if samples.size == 0 or sample_rate <= 0:
        return 0.0
    frame = max(1, int(round(0.02 * sample_rate)))
    quiet = 0
    end = int(samples.size)
    while end > 0:
        start = max(0, end - frame)
        window = samples[start:end]
        rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
        if rms >= rms_silence:
            break
        quiet += end - start
        end = start
    return quiet / sample_rate


def _roughness(samples: np.ndarray) -> float:
    if samples.size < 2:
        return 0.0
    residual = np.diff(np.asarray(samples, dtype=np.float64))
    return float(np.sqrt(np.mean(np.square(residual))))
