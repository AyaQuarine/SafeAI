"""Régressions supplémentaires SafeAI : cycle de vie et tolérance aux erreurs.

Ces deux tests expriment les exigences d'un service réutilisable et d'un
worker robuste. Ils échouent sur les versions du 02/11/2026 jointes à l'audit.
"""

import queue

import numpy as np

from safeai.acquisition.config import AcquisitionConfig, VADSettings
from safeai.acquisition.service import AcquisitionService
from safeai.acquisition.sources import SyntheticSource
from safeai.acquisition.vad import EnergyVADDetector
from safeai.acquisition.contracts import SpeechSegment
from safeai.transcription.contracts import TranscriptSegment
from safeai.transcription.worker import WhisperWorker


def test_run_source_can_start_a_second_session() -> None:
    """L'arrêt de la première session ne doit pas désactiver la suivante."""
    cfg = AcquisitionConfig(vad=VADSettings(backend="energy"))
    service = AcquisitionService(config=cfg, vad=EnergyVADDetector(cfg.vad))
    for index in range(2):
        source = SyntheticSource(
            np.zeros(2 * 16000, dtype=np.float32),
            session_id=f"s{index}",
            chunk_duration_s=1.0,
        )
        result = service.run_source(source)
        assert result.chunks_published == 2, (
            f"Session {index+1} : {result.chunks_published} blocs au lieu de 2"
        )


def test_invalid_model_output_does_not_escape_worker() -> None:
    """Une hypothèse mal formée doit être enregistrée comme échec, pas tuer le worker."""
    class MalformedTranscriber:
        def transcribe_speech(self, segment, source_segment_id):
            return [
                TranscriptSegment(
                    session_id=segment.session_id,
                    source_segment_id=source_segment_id,
                    start_s=segment.start_s,
                    end_s=segment.start_s,  # Invalide intentionnellement
                    text="Bonjour",
                    language="fr",
                    status="completed",
                )
            ]

    speech = SpeechSegment(
        session_id="s", start_s=0.0, end_s=1.0,
        speech_detected=True, vad_mean_probability=0.95,
        audio_available=True, samples=np.zeros(16000, dtype=np.float32),
    )
    worker = WhisperWorker(MalformedTranscriber(), queue.Queue(), clock=lambda: 1.1)
    worker._transcribe(speech)
    assert worker.errors == 1
    assert worker.published[-1].segment.status == "failed"
