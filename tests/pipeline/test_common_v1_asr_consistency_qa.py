"""Tests de supervision : cohérence des sorties ASR / diarisation (Common V1).

À placer dans tests/pipeline/. Ne télécharge pas de modèle.
Le premier test est une régression attendue sur la version du 10/11/2026 :
il empêche un 'completed' non justifié lorsque Whisper invente du texte
sur un WAV silencieux que Community-1 classe 'empty'.

Politique conservatrice pour l'étape intermédiaire : si la validation acoustique
indépendante n'est pas exposée par AcquisitionSnapshot, signaler 'failed'.
Après exposition de la preuve VAD + qualité, faire évoluer vers 'empty' pour
un silence confirmé, en conservant le texte rejeté dans un journal diagnostic.
"""
from pathlib import Path

import numpy as np
import soundfile as sf

from safeai.pipeline.offline import AcquisitionSnapshot, CommonPipeline
from safeai.transcription.contracts import TranscriptSegment, WordTiming
from safeai.diarization.contracts import DiarizationResult, SpeakerTurn

SID = "session_common_quality_qa"


class _Acq:
    def __init__(self, *, silence_confirmed: bool = False) -> None:
        self.silence_confirmed = silence_confirmed

    def run_file(self, path: Path, *, session_id: str) -> AcquisitionSnapshot:
        if self.silence_confirmed:
            return AcquisitionSnapshot(
                session_id=session_id,
                duration_s=20.0,
                source=str(path),
                evidence_available=True,
                signal_usable=True,
                silence_like=True,
            )
        return AcquisitionSnapshot(session_id=session_id, duration_s=20.0, source=str(path))


class _Asr:
    def __init__(self, text: str):
        self.text = text

    def transcribe_file(self, path: Path, *, session_id: str, duration_s: float | None = None):
        if not self.text:
            return [TranscriptSegment(session_id=session_id, source_segment_id="file", start_s=0.0, end_s=20.0,
                                      text="", language="fr", status="empty")]
        return [TranscriptSegment(session_id=session_id, source_segment_id="file", start_s=18.1, end_s=19.5,
                                  text=self.text, language="fr", status="completed",
                                  words=(WordTiming(18.1, 19.5, self.text),))]


class _Diar:
    def __init__(self, speaking: bool):
        self.speaking = speaking

    def run_file(self, path: Path, *, session_id: str) -> DiarizationResult:
        if not self.speaking:
            return DiarizationResult(session_id=session_id, duration_s=20.0, status="empty", turns=(), speakers_estimated=0)
        return DiarizationResult(session_id=session_id, duration_s=20.0, status="completed",
                                 turns=(SpeakerTurn(session_id=session_id, start_s=18.1, end_s=19.6,
                                                    speaker_id="speaker_00"),), speakers_estimated=1)


def _wav(tmp_path: Path) -> Path:
    path = tmp_path / "r02_synthetic.wav"
    sf.write(str(path), np.zeros(16000, np.float32), 16000)
    return path


def test_asr_text_without_diarized_speech_is_not_unverified_completed(tmp_path: Path):
    """R02 : le texte 'Sous' ne peut pas valider à lui seul une parole."""
    result = CommonPipeline(_Acq(), _Asr("Sous"), _Diar(False)).run_file(_wav(tmp_path), session_id=SID)
    assert result.status == "failed", (
        "Contradiction ASR/diarisation : tant que les preuves VAD/qualité ne sont "
        "pas transmises, le statut doit rester failed et non completed."
    )
    assert result.error


def test_both_models_empty_is_empty(tmp_path: Path):
    result = CommonPipeline(
        _Acq(silence_confirmed=True), _Asr(""), _Diar(False)
    ).run_file(_wav(tmp_path), session_id=SID)
    assert result.status == "empty"
    assert result.utterances == ()


def test_both_models_empty_without_evidence_is_failed(tmp_path: Path):
    result = CommonPipeline(_Acq(), _Asr(""), _Diar(False)).run_file(
        _wav(tmp_path), session_id=SID
    )
    assert result.status == "failed"
    assert "Silence non vérifié" in (result.error or "")


def test_agreeing_models_still_complete(tmp_path: Path):
    result = CommonPipeline(_Acq(), _Asr("bonjour"), _Diar(True)).run_file(_wav(tmp_path), session_id=SID)
    assert result.status == "completed"
    assert result.utterances[0].speaker_id == "speaker_00"
