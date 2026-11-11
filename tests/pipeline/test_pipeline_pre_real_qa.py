"""Régressions de supervision avant les premiers R01/R02/R06 bout en bout.

Placement conseillé: tests/pipeline/test_pipeline_pre_real_qa.py
Ne charge aucun modèle et ne sollicite pas le réseau.
"""

from pathlib import Path
import numpy as np
import soundfile as sf

from safeai.diarization.contracts import DiarizationResult, SpeakerTurn
from safeai.pipeline.offline import AcquisitionSnapshot, CommonPipeline
from safeai.transcription.contracts import TranscriptSegment, WordTiming

SID = "session_preflight"

class Acquisition:
    def run_file(self, path: Path, *, session_id: str):
        return AcquisitionSnapshot(session_id=session_id, duration_s=3.0, source=str(path))

class Transcriber:
    def __init__(self, *, empty=False):
        self.empty=empty
    def transcribe_file(self, path: Path, *, session_id: str, duration_s=None):
        if self.empty:
            return [TranscriptSegment(session_id, "file", 0.0, 3.0, "", "fr", "empty")]
        return [TranscriptSegment(session_id, "file", 1.0, 2.0, "bonjour", "fr", "completed", words=(WordTiming(1.0,2.0,"bonjour"),))]

class Diarizer:
    def __init__(self, *, wrong_session=False, speakers=True):
        self.wrong_session=wrong_session
        self.speakers=speakers
    def run_file(self, path: Path, *, session_id: str):
        did = "other_session" if self.wrong_session else session_id
        turns = (SpeakerTurn(did, 1.0, 2.0, "speaker_00"),) if self.speakers else ()
        return DiarizationResult(session_id=did, duration_s=3.0,
                                 status="completed" if turns else "empty",
                                 turns=turns, speakers_estimated=len(turns))

def _wav(tmp_path):
    wav=tmp_path/"example.wav"
    sf.write(wav,np.zeros(48000,dtype=np.float32),16000)
    return wav

def test_pipeline_rejects_diarization_session_mismatch_as_failed(tmp_path):
    """Doit renvoyer un JSON failed, pas remonter InvalidAlignmentError."""
    result=CommonPipeline(Acquisition(),Transcriber(),Diarizer(wrong_session=True)).run_file(_wav(tmp_path),session_id=SID)
    assert result.status=="failed"
    assert result.error is not None and "session" in result.error

def test_asr_empty_but_diarization_detects_speech_is_not_silence(tmp_path):
    """Une absence de texte malgré un tour de parole ne prouve pas le silence."""
    result=CommonPipeline(Acquisition(),Transcriber(empty=True),Diarizer()).run_file(_wav(tmp_path),session_id=SID)
    assert result.status != "empty"
    assert result.status in {"partial", "failed"}
