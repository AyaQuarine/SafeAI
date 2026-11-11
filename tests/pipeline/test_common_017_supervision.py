"""Régressions de supervision COMMON-017 (mocks, pas une évaluation des vrais WAV).

Tests additionnels à exécuter après intégration à tests/pipeline/.
Les assertions définissent les garanties de sécurité des sorties, pas une technique de filtrage.
"""
from pathlib import Path
import numpy as np
import soundfile as sf
from safeai.pipeline.offline import AcquisitionSnapshot, CommonPipeline, SpeechInterval
from safeai.diarization.contracts import DiarizationResult, SpeakerTurn
from safeai.transcription.contracts import TranscriptSegment, WordTiming

SID = 'session_common_017_supervision'
DUR = 20.0

class AcquisitionStub:
    def __init__(self, snapshot):
        self.snapshot = snapshot
    def run_file(self, path: Path, *, session_id: str):
        assert session_id == SID
        return self.snapshot

class TranscriberStub:
    def __init__(self, segments):
        self.segments = segments
    def transcribe_file(self, path: Path, *, session_id: str, duration_s=None):
        assert session_id == SID
        return self.segments

class DiarizerStub:
    def __init__(self, turns=()):
        self.turns = turns
    def run_file(self, path: Path, *, session_id: str):
        assert session_id == SID
        return DiarizationResult(
            session_id=SID, duration_s=DUR,
            status='completed' if self.turns else 'empty',
            turns=tuple(self.turns),
            speakers_estimated=len({t.speaker_id for t in self.turns if t.speaker_id}),
        )

def wav(tmp_path):
    p = tmp_path / 'mock.wav'
    sf.write(p, np.zeros(16000, dtype=np.float32), 16000)
    return p

def transcript(start, end, text):
    return TranscriptSegment(
        session_id=SID, source_segment_id=f'{start}-{end}',
        start_s=start, end_s=end, text=text, language='fr', status='completed',
        words=(WordTiming(start, end, text),),
    )

def test_unknown_quality_does_not_prove_silence():
    snapshot = AcquisitionSnapshot(
        session_id=SID, duration_s=DUR, source='mock', evidence_available=True,
        speech_intervals=(), signal_usable=None, silence_like=None,
    )
    assert snapshot.silence_confirmed is False, 'Les valeurs None ne sont pas une preuve positive de silence'

def test_silero_speech_without_asr_or_diarization_is_not_empty(tmp_path):
    snapshot = AcquisitionSnapshot(
        session_id=SID, duration_s=DUR, source='mock', evidence_available=True,
        speech_intervals=(SpeechInterval(2.0, 3.0, 0.93),),
        signal_usable=True, silence_like=False,
    )
    result = CommonPipeline(
        AcquisitionStub(snapshot), TranscriberStub([]), DiarizerStub()
    ).run_file(wav(tmp_path), session_id=SID)
    assert result.status != 'empty', 'Parole Silero sans texte n’est pas un silence établi'

def test_asr_tail_without_acoustic_support_is_not_declared_complete(tmp_path):
    snapshot = AcquisitionSnapshot(
        session_id=SID, duration_s=DUR, source='mock', evidence_available=True,
        speech_intervals=(SpeechInterval(1.0, 2.5, 0.94),),
        signal_usable=True, silence_like=False,
    )
    turn = SpeakerTurn(session_id=SID, start_s=1.0, end_s=2.5,
                       speaker_id='speaker_00', status='completed')
    result = CommonPipeline(
        AcquisitionStub(snapshot),
        TranscriberStub([transcript(1.0, 2.0, 'Bonjour'),
                         transcript(12.0, 16.0, 'Sous-titres Amara')]),
        DiarizerStub([turn]),
    ).run_file(wav(tmp_path), session_id=SID)
    unsupported_text = any('Sous-titres' in u.text for u in result.utterances)
    assert result.status != 'completed' or not unsupported_text, (
        'Un texte sans support sur sa propre plage acoustique ne doit pas être livré comme completed'
    )
