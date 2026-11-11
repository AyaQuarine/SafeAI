"""Régressions COMMON-019 (qualité temporelle des tokens) et COMMON-020 (parole non transcrite)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from safeai.diarization.contracts import DiarizationResult, SpeakerTurn
from safeai.pipeline.offline import AcquisitionSnapshot, CommonPipeline, SpeechInterval
from safeai.transcription.contracts import TranscriptSegment, WordTiming

SID = "session_common_019_020"
DUR = 40.0


class _Acq:
    def __init__(self, snapshot: AcquisitionSnapshot) -> None:
        self.snapshot = snapshot

    def run_file(self, path: Path, *, session_id: str) -> AcquisitionSnapshot:
        assert session_id == SID
        return self.snapshot


class _Asr:
    def __init__(self, segments: list[TranscriptSegment]) -> None:
        self.segments = segments

    def transcribe_file(self, path: Path, *, session_id: str, duration_s=None):
        assert session_id == SID
        return self.segments


class _Diar:
    def __init__(self, turns: tuple[SpeakerTurn, ...]) -> None:
        self.turns = turns

    def run_file(self, path: Path, *, session_id: str) -> DiarizationResult:
        assert session_id == SID
        speakers = {turn.speaker_id for turn in self.turns if turn.speaker_id}
        return DiarizationResult(
            session_id=SID,
            duration_s=DUR,
            status="completed",
            turns=self.turns,
            speakers_estimated=len(speakers),
        )


def _wav(tmp_path: Path) -> Path:
    path = tmp_path / "mock.wav"
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
    return path


def _word_segment(
    start: float,
    end: float,
    text: str,
    *,
    words: tuple[WordTiming, ...] | None = None,
) -> TranscriptSegment:
    timing = words if words is not None else (WordTiming(start, end, text),)
    return TranscriptSegment(
        session_id=SID,
        source_segment_id=f"{start}-{end}",
        start_s=start,
        end_s=end,
        text=text,
        language="fr",
        status="completed",
        words=timing,
    )


def test_oversized_non_lexical_token_is_discarded_but_real_words_remain(tmp_path: Path):
    """COMMON-019 : un « ? » de 14,86 s est écarté ; les mots et la ponctuation courte restent."""
    snapshot = AcquisitionSnapshot(
        session_id=SID,
        duration_s=DUR,
        source="mock",
        evidence_available=True,
        speech_intervals=(
            SpeechInterval(22.5, 24.7, 0.95),
            SpeechInterval(25.5, 29.0, 0.94),
            SpeechInterval(30.0, 34.7, 0.93),
        ),
        signal_usable=True,
        silence_like=False,
    )
    turns = (
        SpeakerTurn(SID, 22.54, 24.72, "speaker_00"),
        SpeakerTurn(SID, 25.55, 29.04, "speaker_01"),
        SpeakerTurn(SID, 30.07, 34.73, "speaker_00"),
    )
    segments = [
        _word_segment(
            22.76,
            24.64,
            "une phrase court ?",
            words=(
                WordTiming(22.76, 23.36, "une"),
                WordTiming(23.36, 23.80, "phrase"),
                WordTiming(23.80, 24.40, "court"),
                WordTiming(24.40, 24.55, "?"),
            ),
        ),
        _word_segment(24.64, 39.50, "?", words=(WordTiming(24.64, 39.50, "?"),)),
    ]
    result = CommonPipeline(_Acq(snapshot), _Asr(segments), _Diar(turns)).run_file(
        _wav(tmp_path), session_id=SID
    )
    texts = [item.text for item in result.utterances]
    joined = " ".join(texts)
    assert "?" not in texts
    assert not any(item.text.strip() == "?" for item in result.utterances)
    assert "phrase" in joined
    assert "court" in joined
    assert result.status == "partial"
    assert any("token non lexical trop long" in note for note in result.diagnostics)
    # Ponctuation normale courte conservée dans le texte joint ou absente si collée — pas d'écart forcé.
    assert not any("24.40-24.55" in note for note in result.diagnostics)


def test_normal_short_punctuation_is_kept(tmp_path: Path):
    snapshot = AcquisitionSnapshot(
        session_id=SID,
        duration_s=DUR,
        source="mock",
        evidence_available=True,
        speech_intervals=(SpeechInterval(1.0, 2.5, 0.9),),
        signal_usable=True,
        silence_like=False,
    )
    turns = (SpeakerTurn(SID, 1.0, 2.5, "speaker_00"),)
    segments = [
        _word_segment(
            1.0,
            2.2,
            "Bonjour ?",
            words=(
                WordTiming(1.0, 1.8, "Bonjour"),
                WordTiming(1.8, 2.0, "?"),
            ),
        )
    ]
    result = CommonPipeline(_Acq(snapshot), _Asr(segments), _Diar(turns)).run_file(
        _wav(tmp_path), session_id=SID
    )
    joined = " ".join(item.text for item in result.utterances)
    assert "Bonjour" in joined
    assert "?" in joined
    assert not any("token non lexical trop long" in note for note in result.diagnostics)


def test_untranscribed_community1_speech_is_diagnosed(tmp_path: Path):
    """COMMON-020 : un tour Community-1 sans texte lexical apparaît dans diagnostics."""
    snapshot = AcquisitionSnapshot(
        session_id=SID,
        duration_s=DUR,
        source="mock",
        evidence_available=True,
        speech_intervals=(
            SpeechInterval(1.0, 2.5, 0.95),
            SpeechInterval(25.5, 29.0, 0.94),
        ),
        signal_usable=True,
        silence_like=False,
    )
    turns = (
        SpeakerTurn(SID, 1.0, 2.5, "speaker_00"),
        SpeakerTurn(SID, 25.55, 29.04, "speaker_01"),
    )
    segments = [_word_segment(1.0, 2.0, "Bonjour")]
    bundle = CommonPipeline(_Acq(snapshot), _Asr(segments), _Diar(turns)).run_bundle(
        _wav(tmp_path), session_id=SID
    )
    assert bundle.aligned.status == "partial"
    assert any(
        "parole Community-1 non transcrite" in note and "25.55" in note
        for note in bundle.aligned.diagnostics
    )
    assert bundle.diarization is not None
    assert len(bundle.diarization.turns) == 2
    # Aucun énoncé inventé pour combler le trou.
    assert not any(25.0 <= item.start_s <= 29.5 for item in bundle.aligned.utterances)
