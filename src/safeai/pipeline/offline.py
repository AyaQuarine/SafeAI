"""Orchestrateur hors ligne : acquisition + Whisper + Community-1 + alignement.

Les composants sont injectables. Les modèles ne sont pas chargés ici :
un adaptateur peut appeler un autre environnement local si nécessaire.
Aucun jeton Hugging Face n'est journalisé ni passé en argument.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence

from safeai.acquisition.contracts import TARGET_SAMPLE_RATE, new_session_id
from safeai.alignment.contracts import AlignedSession, AlignedUtterance
from safeai.alignment.engine import AlignmentConfig, align_session
from safeai.alignment.errors import InvalidAlignmentError
from safeai.diarization.contracts import DiarizationResult
from safeai.pipeline.errors import PipelineConfigurationError
from safeai.transcription.contracts import TranscriptSegment, WordTiming

AcquisitionStatus = Literal["completed", "failed"]
PROVENANCE = "common_pipeline"
DEFAULT_LOCAL_SUPPORT_RATIO = 0.25
# Plafond initial de robustesse (COMMON-019) : un signe isolé trop long n'est pas lexical.
DEFAULT_MAX_NON_LEXICAL_DURATION_S = 2.0
# Seuil minimal pour signaler une parole Community-1 sans texte (COMMON-020).
DEFAULT_MIN_UNTRANSCRIBED_SPEECH_S = 0.3
_LEXICAL_CHAR = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]", re.UNICODE)


@dataclass(frozen=True)
class SpeechInterval:
    """Intervalle de parole Silero, sans échantillons."""

    start_s: float
    end_s: float
    vad_mean_probability: float = 0.0


@dataclass(frozen=True)
class AcquisitionSnapshot:
    """Résultat d'acquisition pour l'orchestration, avec preuves compactes."""

    session_id: str
    duration_s: float
    source: str
    status: AcquisitionStatus = "completed"
    error: str | None = None
    evidence_available: bool = False
    speech_intervals: tuple[SpeechInterval, ...] = ()
    signal_usable: bool | None = None
    silence_like: bool | None = None
    incidents: tuple[str, ...] = ()

    @property
    def silero_speech(self) -> bool:
        return any(interval.end_s > interval.start_s for interval in self.speech_intervals)

    @property
    def silence_confirmed(self) -> bool:
        """Silence prouvé : preuve VAD/qualité présente, aucune parole Silero, signal exploitable."""
        if not self.evidence_available:
            return False
        if self.silero_speech:
            return False
        # Aucune preuve positive n'est déduite de valeurs inconnues (None).
        if self.signal_usable is not True:
            return False
        if self.silence_like is not True:
            return False
        return True


class AcquisitionComponent(Protocol):
    def run_file(self, path: Path, *, session_id: str) -> AcquisitionSnapshot: ...


class TranscriptionComponent(Protocol):
    def transcribe_file(
        self,
        path: Path,
        *,
        session_id: str,
        duration_s: float | None = None,
    ) -> Sequence[TranscriptSegment]: ...


class DiarizationComponent(Protocol):
    def run_file(self, path: Path, *, session_id: str) -> DiarizationResult: ...


@dataclass(frozen=True)
class CommonBundle:
    """Sorties complémentaires d'une exécution hors ligne."""

    aligned: AlignedSession
    diarization: DiarizationResult | None = None
    acquisition: AcquisitionSnapshot | None = None


class CommonPipeline:
    """Enchaîne les trois moteurs puis ``align_session`` sur un seul WAV."""

    def __init__(
        self,
        acquisition: AcquisitionComponent,
        transcriber: TranscriptionComponent,
        diarizer: DiarizationComponent,
        *,
        alignment_config: AlignmentConfig | None = None,
        provenance: str = PROVENANCE,
        local_support_ratio: float = DEFAULT_LOCAL_SUPPORT_RATIO,
        max_non_lexical_duration_s: float = DEFAULT_MAX_NON_LEXICAL_DURATION_S,
        min_untranscribed_speech_s: float = DEFAULT_MIN_UNTRANSCRIBED_SPEECH_S,
    ) -> None:
        if acquisition is None or transcriber is None or diarizer is None:
            raise PipelineConfigurationError("acquisition, transcriber et diarizer sont requis")
        if not 0.0 <= local_support_ratio <= 1.0:
            raise PipelineConfigurationError("local_support_ratio doit rester dans [0, 1]")
        if max_non_lexical_duration_s < 0:
            raise PipelineConfigurationError("max_non_lexical_duration_s doit être ≥ 0")
        if min_untranscribed_speech_s < 0:
            raise PipelineConfigurationError("min_untranscribed_speech_s doit être ≥ 0")
        self.acquisition = acquisition
        self.transcriber = transcriber
        self.diarizer = diarizer
        self.alignment_config = alignment_config
        self.provenance = provenance
        self.local_support_ratio = local_support_ratio
        self.max_non_lexical_duration_s = max_non_lexical_duration_s
        self.min_untranscribed_speech_s = min_untranscribed_speech_s

    def run_file(
        self,
        path: str | Path,
        *,
        session_id: str | None = None,
    ) -> AlignedSession:
        return self.run_bundle(path, session_id=session_id).aligned

    def run_bundle(
        self,
        path: str | Path,
        *,
        session_id: str | None = None,
    ) -> CommonBundle:
        wav = Path(path)
        sid = session_id or new_session_id()
        if not sid:
            aligned = _failed(
                session_id="session_invalid",
                duration_s=0.0,
                source=str(wav),
                provenance=self.provenance,
                error="session_id manquant",
            )
            return CommonBundle(aligned=aligned)
        if not wav.is_file():
            aligned = _failed(
                session_id=sid,
                duration_s=0.0,
                source=str(wav),
                provenance=self.provenance,
                error=f"fichier audio introuvable : {wav}",
            )
            return CommonBundle(aligned=aligned)

        try:
            acquisition = self.acquisition.run_file(wav, session_id=sid)
        except Exception as exc:
            aligned = _failed(
                session_id=sid,
                duration_s=0.0,
                source=str(wav),
                provenance=self.provenance,
                error=f"acquisition : {exc}",
            )
            return CommonBundle(aligned=aligned)

        if acquisition.session_id != sid:
            aligned = _failed(
                session_id=sid,
                duration_s=max(0.0, float(acquisition.duration_s)),
                source=str(wav),
                provenance=self.provenance,
                error=(
                    "session incompatible : acquisition "
                    f"{acquisition.session_id} ≠ {sid}"
                ),
            )
            return CommonBundle(aligned=aligned, acquisition=acquisition)
        if acquisition.status == "failed":
            aligned = _failed(
                session_id=sid,
                duration_s=max(0.0, float(acquisition.duration_s)),
                source=str(wav),
                provenance=self.provenance,
                error=acquisition.error or "acquisition failed",
            )
            return CommonBundle(aligned=aligned, acquisition=acquisition)

        duration_s = float(acquisition.duration_s)
        if duration_s < 0:
            aligned = _failed(
                session_id=sid,
                duration_s=0.0,
                source=str(wav),
                provenance=self.provenance,
                error="duration_s invalide",
            )
            return CommonBundle(aligned=aligned, acquisition=acquisition)

        try:
            transcripts = list(
                self.transcriber.transcribe_file(
                    wav,
                    session_id=sid,
                    duration_s=duration_s,
                )
            )
        except Exception as exc:
            aligned = _failed(
                session_id=sid,
                duration_s=duration_s,
                source=str(wav),
                provenance=self.provenance,
                error=f"transcription : {exc}",
            )
            return CommonBundle(aligned=aligned, acquisition=acquisition)

        try:
            diarization = self.diarizer.run_file(wav, session_id=sid)
        except Exception as exc:
            aligned = _failed(
                session_id=sid,
                duration_s=duration_s,
                source=str(wav),
                provenance=self.provenance,
                error=f"diarisation : {exc}",
            )
            return CommonBundle(aligned=aligned, acquisition=acquisition)

        decision = _coherence_decision(acquisition, transcripts, diarization)
        if decision.outcome == "failed":
            aligned = _failed(
                session_id=sid,
                duration_s=duration_s,
                source=str(wav),
                provenance=self.provenance,
                error=decision.error or "incohérence des modèles",
                diagnostics=decision.diagnostics,
            )
            return CommonBundle(
                aligned=aligned,
                diarization=diarization,
                acquisition=acquisition,
            )
        if decision.outcome == "empty":
            aligned = AlignedSession(
                session_id=sid,
                duration_s=duration_s,
                utterances=(),
                offset_s=diarization.offset_s,
                language="fr",
                provenance=self.provenance,
                source=str(wav),
                status="empty",
                diagnostics=decision.diagnostics,
            )
            return CommonBundle(
                aligned=aligned,
                diarization=diarization,
                acquisition=acquisition,
            )

        filtered, filter_notes, rejected = _filter_transcripts_temporally(
            acquisition,
            decision.transcripts,
            diarization,
            min_overlap_ratio=self.local_support_ratio,
            max_non_lexical_duration_s=self.max_non_lexical_duration_s,
        )
        notes = decision.diagnostics + filter_notes
        if rejected and not filtered:
            # Toute la transcription locale a été écartée ; la diarisation reste exportée.
            uncovered = _untranscribed_speech_notes(
                diarization,
                (),
                min_duration_s=self.min_untranscribed_speech_s,
            )
            notes = tuple(dict.fromkeys([*notes, *uncovered]))
            aligned = AlignedSession(
                session_id=sid,
                duration_s=duration_s,
                utterances=(),
                offset_s=diarization.offset_s,
                language="fr",
                provenance=self.provenance,
                source=str(wav),
                status="partial" if _diarization_has_speech(diarization) or acquisition.silero_speech else "empty",
                diagnostics=notes,
            )
            # partial exige au moins un énoncé — si diarisation seule, failed explicite.
            if aligned.status == "partial" and not aligned.utterances:
                aligned = _failed(
                    session_id=sid,
                    duration_s=duration_s,
                    source=str(wav),
                    provenance=self.provenance,
                    error=(
                        "transcription locale sans support acoustique ; "
                        "tours Community-1 conservés à part"
                    ),
                    diagnostics=notes,
                )
            return CommonBundle(
                aligned=aligned,
                diarization=diarization,
                acquisition=acquisition,
            )

        try:
            session = align_session(
                filtered,
                diarization,
                session_id=sid,
                duration_s=duration_s,
                offset_s=diarization.offset_s,
                provenance=self.provenance,
                source=str(wav),
                audio_available=True,
                config=self.alignment_config,
            )
        except InvalidAlignmentError as exc:
            aligned = _failed(
                session_id=sid,
                duration_s=duration_s,
                source=str(wav),
                provenance=self.provenance,
                error=str(exc),
                diagnostics=notes,
            )
            return CommonBundle(
                aligned=aligned,
                diarization=diarization,
                acquisition=acquisition,
            )

        uncovered = _untranscribed_speech_notes(
            diarization,
            session.utterances,
            min_duration_s=self.min_untranscribed_speech_s,
        )
        status = session.status
        if (rejected or uncovered) and status == "completed":
            status = "partial"
        merged_notes = tuple(dict.fromkeys([*session.diagnostics, *notes, *uncovered]))
        if status != session.status or merged_notes != session.diagnostics:
            session = AlignedSession(
                session_id=session.session_id,
                duration_s=session.duration_s,
                utterances=session.utterances,
                offset_s=session.offset_s,
                language=session.language,
                provenance=session.provenance,
                source=session.source,
                status=status,
                error=session.error,
                diagnostics=merged_notes,
            )
        return CommonBundle(
            aligned=session,
            diarization=diarization,
            acquisition=acquisition,
        )


@dataclass(frozen=True)
class _CoherenceDecision:
    outcome: Literal["align", "empty", "failed"]
    transcripts: tuple[TranscriptSegment, ...] = ()
    error: str | None = None
    diagnostics: tuple[str, ...] = ()


def _interval_overlap_ratio(
    start_s: float,
    end_s: float,
    other_start: float,
    other_end: float,
) -> float:
    duration = end_s - start_s
    if duration <= 0:
        return 0.0
    overlap = min(end_s, other_end) - max(start_s, other_start)
    if overlap <= 0:
        return 0.0
    return overlap / duration


def _best_support_ratio(
    start_s: float,
    end_s: float,
    ranges: Sequence[tuple[float, float]],
) -> float:
    best = 0.0
    for other_start, other_end in ranges:
        best = max(best, _interval_overlap_ratio(start_s, end_s, other_start, other_end))
    return best


def _join_word_texts(words: Sequence[WordTiming]) -> str:
    parts: list[str] = []
    for word in words:
        token = word.text.strip()
        if not token:
            continue
        if not parts:
            parts.append(token)
            continue
        if parts[-1][-1] in "([{'\"«" or token[0] in ".,;:!?)]}%»":
            parts[-1] = f"{parts[-1]}{token}"
        else:
            parts.append(token)
    return " ".join(parts)


def _is_non_lexical_token(text: str) -> bool:
    """Vrai si le token n'a aucun caractère alphabétique ou numérique."""
    return _LEXICAL_CHAR.search(text.strip()) is None


def _is_oversized_non_lexical(
    word: WordTiming,
    *,
    max_duration_s: float,
) -> bool:
    """COMMON-019 : ponctuation isolée avec durée anormalement longue."""
    if max_duration_s < 0:
        return False
    duration = word.end_s - word.start_s
    if duration <= max_duration_s:
        return False
    return _is_non_lexical_token(word.text)


def _utterance_is_exploitable(utterance: AlignedUtterance) -> bool:
    """Texte lexical exploitable (pas seulement de la ponctuation)."""
    return bool(utterance.text.strip()) and not _is_non_lexical_token(utterance.text)


def _subtract_intervals(
    start_s: float,
    end_s: float,
    covered: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Retourne les sous-intervalles de [start, end] non couverts."""
    remaining = [(start_s, end_s)]
    for cov_start, cov_end in covered:
        next_remaining: list[tuple[float, float]] = []
        for rem_start, rem_end in remaining:
            if cov_end <= rem_start or cov_start >= rem_end:
                next_remaining.append((rem_start, rem_end))
                continue
            if cov_start > rem_start:
                next_remaining.append((rem_start, cov_start))
            if cov_end < rem_end:
                next_remaining.append((cov_end, rem_end))
        remaining = next_remaining
    return [(a, b) for a, b in remaining if b - a > 0]


def _untranscribed_speech_notes(
    diarization: DiarizationResult,
    utterances: Sequence[AlignedUtterance],
    *,
    min_duration_s: float,
) -> tuple[str, ...]:
    """COMMON-020 : parole Community-1 non couverte par un texte lexical."""
    covered = tuple(
        (item.start_s, item.end_s)
        for item in utterances
        if _utterance_is_exploitable(item)
    )
    notes: list[str] = []
    for turn in diarization.turns:
        if turn.speaker_id is None or turn.end_s <= turn.start_s:
            continue
        for gap_start, gap_end in _subtract_intervals(turn.start_s, turn.end_s, covered):
            if gap_end - gap_start < min_duration_s:
                continue
            notes.append(
                f"parole Community-1 non transcrite "
                f"({gap_start:.2f}-{gap_end:.2f}s) : {turn.speaker_id}"
            )
    return tuple(dict.fromkeys(notes))


def _filter_transcripts_temporally(
    acquisition: AcquisitionSnapshot,
    transcripts: Sequence[TranscriptSegment],
    diarization: DiarizationResult,
    *,
    min_overlap_ratio: float,
    max_non_lexical_duration_s: float = DEFAULT_MAX_NON_LEXICAL_DURATION_S,
) -> tuple[tuple[TranscriptSegment, ...], tuple[str, ...], bool]:
    """Écarte les textes sans support acoustique local fiable (COMMON-017/019)."""
    speech_ranges = tuple(
        (item.start_s, item.end_s) for item in acquisition.speech_intervals
    )
    turn_ranges = tuple(
        (turn.start_s, turn.end_s)
        for turn in diarization.turns
        if turn.speaker_id is not None
    )
    kept: list[TranscriptSegment] = []
    notes: list[str] = []
    rejected = False

    for segment in transcripts:
        if segment.status != "completed" or not segment.text.strip():
            kept.append(segment)
            continue

        units: list[WordTiming]
        if segment.words:
            units = list(segment.words)
        else:
            units = [WordTiming(segment.start_s, segment.end_s, segment.text)]

        kept_words: list[WordTiming] = []
        for word in units:
            if not word.text.strip():
                continue
            # COMMON-019 : indépendant du VAD — ponctuation isolée trop longue.
            if _is_oversized_non_lexical(
                word, max_duration_s=max_non_lexical_duration_s
            ):
                rejected = True
                notes.append(
                    f"token non lexical trop long écarté "
                    f"({word.start_s:.2f}-{word.end_s:.2f}s) : {word.text.strip()}"
                )
                continue
            silero_ratio = _best_support_ratio(word.start_s, word.end_s, speech_ranges)
            diar_ratio = _best_support_ratio(word.start_s, word.end_s, turn_ranges)
            if silero_ratio >= min_overlap_ratio or diar_ratio >= min_overlap_ratio:
                kept_words.append(word)
                continue
            # Sans support local : n'écarter que si le silence local est fiable.
            local_silence_reliable = (
                acquisition.evidence_available
                and acquisition.signal_usable is True
                and silero_ratio <= 0.0
                and diar_ratio <= 0.0
            )
            if local_silence_reliable:
                rejected = True
                notes.append(
                    f"texte suspect écarté ({word.start_s:.2f}-{word.end_s:.2f}s) : "
                    f"{word.text.strip()}"
                )
                continue
            # Support insuffisant ou inconnu : conserver comme incertain.
            kept_words.append(word)
            notes.append(
                f"texte incertain conservé ({word.start_s:.2f}-{word.end_s:.2f}s) : "
                f"{word.text.strip()}"
            )

        if not kept_words:
            rejected = True
            continue
        if segment.words and len(kept_words) == len(segment.words):
            kept.append(segment)
            continue
        rebuilt = TranscriptSegment(
            session_id=segment.session_id,
            source_segment_id=segment.source_segment_id,
            start_s=kept_words[0].start_s,
            end_s=kept_words[-1].end_s,
            text=_join_word_texts(kept_words) if segment.words else kept_words[0].text,
            language=segment.language,
            status="completed",
            words=tuple(kept_words) if segment.words else (),
        )
        rebuilt.validate()
        kept.append(rebuilt)
        if len(kept_words) < len(units):
            rejected = True

    return tuple(kept), tuple(dict.fromkeys(notes)), rejected


def _coherence_decision(
    acquisition: AcquisitionSnapshot,
    transcripts: Sequence[TranscriptSegment],
    diarization: DiarizationResult,
) -> _CoherenceDecision:
    """Politique de cohérence acoustique avant alignement."""
    asr_empty = _asr_empty(transcripts)
    diar_speech = _diarization_has_speech(diarization)
    items = tuple(transcripts)

    if asr_empty and diar_speech:
        return _CoherenceDecision(
            outcome="failed",
            error=(
                "incohérence des modèles : parole détectée par la "
                "diarisation sans transcription exploitable"
            ),
        )

    if asr_empty and not diar_speech:
        if acquisition.silero_speech:
            return _CoherenceDecision(
                outcome="failed",
                error=(
                    "incohérence des modèles : Silero a détecté de la parole "
                    "sans transcription ni locuteur exploitable"
                ),
                diagnostics=(
                    "parole Silero conservée comme indice, non transformée en silence",
                ),
            )
        if not acquisition.silence_confirmed:
            return _CoherenceDecision(
                outcome="failed",
                error="Silence non vérifié : preuve acoustique insuffisante",
            )
        return _CoherenceDecision(outcome="align", transcripts=items)

    # ASR a du texte.
    if diar_speech:
        return _CoherenceDecision(outcome="align", transcripts=items)

    # Texte ASR sans locuteur Community-1.
    if not acquisition.evidence_available:
        rejected = _asr_text_summary(items)
        return _CoherenceDecision(
            outcome="failed",
            error=(
                "contradiction ASR/diarisation : preuve acoustique absente "
                "(Silero/qualité non transmises)"
            ),
            diagnostics=(f"texte ASR non vérifié : {rejected}",) if rejected else (),
        )

    if acquisition.silence_confirmed:
        rejected = _asr_text_summary(items)
        note = (
            f"hallucination ASR écartée (silence confirmé Silero+qualité) : {rejected}"
            if rejected
            else "hallucination ASR écartée (silence confirmé)"
        )
        return _CoherenceDecision(outcome="empty", diagnostics=(note,))

    if acquisition.silero_speech:
        # Parole faible possible : Community-1 a manqué, Silero a entendu.
        # On conserve le texte ; l'alignement le marquera unknown.
        return _CoherenceDecision(
            outcome="align",
            transcripts=items,
            diagnostics=(
                "texte ASR conservé : Silero a détecté de la parole "
                "sans tour Community-1",
            ),
        )

    return _CoherenceDecision(
        outcome="failed",
        error=(
            "incohérence des modèles : transcription sans support "
            "acoustique confirmé"
        ),
        diagnostics=(f"texte ASR : {_asr_text_summary(items)}",),
    )


@dataclass
class SoundFileAcquisition:
    """Acquisition minimale : durée du WAV, sans preuve Silero."""

    def run_file(self, path: Path, *, session_id: str) -> AcquisitionSnapshot:
        import soundfile as sf

        info = sf.info(str(path))
        duration_s = float(info.duration)
        if duration_s < 0:
            return AcquisitionSnapshot(
                session_id=session_id,
                duration_s=0.0,
                source=str(path),
                status="failed",
                error="durée audio invalide",
            )
        return AcquisitionSnapshot(
            session_id=session_id,
            duration_s=duration_s,
            source=str(path),
            status="completed",
            evidence_available=False,
        )


@dataclass
class LocalAcquisition:
    """Passe par ``AcquisitionService`` avec le ``session_id`` commun."""

    def run_file(self, path: Path, *, session_id: str) -> AcquisitionSnapshot:
        from safeai.acquisition.contracts import (
            QualityStatus,
            SpeechActivity,
            StreamStatus,
        )
        from safeai.acquisition.service import run_file

        try:
            result = run_file(path, session_id=session_id)
        except Exception as exc:
            return AcquisitionSnapshot(
                session_id=session_id,
                duration_s=0.0,
                source=str(path),
                status="failed",
                error=str(exc),
            )
        duration_s = result.chunks[-1].end_s if result.chunks else 0.0
        if result.session.session_id != session_id:
            return AcquisitionSnapshot(
                session_id=session_id,
                duration_s=duration_s,
                source=str(path),
                status="failed",
                error=(
                    "session incompatible : acquisition interne "
                    f"{result.session.session_id} ≠ {session_id}"
                ),
            )
        mismatched = [
            segment.session_id
            for segment in result.speech_segments
            if segment.session_id != session_id
        ]
        if mismatched:
            return AcquisitionSnapshot(
                session_id=session_id,
                duration_s=duration_s,
                source=str(path),
                status="failed",
                error="session incompatible : segments d'acquisition",
            )
        if result.quality_events:
            last = result.quality_events[-1].stream_status
            if last == StreamStatus.UNAVAILABLE:
                return AcquisitionSnapshot(
                    session_id=session_id,
                    duration_s=duration_s,
                    source=str(path),
                    status="failed",
                    error="flux audio indisponible",
                    evidence_available=True,
                    signal_usable=False,
                    incidents=tuple(result.incidents),
                )

        intervals = tuple(
            SpeechInterval(
                start_s=segment.start_s,
                end_s=segment.end_s,
                vad_mean_probability=segment.vad_mean_probability,
            )
            for segment in result.speech_segments
            if segment.speech_detected and segment.end_s > segment.start_s
        )
        signal_usable = True
        silence_like = len(intervals) == 0
        if result.quality_events:
            if any(
                event.stream_status == StreamStatus.UNAVAILABLE
                or event.quality_status == QualityStatus.UNUSABLE
                for event in result.quality_events
            ):
                signal_usable = False
            if any(
                event.speech_activity == SpeechActivity.SPEECH for event in result.quality_events
            ):
                # Activité signalée sans segment consolidé : pas un silence confirmé.
                if not intervals:
                    silence_like = False
            elif any(
                event.reason == "silence" or event.speech_activity == SpeechActivity.NON_SPEECH
                for event in result.quality_events
            ):
                silence_like = silence_like and True

        return AcquisitionSnapshot(
            session_id=session_id,
            duration_s=duration_s,
            source=str(path),
            status="completed",
            evidence_available=True,
            speech_intervals=intervals,
            signal_usable=signal_usable,
            silence_like=silence_like,
            incidents=tuple(result.incidents),
        )


@dataclass
class WhisperFileTranscriber:
    """Adaptateur autour de ``WhisperTranscriber.transcribe_file``."""

    transcriber: object

    def transcribe_file(
        self,
        path: Path,
        *,
        session_id: str,
        duration_s: float | None = None,
    ) -> Sequence[TranscriptSegment]:
        return self.transcriber.transcribe_file(
            path,
            session_id=session_id,
            source_segment_id="file",
            offset_s=0.0,
            duration_s=duration_s,
        )


@dataclass
class DiarizationServiceAdapter:
    """Adaptateur autour de ``DiarizationService.run_file``."""

    service: object

    def run_file(self, path: Path, *, session_id: str) -> DiarizationResult:
        return self.service.run_file(path, session_id=session_id)


def wav_duration_s(path: Path) -> float:
    """Durée du fichier à 16 kHz mono après lecture légère des métadonnées."""
    import soundfile as sf

    info = sf.info(str(path))
    if info.samplerate == TARGET_SAMPLE_RATE:
        return float(info.duration)
    return float(info.duration)


def _asr_empty(transcripts: Sequence[TranscriptSegment]) -> bool:
    return not any(
        item.status == "completed" and item.text.strip() for item in transcripts
    )


def _diarization_has_speech(diarization: DiarizationResult) -> bool:
    if diarization.status == "failed":
        return False
    if diarization.speakers_estimated > 0:
        return True
    return any(turn.speaker_id is not None for turn in diarization.turns)


def _asr_text_summary(transcripts: Sequence[TranscriptSegment]) -> str:
    parts = [
        item.text.strip()
        for item in transcripts
        if item.status == "completed" and item.text.strip()
    ]
    return " | ".join(parts)


def _failed(
    *,
    session_id: str,
    duration_s: float,
    source: str,
    provenance: str,
    error: str,
    diagnostics: tuple[str, ...] = (),
) -> AlignedSession:
    session = AlignedSession(
        session_id=session_id,
        duration_s=duration_s,
        utterances=(),
        offset_s=0.0,
        language="fr",
        provenance=provenance,
        source=source,
        status="failed",
        error=error,
        diagnostics=diagnostics,
    )
    session.validate()
    return session
