"""Une phrase qui traverse le recouvrement audio n'est publiée qu'une fois."""

import queue

import numpy as np

from safeai.acquisition.contracts import SpeechSegment
from safeai.transcription.contracts import TranscriptSegment, WordTiming
from safeai.transcription.worker import WhisperWorker


def _speech(start: float, end: float) -> SpeechSegment:
    count = int(round((end - start) * 16000))
    return SpeechSegment(
        session_id="s",
        start_s=start,
        end_s=end,
        speech_detected=True,
        vad_mean_probability=0.9,
        audio_available=True,
        samples=np.zeros(count, dtype=np.float32),
    )


class BoundaryTranscriber:
    def transcribe_speech(self, segment: SpeechSegment, source_segment_id: str) -> list[TranscriptSegment]:
        if segment.start_s < 1.0:
            words = (
                WordTiming(6.20, 6.50, "le"),
                WordTiming(6.55, 7.10, "train"),
                WordTiming(7.20, 7.80, "arrive"),
            )
        else:
            words = (
                WordTiming(7.20, 7.80, "arrive"),
                WordTiming(8.10, 8.30, "à"),
                WordTiming(8.40, 9.10, "l'heure"),
            )
        return [
            TranscriptSegment(
                session_id=segment.session_id,
                source_segment_id=source_segment_id,
                start_s=words[0].start_s,
                end_s=words[-1].end_s,
                text=" ".join(word.text for word in words),
                language="fr",
                status="completed",
                words=words,
            )
        ]


def test_sentence_crossing_the_overlap_is_published_once() -> None:
    worker = WhisperWorker(BoundaryTranscriber(), queue.Queue(), clock=lambda: 0.0)
    worker._transcribe(_speech(0.0, 8.0))
    worker._transcribe(_speech(7.0, 15.0))
    text = " ".join(item.segment.text for item in worker.published)
    assert text == "le train arrive à l'heure"
    assert text.count("arrive") == 1


def test_real_repetition_inside_one_segment_is_kept() -> None:
    class Repeat:
        def transcribe_speech(self, segment, source_segment_id):
            words = (WordTiming(1.00, 1.12, "Non"), WordTiming(1.20, 1.35, "non"))
            return [
                TranscriptSegment(
                    session_id=segment.session_id,
                    source_segment_id=source_segment_id,
                    start_s=1.0,
                    end_s=1.5,
                    text="Non non",
                    language="fr",
                    status="completed",
                    words=words,
                )
            ]

    worker = WhisperWorker(Repeat(), queue.Queue(), clock=lambda: 0.0)
    worker._transcribe(_speech(0.0, 2.0))
    assert [word.text.lower() for word in worker.published[-1].segment.words] == ["non", "non"]
