"""Une parole continue ne doit pas retenir tout le signal dans le tampon."""

import numpy as np

from safeai.acquisition.config import VADSettings
from safeai.acquisition.contracts import TARGET_SAMPLE_RATE, AudioChunk, VADFrame
from safeai.acquisition.segmenter import AudioSegmenter


def _speech_chunk(index: int) -> tuple[AudioChunk, list[VADFrame]]:
    start = float(index)
    samples = np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32)
    chunk = AudioChunk(
        session_id="long",
        source_type="synthetic",
        start_s=start,
        end_s=start + 1.0,
        sample_rate=TARGET_SAMPLE_RATE,
        samples=samples,
    )
    step = 512 / TARGET_SAMPLE_RATE
    frames = []
    cursor = start
    while cursor < start + 1.0 - 1e-9:
        frame_end = min(start + 1.0, cursor + step)
        frames.append(VADFrame(cursor, frame_end, 0.99))
        cursor = frame_end
    return chunk, frames


def test_continuous_speech_is_split_before_the_buffer_holds_everything() -> None:
    segmenter = AudioSegmenter(
        VADSettings(
            backend="energy",
            max_speech_segment_s=8.0,
            segment_overlap_s=1.0,
            min_speech_duration_s=0.2,
            speech_pad_s=0.0,
        )
    )
    closed = []
    for index in range(20):
        chunk, frames = _speech_chunk(index)
        closed.extend(segmenter.ingest(chunk, frames).segments)
    buffered = sum(len(samples) for _start, samples in segmenter._buffer.pieces)
    assert closed
    assert buffered <= 8 * TARGET_SAMPLE_RATE
    assert all(segment.end_s - segment.start_s <= 8.0 + 1e-3 for segment in closed)
    tail = segmenter.flush(20.0)
    ends = [segment.end_s for segment in closed + tail]
    assert max(ends) == 20.0
    for segment in closed + tail:
        segment.validate()
