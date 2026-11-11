"""Contexte glissant : les régions commises se suivent, le texte n'est pas publié deux fois."""

from safeai.transcription.contracts import TranscriptSegment, WordTiming
from safeai.transcription.windows import commit_transcripts, pause_windows, sliding_windows


def _segment(start: float, end: float, text: str, words: list[tuple[float, float, str]]) -> TranscriptSegment:
    return TranscriptSegment(
        session_id="s",
        source_segment_id="w",
        start_s=start,
        end_s=end,
        text=text,
        language="fr",
        status="completed",
        words=tuple(WordTiming(a, b, word) for a, b, word in words),
    )


def test_sliding_commit_regions_tile_the_recording() -> None:
    windows = sliding_windows(30.0, window_s=10.0, overlap_s=2.0)
    assert windows[0].start_s == 0.0
    assert windows[0].end_s == 10.0
    assert windows[0].commit_end_s == 8.0
    assert windows[1].start_s == 8.0
    cursor = 0.0
    for window in windows:
        assert window.commit_start_s == cursor
        assert window.end_s > window.commit_end_s or window is windows[-1]
        cursor = window.commit_end_s
    assert cursor == 30.0
    assert windows[-1].commit_end_s == 30.0


def test_short_recording_is_one_window() -> None:
    windows = sliding_windows(5.0, window_s=12.0, overlap_s=2.0)
    assert len(windows) == 1
    assert windows[0].end_s == 5.0
    assert windows[0].commit_end_s == 5.0


def test_overlap_word_is_published_once() -> None:
    first = commit_transcripts(
        [_segment(7.5, 9.5, "alpha beta", [(7.6, 8.0, "alpha"), (8.4, 9.2, "beta")])],
        0.0,
        8.0,
    )
    second = commit_transcripts(
        [_segment(8.2, 11.0, "beta gamma", [(8.4, 9.2, "beta"), (9.5, 10.4, "gamma")])],
        8.0,
        16.0,
    )
    words = [word.text for segment in first + second for word in segment.words]
    assert words == ["alpha", "beta", "gamma"]


def test_rejoined_apostrophe_stays_one_token() -> None:
    kept = commit_transcripts(
        [_segment(1.0, 2.0, "qu'on veut", [(1.0, 1.2, "qu"), (1.2, 1.5, "'on"), (1.5, 1.9, "veut")])],
        0.0,
        1.8,
    )
    assert kept[0].text == "qu'on veut"


def test_pause_grouping_splits_on_long_silence_and_on_length() -> None:
    regions = [(0.0, 3.0), (3.4, 6.0), (9.0, 11.0), (11.2, 22.0)]
    windows = pause_windows(regions, max_window_s=12.0, max_pause_s=0.8)
    assert [(window.start_s, window.end_s) for window in windows] == [
        (0.0, 6.0),
        (9.0, 11.0),
        (11.2, 22.0),
    ]


def test_same_acoustic_interval_from_another_window_is_published_once() -> None:
    from safeai.transcription.windows import _suppress_repeated_tokens

    first = TranscriptSegment(
        session_id="s",
        source_segment_id="window_001",
        start_s=8.4,
        end_s=9.2,
        text="beta",
        language="fr",
        status="completed",
        words=(WordTiming(8.4, 9.2, "beta"),),
    )
    second = TranscriptSegment(
        session_id="s",
        source_segment_id="window_002",
        start_s=8.45,
        end_s=10.2,
        text="beta gamma",
        language="fr",
        status="completed",
        words=(WordTiming(8.45, 9.1, "beta"), WordTiming(9.5, 10.2, "gamma")),
    )
    kept = _suppress_repeated_tokens([first, second])
    words = [word.text for segment in kept for word in segment.words]
    assert words == ["beta", "gamma"]
