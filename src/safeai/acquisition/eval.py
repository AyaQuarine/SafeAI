"""Métriques d'évaluation VAD — frames de 32 ms, annotations manuelles."""

from __future__ import annotations

from dataclasses import dataclass

from safeai.acquisition.contracts import TARGET_SAMPLE_RATE, SpeechSegment


@dataclass(frozen=True)
class Interval:
    start_s: float
    end_s: float


@dataclass(frozen=True)
class FrameMetrics:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    frames: int


def _covers(intervals: list[Interval], t: float) -> bool:
    return any(item.start_s <= t < item.end_s for item in intervals)


def frame_metrics(
    gold: list[Interval],
    predicted: list[Interval],
    duration_s: float,
    frame_s: float = 512 / TARGET_SAMPLE_RATE,
) -> FrameMetrics:
    if duration_s <= 0 or frame_s <= 0:
        raise ValueError("duration_s et frame_s doivent être > 0")
    n = max(1, int(round(duration_s / frame_s)))
    tp = fp = fn = 0
    for index in range(n):
        t = index * frame_s
        truth = _covers(gold, t)
        pred = _covers(predicted, t)
        if truth and pred:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif truth and not pred:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return FrameMetrics(precision, recall, f1, tp, fp, fn, n)


def segments_to_intervals(segments: list[SpeechSegment]) -> list[Interval]:
    return [Interval(segment.start_s, segment.end_s) for segment in segments]
