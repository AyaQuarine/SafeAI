"""Normalisation et WER. La référence doit être une transcription corrigée, pas une hypothèse."""

from __future__ import annotations

import re
import unicodedata


def normalize_french(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).lower().replace("_", " ").replace("+", " ")
    folded = folded.replace("’", "'")
    folded = re.sub(r"[^\w\s']", " ", folded, flags=re.UNICODE)
    return re.sub(r"\s+", " ", folded).strip()


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    counts = alignment_counts(reference, hypothesis)
    if counts is None:
        return None
    return counts["wer"]


def alignment_counts(reference: str, hypothesis: str) -> dict[str, object] | None:
    """Compte les omissions sur la référence normalisée. None si elle est vide."""
    ref = normalize_french(reference)
    hyp = normalize_french(hypothesis)
    if not ref:
        return None
    if not hyp:
        words = ref.split()
        return {
            "wer": 1.0,
            "hits": 0,
            "substitutions": 0,
            "deletions": len(words),
            "insertions": 0,
            "missed_words": words,
        }
    from jiwer import process_words

    output = process_words(ref, hyp)
    missed: list[str] = []
    for chunk in output.alignments[0]:
        if chunk.type == "delete":
            missed.extend(output.references[0][chunk.ref_start_idx : chunk.ref_end_idx])
    total = output.hits + output.substitutions + output.deletions
    wer = float(output.wer) if total else 0.0
    return {
        "wer": wer,
        "hits": output.hits,
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "missed_words": missed,
    }
