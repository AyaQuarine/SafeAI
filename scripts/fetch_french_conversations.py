"""Télécharge des extraits de conversations françaises ouvertes (CC BY-SA).

Sources prioritaires MVP :
- SUMM-RE (LINAGORA / Aix-Marseille) : réunions 3–4 locuteurs
- Accueil_UBS (Université de Bretagne Sud) : dialogues téléphoniques 2 locuteurs

Aucune donnée anglaise n'est téléchargée. Les WAV restent hors Git.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "conversations_fr"
AUDIO_DIR = OUT_DIR / "audio"
MANIFEST_PATH = OUT_DIR / "manifest.json"
TARGET_SR = 16_000


def _decode_audio(audio) -> tuple[np.ndarray, int]:
    if isinstance(audio, dict) and "array" in audio:
        return np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"])
    payload = audio
    if isinstance(audio, dict):
        payload = audio.get("bytes") or audio.get("path")
    if isinstance(payload, (bytes, bytearray)):
        samples, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
        return np.asarray(samples, dtype=np.float32), int(rate)
    if isinstance(payload, str) and Path(payload).is_file():
        samples, rate = sf.read(payload, dtype="float32", always_2d=False)
        return np.asarray(samples, dtype=np.float32), int(rate)
    raise TypeError(f"Format audio non reconnu : {type(audio)}")


def _resample(samples: np.ndarray, source_rate: int) -> np.ndarray:
    from safeai.acquisition.normalizer import AudioNormalizer

    return AudioNormalizer(TARGET_SR).convert(samples, source_rate)


def _write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.ascontiguousarray(np.clip(samples, -1.0, 1.0), dtype=np.float32)
    sf.write(str(path), audio, TARGET_SR, subtype="PCM_16")


def _clip(samples: np.ndarray, start_s: float, duration_s: float) -> np.ndarray:
    start = int(round(start_s * TARGET_SR))
    end = int(round((start_s + duration_s) * TARGET_SR))
    return samples[start:end]


def _shift_intervals(intervals: list[dict], offset_s: float, duration_s: float) -> list[dict]:
    shifted = []
    for item in intervals:
        start = max(0.0, item["start_s"] - offset_s)
        end = min(duration_s, item["end_s"] - offset_s)
        if end - start < 0.05:
            continue
        shifted.append(
            {
                "start_s": round(start, 3),
                "end_s": round(end, 3),
                "speaker": item.get("speaker", "unknown"),
                "text": item.get("text", ""),
            }
        )
    return shifted


def _union_speech(intervals: list[dict]) -> list[dict]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: item["start_s"])
    merged = [dict(ordered[0])]
    for item in ordered[1:]:
        if item["start_s"] <= merged[-1]["end_s"] + 0.15:
            merged[-1]["end_s"] = max(merged[-1]["end_s"], item["end_s"])
            if item.get("text"):
                merged[-1]["text"] = (merged[-1].get("text", "") + " " + item["text"]).strip()
        else:
            merged.append(dict(item))
    return merged


def _load_summre_rows():
    cache = OUT_DIR / "_cache" / "summre_example.parquet"
    if cache.is_file() and cache.stat().st_size > 1_000_000:
        print(f"Lecture du cache local {cache}...")
        import pyarrow.parquet as pq

        table = pq.read_table(cache)
        return table.to_pylist()

    from datasets import load_dataset

    print("Telechargement SUMM-RE split example (4 pistes, ~100 Mo)...")
    dataset = load_dataset("linagora/SUMM-RE", "example")
    split_name = "example" if "example" in dataset else next(iter(dataset.keys()))
    return list(dataset[split_name])


def fetch_summre_example() -> list[dict]:
    rows = _load_summre_rows()
    if len(rows) < 2:
        raise RuntimeError("SUMM-RE example inattendu : moins de 2 pistes.")

    meeting_id = str(rows[0].get("meeting_id") or "summre_example")
    tracks = []
    all_intervals: list[dict] = []
    max_len = 0
    for row in rows:
        samples, rate = _decode_audio(row["audio"])
        samples = _resample(samples, rate)
        max_len = max(max_len, len(samples))
        speaker = str(row.get("speaker_id") or "spk")
        intervals = []
        for segment in row.get("segments") or []:
            intervals.append(
                {
                    "start_s": float(segment["start"]),
                    "end_s": float(segment["end"]),
                    "speaker": speaker,
                    "text": str(segment.get("transcript") or ""),
                }
            )
        all_intervals.extend(intervals)
        tracks.append({"speaker": speaker, "samples": samples, "intervals": intervals})

    mixed = np.zeros(max_len, dtype=np.float64)
    for track in tracks:
        mixed[: len(track["samples"])] += track["samples"]
    mixed = np.ascontiguousarray(mixed / max(len(tracks), 1), dtype=np.float32)
    peak = float(np.max(np.abs(mixed))) or 1.0
    mixed = mixed / peak * 0.7

    entries = []
    # Reunion mixee : extraits utiles pour VAD / multi-locuteurs.
    excerpts = [
        ("fr_reunion_mix_00_30s.wav", 0.0, 30.0, "reunion_mix", "Debut de reunion, mix 4 micros"),
        ("fr_reunion_mix_30_60s.wav", 30.0, 30.0, "reunion_mix", "Suite de reunion, tours de parole"),
        ("fr_reunion_mix_90_45s.wav", 90.0, 45.0, "reunion_mix", "Extrait plus long, chevauchements possibles"),
    ]
    for filename, start_s, duration_s, kind, description in excerpts:
        if start_s * TARGET_SR >= len(mixed):
            continue
        clip = _clip(mixed, start_s, duration_s)
        if clip.size < TARGET_SR:
            continue
        path = AUDIO_DIR / filename
        _write_wav(path, clip)
        gold = _shift_intervals(all_intervals, start_s, len(clip) / TARGET_SR)
        entries.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "source": "linagora/SUMM-RE",
                "license": "CC BY-SA 4.0",
                "language": "fr",
                "kind": kind,
                "speakers": len(tracks),
                "duration_s": round(len(clip) / TARGET_SR, 3),
                "meeting_id": meeting_id,
                "description": description,
                "gold_speech": _union_speech(gold),
                "gold_turns": gold,
            }
        )

    # Une piste close-talk pour comparer (un locuteur dominant).
    first = tracks[0]
    close = _clip(first["samples"], 0.0, 30.0)
    close_path = AUDIO_DIR / "fr_reunion_closetalk_00_30s.wav"
    _write_wav(close_path, close)
    gold = _shift_intervals(first["intervals"], 0.0, len(close) / TARGET_SR)
    entries.append(
        {
            "path": str(close_path.relative_to(ROOT)).replace("\\", "/"),
            "source": "linagora/SUMM-RE",
            "license": "CC BY-SA 4.0",
            "language": "fr",
            "kind": "reunion_closetalk",
            "speakers": 1,
            "duration_s": round(len(close) / TARGET_SR, 3),
            "meeting_id": meeting_id,
            "description": "Piste individuelle (micro porte). Les silences correspondent souvent aux tours des autres.",
            "gold_speech": _union_speech(gold),
            "gold_turns": gold,
        }
    )
    return entries


UBS_CLIPS = [
    {
        "filename": "017_00000012.wav",
        "url": "https://huggingface.co/datasets/amorin-gladia/Accueil_UBS_long_form/resolve/main/data/test/017_00000012.wav",
        "turns": [
            {"start_s": 0.0, "end_s": 2.122, "speaker": "spk1", "text": "U B S bonjour"},
            {"start_s": 2.122, "end_s": 8.016, "speaker": "spk2", "text": "oui bonjour e j'appelle je sais pas si j'appelle au bon endroit e"},
            {"start_s": 8.016, "end_s": 8.742, "speaker": "spk1", "text": "je vous ecoute"},
            {"start_s": 8.742, "end_s": 20.39, "speaker": "spk2", "text": "e c'est pour savoir si la fac pendant l'ete"},
        ],
    },
    {
        "filename": "018_00000013.wav",
        "url": "https://huggingface.co/datasets/amorin-gladia/Accueil_UBS_long_form/resolve/main/data/test/018_00000013.wav",
        "turns": [
            {"start_s": 0.0, "end_s": 1.614, "speaker": "spk1", "text": "U B S bonjour"},
            {"start_s": 1.614, "end_s": 9.094, "speaker": "spk2", "text": "oui bonjour ce serait pour savoir comment obtenir un dossier de candidature"},
            {"start_s": 9.094, "end_s": 11.421, "speaker": "spk1", "text": "e oui je vais vous passer le secretariat"},
        ],
    },
    {
        "filename": "019_00000014.wav",
        "url": "https://huggingface.co/datasets/amorin-gladia/Accueil_UBS_long_form/resolve/main/data/test/019_00000014.wav",
        "turns": [
            {"start_s": 0.0, "end_s": 1.719, "speaker": "spk1", "text": "U B S bonjour"},
            {"start_s": 1.719, "end_s": 5.976, "speaker": "spk2", "text": "oui bonjour e j'ai appele ce matin pour avoir monsieur Nom"},
            {"start_s": 5.976, "end_s": 13.021, "speaker": "spk1", "text": "oui vous etes madame c'est pour une connexion Internet"},
        ],
    },
]


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 10_000:
        return
    from urllib.request import urlretrieve

    print(f"  telechargement {dest.name}...")
    urlretrieve(url, dest)


def fetch_accueil_ubs(limit: int = 3) -> list[dict]:
    print("Telechargement Accueil_UBS (dialogues telephoniques FR, WAV directs)...")
    entries = []
    for index, clip in enumerate(UBS_CLIPS[:limit]):
        cache = OUT_DIR / "_cache" / clip["filename"]
        _download_file(clip["url"], cache)
        samples, rate = sf.read(str(cache), dtype="float32", always_2d=False)
        samples = _resample(np.asarray(samples, dtype=np.float32), int(rate))
        duration = len(samples) / TARGET_SR
        if duration < 8.0:
            continue
        clip_duration = min(40.0, duration)
        audio = _clip(samples, 0.0, clip_duration)
        filename = f"fr_accueil_ubs_{index + 1:02d}_{int(clip_duration)}s.wav"
        path = AUDIO_DIR / filename
        _write_wav(path, audio)
        gold = _shift_intervals(clip["turns"], 0.0, clip_duration)
        n_speakers = len({turn["speaker"] for turn in gold}) or 2
        entries.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "source": "amorin-gladia/Accueil_UBS_long_form",
                "license": "CC BY-SA",
                "language": "fr",
                "kind": "dialogue_telephone",
                "speakers": n_speakers,
                "duration_s": round(len(audio) / TARGET_SR, 3),
                "description": "Dialogue reel standard universitaire, 2 locuteurs, tours annotes.",
                "gold_speech": _union_speech(gold),
                "gold_turns": gold,
            }
        )
        print(f"  extrait {filename} ({clip_duration:.1f}s, {n_speakers} locuteurs)")
    if not entries:
        raise RuntimeError("Aucun dialogue Accueil_UBS exploitable.")
    return entries


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    entries.extend(fetch_summre_example())
    entries.extend(fetch_accueil_ubs(limit=3))
    manifest = {
        "purpose": "Jeux de test MVP : conversations et reunions francaises multi-locuteurs.",
        "priority": "fr",
        "excluded": "Aucun corpus anglais n'est inclus.",
        "note": "Les WAV sont locaux et ignores par Git. Relancer ce script pour les recreer.",
        "files": entries,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(entries)} fichiers ecrits dans {AUDIO_DIR}")
    print(f"Manifeste : {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
