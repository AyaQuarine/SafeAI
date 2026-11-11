"""Orchestration hors ligne d'un WAV vers DiarizationResult.

Le service n'ouvre pas le modèle lui-même : il délègue à l'adaptateur,
conserve ``session_id`` et ``offset_s``, et laisse les erreurs de fichier,
de chargement ou d'inférence distinctes.
"""

from __future__ import annotations

from pathlib import Path

from safeai.acquisition.contracts import new_session_id
from safeai.diarization.adapters import PyannoteAdapter
from safeai.diarization.config import DiarizationConfig, load_config
from safeai.diarization.contracts import DiarizationResult


class DiarizationService:
    def __init__(
        self,
        config: DiarizationConfig | None = None,
        adapter: PyannoteAdapter | None = None,
    ) -> None:
        self.config = config or load_config()
        self.adapter = adapter or PyannoteAdapter(self.config)

    def run_file(self, path: str | Path, *, session_id: str | None = None) -> DiarizationResult:
        return self.adapter.diarize(
            path,
            session_id=session_id or new_session_id(),
            offset_s=self.config.offset_s,
        )
