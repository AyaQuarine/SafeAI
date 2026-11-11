"""Alignement voix–texte — contrat commun et moteur, sans identité personnelle.

Les analyses textuelle et vocale consomment ``AlignedUtterance``.
``align_session`` relie Whisper et Community-1 sans charger de modèle.
"""

from safeai.alignment.contracts import (
    AlignedSession,
    AlignedUtterance,
    load_aligned_session,
    session_from_dict,
)
from safeai.alignment.engine import AlignmentConfig, align_session
from safeai.alignment.errors import AlignmentError, InvalidAlignmentError

__all__ = [
    "AlignedSession",
    "AlignedUtterance",
    "AlignmentConfig",
    "AlignmentError",
    "InvalidAlignmentError",
    "align_session",
    "load_aligned_session",
    "session_from_dict",
]
