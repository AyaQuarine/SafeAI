"""Erreurs contrôlées du module de diarisation."""


class DiarizationError(Exception):
    """Erreur de base du module de diarisation."""


class ConfigurationError(DiarizationError):
    """Configuration invalide."""


class ModelUnavailableError(DiarizationError):
    """Modèle absent, non autorisé, ou téléchargement impossible."""


class ModelInferenceError(DiarizationError):
    """Le modèle est chargé, mais l'inférence a échoué."""


class InvalidDiarizationError(DiarizationError):
    """Résultat de diarisation invalide ou fichier audio non exploitable."""
