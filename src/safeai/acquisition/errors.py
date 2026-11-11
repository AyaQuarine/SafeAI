"""Erreurs contrôlées du module d'acquisition."""


class AcquisitionError(Exception):
    """Erreur de base du module d'acquisition."""


class ConsentRequiredError(AcquisitionError):
    """Capture refusée : le consentement d'enregistrement n'est pas fourni."""


class InvalidAudioError(AcquisitionError):
    """Fichier audio vide, corrompu ou non exploitable."""


class SourceUnavailableError(AcquisitionError):
    """Source audio indisponible ou interrompue."""


class ConfigurationError(AcquisitionError):
    """Configuration invalide."""
