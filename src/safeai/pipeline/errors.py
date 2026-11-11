"""Erreurs contrôlées du pipeline commun hors ligne."""


class PipelineError(Exception):
    """Erreur de base du pipeline commun."""


class PipelineConfigurationError(PipelineError):
    """Composant manquant ou configuration invalide."""
