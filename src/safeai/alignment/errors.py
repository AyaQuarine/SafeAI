"""Erreurs du contrat d'alignement voix–texte."""


class AlignmentError(Exception):
    """Erreur de base de l'alignement."""


class InvalidAlignmentError(AlignmentError):
    """Résultat d'alignement invalide ou échantillon d'interface illisible."""
