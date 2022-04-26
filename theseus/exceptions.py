class UnsupportedLanguageError(Exception):
    """Thrown when target language is not supported by model."""


class ModelNotFoundError(Exception):
    """Thrown when fastText language detection model is not found."""
