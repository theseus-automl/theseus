class UnsupportedLanguageError(Exception):
    """Thrown when target language is not supported by model."""


class ModelNotFoundError(Exception):
    """Thrown when fastText language detection model is not found."""


class NotEnoughResourcesError(Exception):
    """Thrown when unable to use heavy model due to RAM or CUDA memory issues."""


class DeviceError(Exception):
    """Thrown when CUDA device is unavailable."""
