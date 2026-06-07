"""Domain-specific exception types for BUCAD runtime failures."""

from __future__ import annotations


class BucadError(Exception):
    """Base exception for BUCAD runtime failures."""


class InvalidInputError(BucadError):
    """Raised when an uploaded image cannot be processed safely."""


class QualityBlockedError(BucadError):
    """Raised when an image is too poor for reliable analysis."""


class ClassificationUnavailableError(BucadError):
    """Raised when no classifier path is available at runtime."""


class OptionalOutputUnavailableError(BucadError):
    """Raised when an optional visualization cannot be generated."""


class UnexpectedRuntimeError(BucadError):
    """Raised for uncaught runtime issues that should surface cleanly."""
