"""Error hierarchy for rxn_platform."""

from __future__ import annotations

from typing import Any, Mapping, Optional


class RxnPlatformError(Exception):
    """Base exception for rxn_platform failures."""

    def __init__(
        self,
        message: str,
        *,
        user_message: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.user_message = message if user_message is None else user_message
        self.context = dict(context) if context else {}

    def log_message(self) -> str:
        if not self.context:
            return str(self)
        return f"{self}: {self.context}"


class ConfigError(RxnPlatformError):
    """Configuration loading or validation error."""


class ArtifactError(RxnPlatformError):
    """Artifact storage or retrieval error."""


class BackendError(RxnPlatformError):
    """Backend registration or execution error."""


class TaskError(RxnPlatformError):
    """Task resolution or execution error."""


class ValidationError(RxnPlatformError):
    """Validation error for input data or schema."""


__all__ = [
    "RxnPlatformError",
    "ConfigError",
    "ArtifactError",
    "BackendError",
    "TaskError",
    "ValidationError",
]
