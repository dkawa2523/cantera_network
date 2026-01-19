"""Logging configuration and error reporting helpers."""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any, Optional, TypeVar

from rxn_platform.errors import RxnPlatformError

DEFAULT_LOGGER_NAME = "rxn_platform"
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

_T = TypeVar("_T")


def configure_logging(
    level: int = DEFAULT_LOG_LEVEL,
    *,
    logger_name: str = DEFAULT_LOGGER_NAME,
    fmt: str = DEFAULT_LOG_FORMAT,
    force: bool = False,
) -> logging.Logger:
    logging.basicConfig(level=level, format=fmt, force=force)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    return logger


def get_user_message(exc: BaseException) -> str:
    if isinstance(exc, RxnPlatformError):
        return exc.user_message
    return f"Unexpected error: {exc}"


def log_exception(
    logger: logging.Logger,
    exc: BaseException,
    *,
    show_traceback: bool = False,
) -> str:
    user_message = get_user_message(exc)
    logger.error(user_message)
    if show_traceback:
        logger.error("Detailed traceback:", exc_info=exc)
    else:
        logger.debug("Detailed traceback:", exc_info=exc)
    return user_message


def run_with_error_handling(
    func: Callable[..., _T],
    *args: Any,
    logger: logging.Logger,
    show_traceback: bool = False,
    **kwargs: Any,
) -> _T:
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        log_exception(logger, exc, show_traceback=show_traceback)
        raise


__all__ = [
    "DEFAULT_LOGGER_NAME",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_LOG_FORMAT",
    "configure_logging",
    "get_user_message",
    "log_exception",
    "run_with_error_handling",
]
