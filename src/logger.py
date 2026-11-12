"""XML diagnostics on stderr, with log handles owned by the current run."""

import builtins
from contextlib import contextmanager
from contextvars import ContextVar
import os
from pathlib import Path
import sys
from threading import RLock
from typing import Optional, TextIO


_current_logger: ContextVar[Optional['FoamAgentLogger']] = ContextVar('foamagent_logger', default=None)
LOG_FILENAMES = ('workflow.log', 'review.log')


def active_log_paths() -> tuple[Path, ...]:
    """Paths currently owned by diagnostics in this run's logging context."""
    logger = _current_logger.get()
    return tuple(Path(handle.name) for handle in (logger._workflow_file, logger._review_file)
                 if handle is not None) if logger is not None else ()


class FoamAgentLogger:
    def __init__(self):
        self._workflow_file: Optional[TextIO] = None
        self._review_file: Optional[TextIO] = None
        self._lock = RLock()

    @classmethod
    def get_instance(cls) -> 'FoamAgentLogger':
        logger = _current_logger.get()
        if logger is None:
            logger = cls()
            _current_logger.set(logger)
        return logger

    @property
    def initialized(self) -> bool:
        return self._workflow_file is not None

    def setup(self, output_dir: str, *, append: bool = False) -> None:
        if self.initialized:
            return
        paths = [Path(output_dir).absolute() / name for name in LOG_FILENAMES]
        for path in paths:
            for component in (*path.parents, path):
                if component.is_symlink():
                    raise ValueError(f"Log paths must not contain symbolic links: {component}")
            if path.exists() and not path.is_file():
                raise IsADirectoryError(f"Log target is not a regular file: {path}")
        os.makedirs(output_dir, exist_ok=True)
        mode = 'a' if append else 'w'
        self._workflow_file = open(paths[0], mode)
        self._review_file = open(paths[1], mode)

    def close(self) -> None:
        with self._lock:
            for handle in (self._workflow_file, self._review_file):
                if handle is not None:
                    handle.close()
            self._workflow_file = self._review_file = None

    def write(self, text: str, *, review: bool = False) -> None:
        with self._lock:
            sys.stderr.write(text)
            sys.stderr.flush()
            for handle in (self._workflow_file, self._review_file if review else None):
                if handle is not None:
                    handle.write(text)
                    handle.flush()

    def log_review(self, message: str, tag: str) -> None:
        self.write(f'<{tag}>\n{message}\n</{tag}>\n', review=True)


def print(*values, sep=' ', end='\n', file=None, flush=False) -> None:
    """Keep diagnostic call sites and text while leaving protocol stdout alone."""
    if file is not None and file is not sys.stderr:
        builtins.print(*values, sep=sep, end=end, file=file, flush=flush)
        return
    from io import StringIO
    text = StringIO()
    builtins.print(*values, sep=sep, end=end, file=text)
    FoamAgentLogger.get_instance().write(text.getvalue())


@contextmanager
def logging_context():
    """Create a shared handle before child graph/thread contexts are copied."""
    logger = FoamAgentLogger()
    token = _current_logger.set(logger)
    try:
        yield logger
    finally:
        logger.close()
        _current_logger.reset(token)


def setup_logging(output_dir: str, *, append: bool = False) -> None:
    FoamAgentLogger.get_instance().setup(output_dir, append=append)


def close_logging() -> None:
    logger = _current_logger.get()
    if logger is not None:
        logger.close()


def log_review(message: str, tag: str) -> None:
    FoamAgentLogger.get_instance().log_review(message, tag)
