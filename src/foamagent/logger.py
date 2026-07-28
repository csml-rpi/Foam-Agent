# logger.py
"""Centralized logging for Foam-Agent.

Diagnostics go to **stderr** through the standard `logging` module. This matters beyond
tidiness: when the MCP server speaks stdio, stdout *is* the protocol channel, so anything
written there corrupts the session. Only the CLI's own machine-readable markers
(`<workflow_end>`, `<case_dir>`, ...) belong on stdout, and `main.py` prints those directly.

`setup_logging(case_dir)` additionally captures a run's output into two files inside the
case directory:

- ``workflow.log`` — every log record, plus whatever the CLI writes to stdout
- ``review.log``   — only reviewer output (error logs, review analysis, rewrite plans)

Usage:
    from foamagent.logger import get_logger, setup_logging, close_logging, log_review

    logger = get_logger(__name__)
    setup_logging("/path/to/case_dir")   # call once case_dir is known
    log_review(error_text, "error_logs")
    close_logging()
"""

import logging
import os
import sys
from typing import Optional, TextIO

_ROOT_LOGGER_NAME = "foamagent"
_LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"


def _root_logger() -> logging.Logger:
    """Return the package logger, attaching a stderr handler on first use."""
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    if not any(getattr(h, "_foamagent_stream", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        handler._foamagent_stream = True
        logger.addHandler(handler)
        logger.propagate = False
    level = os.getenv("FOAMAGENT_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger under the ``foamagent`` hierarchy.

    Pass ``__name__`` from a module inside the package; anything outside it is nested
    under ``foamagent.`` so a single handler covers every caller.
    """
    root = _root_logger()
    if not name or name == _ROOT_LOGGER_NAME:
        return root
    if name.startswith(_ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


class _TeeWriter:
    """Write to both the original stream and a log file."""

    def __init__(self, original: TextIO, log_file: TextIO):
        self._original = original
        self._log_file = log_file

    def write(self, text: str):
        self._original.write(text)
        if self._log_file and not self._log_file.closed:
            self._log_file.write(text)
            self._log_file.flush()

    def flush(self):
        self._original.flush()
        if self._log_file and not self._log_file.closed:
            self._log_file.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


class FoamAgentLogger:
    """Singleton that routes a run's output into workflow.log and review.log."""

    _instance: Optional["FoamAgentLogger"] = None

    def __init__(self):
        self._workflow_file: Optional[TextIO] = None
        self._review_file: Optional[TextIO] = None
        self._original_stdout: Optional[TextIO] = None
        self._file_handler: Optional[logging.Handler] = None
        self._initialized = False

    @classmethod
    def get_instance(cls) -> "FoamAgentLogger":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def initialized(self) -> bool:
        return self._initialized

    def setup(self, output_dir: str) -> None:
        """Open the log files, and start capturing log records and stdout into them."""
        if self._initialized:
            return
        os.makedirs(output_dir, exist_ok=True)

        self._workflow_file = open(os.path.join(output_dir, "workflow.log"), "w")
        self._review_file = open(os.path.join(output_dir, "review.log"), "w")

        self._file_handler = logging.StreamHandler(self._workflow_file)
        self._file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        _root_logger().addHandler(self._file_handler)

        # The CLI still writes its markers to stdout; tee them so a run's log file holds
        # the complete picture rather than only the parts that went through logging.
        self._original_stdout = sys.stdout
        sys.stdout = _TeeWriter(self._original_stdout, self._workflow_file)
        self._initialized = True

    def close(self) -> None:
        """Stop capturing, restore stdout, and close the log files."""
        if self._file_handler is not None:
            _root_logger().removeHandler(self._file_handler)
            self._file_handler = None
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
            self._original_stdout = None
        if self._workflow_file and not self._workflow_file.closed:
            self._workflow_file.close()
            self._workflow_file = None
        if self._review_file and not self._review_file.closed:
            self._review_file.close()
            self._review_file = None
        self._initialized = False

    def log_review(self, message: str, tag: str) -> None:
        """Log a tagged reviewer message to the normal log and to review.log."""
        output = f"<{tag}>\n{message}\n</{tag}>"
        get_logger("review").info("%s", output)
        if self._review_file and not self._review_file.closed:
            self._review_file.write(output + "\n")
            self._review_file.flush()


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def setup_logging(output_dir: str) -> None:
    """Initialize logging to output_dir. Call once after case_dir is created."""
    FoamAgentLogger.get_instance().setup(output_dir)


def close_logging() -> None:
    """Close log files and restore stdout."""
    FoamAgentLogger.get_instance().close()


def log_review(message: str, tag: str) -> None:
    """Log to the normal log stream and review.log, wrapped in <tag>...</tag>."""
    FoamAgentLogger.get_instance().log_review(message, tag)
