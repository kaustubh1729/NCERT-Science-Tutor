"""Logging setup.

The previous implementation called ``logging.basicConfig()`` inside ``get_logger``,
which meant the first module to import won the configuration race and every later
call silently did nothing. Configuration now happens exactly once, explicitly.

Output goes to **stdout**, not stderr. Render (like most container platforms)
captures both, but interleaves them unpredictably; putting everything on one stream
keeps the log ordering in the dashboard faithful to the order events actually
happened.
"""

from __future__ import annotations

import contextlib
import logging
import sys

_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATEFMT = "%H:%M:%S"


def _force_utf8_stdio() -> None:
    """Make stdout/stderr UTF-8 regardless of the console's code page.

    NCERT text contains typographic ligatures (U+FB01 'ﬁ'), curly quotes and the
    degree sign. Windows consoles default to cp1252, which cannot encode any of them,
    so printing a retrieved passage raised:

        UnicodeEncodeError: 'charmap' codec can't encode character '\\ufb01'

    That is a crash in the *output* layer for content the pipeline handled correctly
    — the most annoying kind, because it looks like a retrieval failure. ``replace``
    rather than ``strict`` so an unmappable character degrades to '?' instead of
    taking down the process.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            # Suppressed: the stream may be redirected to a pipe that does not
            # support reconfiguration, which is not a reason to fail startup.
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


def configure_logging(level: str = "INFO", stream=None) -> None:
    """Install the root handler. Idempotent; safe to call from every entrypoint.

    Args:
        level: Root log level.
        stream: Where log records go. Defaults to stdout, which is right for a
            server. Pass ``sys.stderr`` when stdout is a **data channel** — the CLI
            does this for ``--json``, because a log line interleaved with the
            document makes the output unparseable.

            Note that the handler binds this object once. Wrapping a later call in
            ``contextlib.redirect_stdout`` will not move existing log output, since
            that rebinds ``sys.stdout`` and the handler already holds the original.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    _force_utf8_stdio()

    handler = logging.StreamHandler(stream=stream if stream is not None else sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # These libraries log a request line per call at INFO. On the query path that is
    # three lines of noise per question with no diagnostic value.
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Fetch a namespaced logger. Assumes ``configure_logging`` ran at startup."""
    return logging.getLogger(name)
