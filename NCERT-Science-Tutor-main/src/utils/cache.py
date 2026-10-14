"""Build-time artifact cache.

Scope note: this cache exists for the *index build*, so that iterating on chunking
parameters does not mean re-extracting 1,500 pages of PDF each time. It is not on the
query path — serving reads the Chroma and FTS5 artifacts directly.

## On pickle

Pickle executes arbitrary code on load, so this would be a serious vulnerability if
the cache directory were ever attacker-writable. It is not: these files are produced
by the build script on a developer machine and, where committed, reviewed in a diff.
The alternative (JSON) cannot round-trip LangChain ``Document`` objects without a
custom codec, and the risk profile does not justify writing one.

What pickle *does* cause in practice is version-skew breakage: a cache written under
one LangChain version can fail to load under another, with an exception that names an
attribute rather than the real problem. Loads are therefore guarded and return
``None`` on failure so callers can rebuild rather than crash.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    """Pickle-backed key/value store over a directory."""

    def __init__(self, cache_dir: str | Path = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.pkl"

    def exists(self, name: str) -> bool:
        path = self._path(name)
        # A zero-byte file is what an interrupted write leaves behind. Reporting it
        # as present sends the caller into `load`, which then raises EOFError.
        return path.exists() and path.stat().st_size > 0

    def save(self, name: str, data: Any) -> None:
        """Write atomically.

        Straight into the final path, a build interrupted mid-write leaves a
        truncated pickle that `exists()` reports as present and `load()` cannot read.
        Writing to a temp file and renaming makes the swap atomic on POSIX and
        near-atomic on Windows.
        """
        path = self._path(name)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        logger.debug("cached %s (%.1f MB)", name, path.stat().st_size / 1e6)

    def load(self, name: str, default: Any = None) -> Any:
        """Read a cached value, returning ``default`` rather than raising.

        The original code called this without checking existence first, which turned
        a cold cache into a ``FileNotFoundError`` at application startup. Callers
        should be able to treat a cache miss as a cache miss.
        """
        path = self._path(name)
        if not self.exists(name):
            logger.debug("cache miss: %s", name)
            return default

        try:
            with open(path, "rb") as handle:
                return pickle.load(handle)
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, ModuleNotFoundError) as exc:
            logger.warning(
                "cache entry %s is unreadable (%s) — treating as a miss. "
                "This usually means it was written by a different library version.",
                name,
                exc,
            )
            return default

    def clear(self, name: str | None = None) -> None:
        """Delete one entry, or all of them."""
        if name is not None:
            self._path(name).unlink(missing_ok=True)
            return
        for path in self.cache_dir.glob("*.pkl"):
            path.unlink(missing_ok=True)
