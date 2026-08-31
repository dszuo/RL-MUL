"""Persistent memo for synthesis results.

Search revisits the same compressor tree constantly -- a random walk backtracks,
episodes restart from the same seed, and separate runs share a starting point --
while a single synthesis costs seconds.  Keying results on the tree itself turns
those repeats into lookups, which is the single biggest speedup available to
this loop.

Backed by SQLite, and every access is best-effort: a cache that cannot be opened,
read or written degrades to a miss rather than taking down a run that may have
hours of completed synthesis behind it. Losing a memo costs one resynthesis;
raising out of here costs everything.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ppa (
    key    TEXT PRIMARY KEY,
    area   REAL NOT NULL,
    delay  REAL NOT NULL
)
"""


def tree_key(counts: np.ndarray, width: int, target_delay: float, tag: str = "") -> str:
    """Identity of one synthesis job: the tree, its width, and the constraint."""
    flat = ",".join(str(int(v)) for v in np.asarray(counts).ravel())
    return f"{tag}|{width}|{target_delay:g}|{flat}"


class PPACache:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._mem: dict[str, tuple[float, float]] = {}
        self._db: sqlite3.Connection | None = None
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._db = sqlite3.connect(self.path, check_same_thread=False)
                self._db.execute(_SCHEMA)
                self._db.commit()
            except (sqlite3.Error, OSError) as exc:
                # Another process holding a write lock, a read-only directory, a
                # full disk. Run without a persistent cache rather than not at all.
                log.warning("running without a persistent PPA cache: %s", exc)
                self._db = None

    def get(self, key: str) -> tuple[float, float] | None:
        with self._lock:
            hit = self._mem.get(key)
            if hit is not None or self._db is None:
                return hit
            try:
                row = self._db.execute(
                    "SELECT area, delay FROM ppa WHERE key = ?", (key,)
                ).fetchone()
            except sqlite3.Error as exc:
                log.warning("PPA cache read failed, treating as a miss: %s", exc)
                return None
            if row is None:
                return None
            self._mem[key] = (row[0], row[1])
            return self._mem[key]

    def put(self, key: str, area: float, delay: float) -> None:
        with self._lock:
            self._mem[key] = (area, delay)
            if self._db is not None:
                try:
                    self._db.execute(
                        "INSERT OR REPLACE INTO ppa (key, area, delay) VALUES (?, ?, ?)",
                        (key, area, delay),
                    )
                    self._db.commit()
                except sqlite3.Error as exc:
                    log.warning("PPA cache write failed: %s", exc)

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                try:
                    self._db.close()
                except sqlite3.Error:
                    pass
                self._db = None
