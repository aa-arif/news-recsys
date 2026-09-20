"""Small logging helper so every script prints timings the same way."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        # Windows consoles default to cp1252; MIND titles are full of non-latin-1 text.
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(encoding="utf-8", errors="replace")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("news_recsys")
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(f"news_recsys.{name}")


@contextmanager
def timed(logger: logging.Logger, label: str) -> Iterator[dict[str, float]]:
    """Log wall-clock time for a block; yields a dict that receives ``seconds``."""
    result: dict[str, float] = {}
    start = time.perf_counter()
    logger.info("%s ...", label)
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result["seconds"] = elapsed
        logger.info("%s done in %.2fs", label, elapsed)
