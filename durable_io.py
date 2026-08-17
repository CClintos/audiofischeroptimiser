"""Small durable-file helpers shared by workers, merger, and GUI state."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


def replace_with_retry(source: Path, destination: Path, attempts: int = 16) -> None:
    """Atomically replace a file despite brief Windows reader/AV locks."""
    attempts = max(1, int(attempts))
    for attempt in range(attempts):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(min(0.025 * (2 ** attempt), 0.4))


def atomic_write_text(
    path: Path, text: str, *, encoding: str = "utf-8", attempts: int = 16,
) -> Path:
    """Write through a unique sibling and atomically publish the complete file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(text, encoding=encoding)
        replace_with_retry(temporary, path, attempts=attempts)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def atomic_write_json(
    path: Path, payload: Any, *, indent: int | None = None, attempts: int = 16,
) -> Path:
    return atomic_write_text(
        path, json.dumps(payload, indent=indent), attempts=attempts,
    )
