"""Small file helpers shared by caches, logs and saved knowledge: atomic writes."""
import os
from pathlib import Path


def write_atomic(path, text):
    """Write via a temp file and rename, so a kill never leaves a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
