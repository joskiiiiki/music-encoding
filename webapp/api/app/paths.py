"""Filesystem layout for the web app.

Everything here is resolved as: **environment variable -> derived default**, so a
deployment (or a test) can point the app anywhere without editing code.

The one non-obvious rule is :func:`chroma_db_dir`. ``chroma_db/`` is gitignored, so
a linked git worktree does **not** contain it — but the web app is developed in a
worktree. ``git rev-parse --git-common-dir`` resolves to the *main* checkout's
``.git`` even when run from a linked worktree, so its parent is the directory that
actually holds the 400 MB database. That lets the API read the real DB in place
with no copy and no symlink.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# webapp/api/app/paths.py -> app -> api -> webapp
WEBAPP_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = WEBAPP_ROOT.parent


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else None


def main_checkout() -> Path | None:
    """The main working tree's root, even when called from a linked worktree."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=WEBAPP_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    git_dir = Path(out)
    # A normal checkout gives "<root>/.git"; a bare repo or unusual layout gives
    # something else, in which case we simply fall back to the caller's default.
    return git_dir.parent if git_dir.name == ".git" else None


def data_dir() -> Path:
    """Where the exported catalog + vectors live (gitignored)."""
    return _env_path("WEBAPP_DATA_DIR") or WEBAPP_ROOT / "data"


def catalog_path() -> Path:
    return data_dir() / "catalog.sqlite"


def audio_db_path() -> Path:
    """Audio availability, in its own file on purpose.

    Re-exporting the catalog (e.g. for a new checkpoint) rewrites ``catalog.sqlite``
    wholesale, but the preview cache is the product of an hours-long pre-warm against
    third-party APIs. Keeping it separate means a re-export can never wipe it.
    """
    return data_dir() / "audio.sqlite"


def vectors_path() -> Path:
    return data_dir() / "vectors.npz"


def chroma_db_dir() -> Path:
    """The source chroma DB. Falls back to the main checkout, then the repo root."""
    if explicit := _env_path("WEBAPP_CHROMA_DB"):
        return explicit
    if (root := main_checkout()) and (root / "chroma_db").is_dir():
        return root / "chroma_db"
    return _REPO_ROOT / "chroma_db"


def mtg_data_dir() -> Path:
    """MTG-Jamendo's ``data/`` directory (the TSVs)."""
    return _env_path("MTG_DATA_DIR") or Path.home() / "mtg-jamendo-dataset" / "data"


def mtg_audio_dir() -> Path:
    """Directory holding the MTG audio archives.

    ``raw_30s/audio-low`` is published as one tar per bucket, where the bucket is
    ``track_num % 100``. ``fetch_mtg_audio.py`` pulls the buckets the corpus needs and
    ``local_audio_index.py`` records each member's byte offset, so the API streams from
    the archives without unpacking them -- the tar is the only disk cost. A local row
    names its archive in ``audio.tar``.
    """
    return _env_path("MTG_AUDIO_DIR") or Path.home() / "mtg"


def mtg_audio_tar() -> Path:
    """Kept for callers written before the audio split into per-bucket archives."""
    return _env_path("MTG_AUDIO_TAR") or mtg_audio_dir() / "raw_30s_audio-low-00.tar"
