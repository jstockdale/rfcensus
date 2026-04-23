"""Standard paths for rfcensus configuration and data.

We follow XDG Base Directory conventions on Linux:

• Config: $XDG_CONFIG_HOME/rfcensus or ~/.config/rfcensus
• Data: $XDG_DATA_HOME/rfcensus or ~/.local/share/rfcensus
• State: $XDG_STATE_HOME/rfcensus or ~/.local/state/rfcensus (for logs)
• Cache: $XDG_CACHE_HOME/rfcensus or ~/.cache/rfcensus

On macOS we honor XDG env vars if set, otherwise fall back to the same
~/.config layout (deliberately not using ~/Library/... to keep things
portable and greppable).

All directories are lazily created on first access.
"""

from __future__ import annotations

import os
from pathlib import Path


def _xdg_dir(env_var: str, fallback: str) -> Path:
    value = os.environ.get(env_var)
    if value:
        return Path(value).expanduser().resolve() / "rfcensus"
    return Path.home() / fallback / "rfcensus"


def config_dir() -> Path:
    """~/.config/rfcensus/ by default."""
    path = _xdg_dir("XDG_CONFIG_HOME", ".config")
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    """~/.local/share/rfcensus/ by default. Holds SQLite DB and captures."""
    path = _xdg_dir("XDG_DATA_HOME", ".local/share")
    path.mkdir(parents=True, exist_ok=True)
    return path


def state_dir() -> Path:
    """~/.local/state/rfcensus/ by default. Holds log files."""
    path = _xdg_dir("XDG_STATE_HOME", ".local/state")
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    """~/.cache/rfcensus/ by default."""
    path = _xdg_dir("XDG_CACHE_HOME", ".cache")
    path.mkdir(parents=True, exist_ok=True)
    return path


def site_config_path() -> Path:
    """Primary user site configuration."""
    return config_dir() / "site.toml"


def database_path() -> Path:
    """Primary SQLite database."""
    return data_dir() / "rfcensus.db"


def log_path() -> Path:
    """Default log file."""
    return state_dir() / "rfcensus.log"


def capture_dir() -> Path:
    """Directory for IQ captures and recordings."""
    path = data_dir() / "captures"
    path.mkdir(parents=True, exist_ok=True)
    return path
