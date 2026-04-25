"""Semantic color palette for the TUI.

Color carries STATUS, not identity. The dongle slot number is white
no matter which dongle it is — the `●` next to it is colored to
reflect that dongle's current state. Color stasis means "nothing to
worry about"; color change means "look here."

Palette
-------

  • green     — active, healthy, baseline "everything's working"
  • cyan      — new emitter / detection (the "look at this!" color,
                rare by design so it stands out)
  • yellow    — degraded, slow, warning (noteworthy but not broken)
  • red       — failed, broken, permanent error
  • dim       — idle, inactive, historical
  • bold      — focus indicator only — never used for content meaning

Colors NOT used: magenta, blue, bright variants other than cyan. They
compete with the meaningful colors above and confuse the eye.

NO_COLOR support
----------------

If the user passes `--no-color` or sets `NO_COLOR=1` in their env,
all `style_for(...)` calls return empty strings. Widgets prepend the
style tag inline, so an empty style means plain text rendering.
Textual respects NO_COLOR natively in its own widgets, so this only
governs our explicit color choices.
"""

from __future__ import annotations

import os


# ────────────────────────────────────────────────────────────────────
# Style tokens (Textual / rich markup)
# ────────────────────────────────────────────────────────────────────


# Map of semantic name → Rich/Textual style string. Empty string in
# no-color mode. Uses Textual's color names so the terminal's own
# color scheme is respected — we don't hardcode RGB.
_STYLES: dict[str, str] = {
    "active":    "green",
    "highlight": "bright_cyan",
    "warning":   "yellow",
    "error":     "red",
    "idle":      "dim",
    "focus":     "bold white",
    "info":      "white",
    "good":      "green",
    # Stream-entry severities:
    "stream_info":      "white",
    "stream_good":      "green",
    "stream_warning":   "yellow",
    "stream_error":     "red bold",
    "stream_highlight": "bright_cyan bold",
}


# Module-level toggle. The CLI sets this once at TUI startup based on
# the --no-color flag and the NO_COLOR env var.
_color_enabled: bool = True


def configure_color(enabled: bool) -> None:
    """Set the global color-enabled flag.

    Called once at TUI startup. The Textual app respects the same flag
    natively for its own widgets (via the NO_COLOR env var which we
    set when this is False). Our explicit `style(...)` calls also
    consult this flag, so widget code doesn't need to branch.
    """
    global _color_enabled
    _color_enabled = bool(enabled)
    if not enabled:
        # NO_COLOR is the de-facto standard env var (https://no-color.org);
        # setting it ensures any subprocess output we render also drops
        # color, and Textual itself honors it.
        os.environ["NO_COLOR"] = "1"


def detect_color_support() -> bool:
    """Default color enablement based on env. Can be overridden by
    `configure_color(False)` after this returns True."""
    if os.environ.get("NO_COLOR"):
        return False
    return True


def style(name: str) -> str:
    """Return the Rich/Textual style string for a semantic token.

    Empty string in no-color mode (widgets just render plain text).
    Unknown tokens return empty (caller gets plain text rather than
    a crash).
    """
    if not _color_enabled:
        return ""
    return _STYLES.get(name, "")


# ────────────────────────────────────────────────────────────────────
# Convenience: style a piece of text inline
# ────────────────────────────────────────────────────────────────────


def styled(name: str, text: str) -> str:
    """Wrap `text` in Rich markup for the named style.

    Plain text in no-color mode. Use sparingly — most widgets are
    better served by passing styles to Textual's renderable system
    directly. This helper is for one-off inline highlights in a
    plain string.
    """
    s = style(name)
    if not s:
        return text
    return f"[{s}]{text}[/]"


# ────────────────────────────────────────────────────────────────────
# Per-status mapping helpers (used by widgets)
# ────────────────────────────────────────────────────────────────────


def style_for_dongle_status(status: str) -> str:
    """Map DongleState.status → style name."""
    return {
        "active":           "active",
        "idle":             "idle",
        "degraded":         "warning",
        "failed":           "error",
        "permanent_failed": "error",
    }.get(status, "info")


def style_for_severity(severity: str) -> str:
    """Map StreamEntry.severity → style name."""
    return {
        "info":      "stream_info",
        "good":      "stream_good",
        "warning":   "stream_warning",
        "error":     "stream_error",
        "highlight": "stream_highlight",
    }.get(severity, "stream_info")


def dongle_status_glyph(status: str) -> str:
    """The single-character status indicator next to a dongle's slot.

    `●` for any active/idle state, `✗` for failed/permanent. The
    color, applied separately, conveys the live status.
    """
    if status in ("failed", "permanent_failed"):
        return "✗"
    if status == "idle":
        return "○"
    return "●"
