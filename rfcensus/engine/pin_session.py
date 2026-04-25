"""Headless pin-selection logic shared by the CLI wizard and the TUI.

The CLI wizard in `rfcensus.commands.pin` previously embedded the
selection logic alongside `click.echo` / `click.prompt` calls. That
made the same logic unreachable from the Textual TUI's "edit pin"
modal — Textual takes over stdout, so any call into `click.prompt`
hangs indefinitely.

This module extracts the pure selection logic into pure functions and
small dataclasses. Each function takes the runtime context (dongle,
registry, current selection) and returns the available options or a
validated commit. No I/O. No prompting. The CLI wizard wraps these
calls with `click.echo` / `click.prompt`; the TUI wraps them with
Textual widgets.

# Design

Three reusable building blocks:

  • `available_frequencies(dongle)` → list of FrequencyOption that this
    dongle can tune given its antenna + hardware range. Includes the
    common-catalogue picks, marked. The TUI shows these in a list box;
    the CLI shows them in a numbered menu.

  • `available_decoders(freq_hz)` → list of DecoderOption sorted with
    suggested-for-freq decoders first. Same shape, same use.

  • `parse_custom_freq(raw)` → int | ValueError. Reuses the existing
    `engine.pinning._parse_freq_str` so the parsing rules stay in one
    place.

  • `validate_pin(dongle, freq_hz, decoder, sample_rate)` → ValidatedPin
    | list[ValidationError]. Last-mile sanity checks before committing
    (antenna covers freq, dongle covers freq, decoder is registered).

The CLI wizard pattern:

    opts = available_frequencies(dongle)
    choice = click.prompt(...)         # or _menu_with_back(...)
    freq_hz = opts[choice].freq_hz
    decoders = available_decoders(freq_hz)
    decoder = decoders[click.prompt(...)].name
    pin = validate_pin(dongle, freq_hz, decoder, None)
    if pin.errors: ...

The TUI pattern (will land in v0.7.0):

    opts = available_frequencies(dongle)
    self.freq_list.options = [o.label for o in opts]
    # … on user selection …
    decoders = available_decoders(opts[idx].freq_hz)
    self.decoder_list.options = [d.label for d in decoders]
    # … on user commit …
    pin = validate_pin(...)
    if not pin.errors:
        commit_to_session(pin.spec)

By keeping `pin_session` UI-agnostic, both surfaces stay in sync as
new validation rules or catalogue entries are added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from rfcensus.commands._frequency_guide import (
    COMMON_FREQUENCIES,
    FrequencyProfile,
)
from rfcensus.engine.pinning import _parse_freq_str


# ────────────────────────────────────────────────────────────────────
# Data shapes
# ────────────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class FrequencyOption:
    """One choice in the "pick a frequency" picker.

    `freq_hz` and `label` are always set. `profile` is the underlying
    catalogue entry when this option came from the common-frequencies
    list, or None for a custom-typed frequency.
    """

    freq_hz: int
    label: str
    profile: FrequencyProfile | None = None

    @property
    def is_custom(self) -> bool:
        """True if the user typed this freq in (not from the catalogue)."""
        return self.profile is None


@dataclass(slots=True, frozen=True)
class DecoderOption:
    """One choice in the "pick a decoder" picker."""

    name: str
    label: str
    suggested: bool = False


@dataclass(slots=True, frozen=True)
class ValidationError:
    """A specific reason a proposed pin can't be committed.

    `field` identifies which input the error is about, so a TUI form
    can highlight the right widget. Use `'general'` for cross-field
    constraints (e.g. "antenna doesn't cover this freq").
    """

    field: str  # 'freq_hz' | 'decoder' | 'sample_rate' | 'general'
    message: str


@dataclass(slots=True, frozen=True)
class ValidatedPin:
    """Result of `validate_pin`. Either `errors` is empty and `spec` is
    set, or `errors` is non-empty and `spec` is None.

    `spec` is the dict shape that `_apply_pin_dicts_to_toml` and the
    runtime PinSpec parser both consume — this keeps the existing
    write path unchanged.
    """

    spec: dict[str, object] | None
    errors: tuple[ValidationError, ...] = ()

    @property
    def ok(self) -> bool:
        return self.spec is not None and not self.errors


# ────────────────────────────────────────────────────────────────────
# Frequency selection
# ────────────────────────────────────────────────────────────────────


def available_frequencies(
    dongle, *, include_custom: bool = True,
) -> list[FrequencyOption]:
    """Return common-catalogue frequencies this dongle can tune.

    Filters by:
      • The dongle's antenna `covers(freq_hz)` (skipped if antenna is None)
      • The dongle's own hardware tuning range `dongle.covers(freq_hz)`

    The order matches `COMMON_FREQUENCIES` so the catalogue's curated
    grouping (popular signals first, niche ones later) is preserved.
    Caller is responsible for adding a "Custom" affordance if
    `include_custom` is True; we don't materialise a sentinel option
    here because the UI handling differs (CLI prompts inline, TUI
    pops a sub-modal).

    Returning an empty list is a meaningful signal — it means there
    are no catalogue picks at all, and the caller should jump straight
    to the custom-freq prompt.
    """
    out: list[FrequencyOption] = []
    for prof in COMMON_FREQUENCIES:
        if dongle.antenna is not None and not dongle.antenna.covers(prof.freq_hz):
            continue
        if not dongle.covers(prof.freq_hz):
            continue
        out.append(
            FrequencyOption(
                freq_hz=prof.freq_hz,
                label=prof.label,
                profile=prof,
            )
        )
    return out


def parse_custom_freq(raw: str) -> int:
    """Parse a free-form frequency string. Raises ValueError on bad input.

    Thin wrapper around `engine.pinning._parse_freq_str` so callers
    don't need to reach into a private name. Accepted formats include
    `433.92M`, `162M`, `850k`, plain integers in Hz.
    """
    return _parse_freq_str(raw)


# ────────────────────────────────────────────────────────────────────
# Decoder selection
# ────────────────────────────────────────────────────────────────────


def available_decoders(
    freq_hz: int,
    *,
    catalogue: Iterable[FrequencyProfile] = COMMON_FREQUENCIES,
    suggestion_tolerance: float = 0.05,
) -> list[DecoderOption]:
    """Return decoders sorted with suggested-for-freq ones first.

    A decoder is "suggested" if `freq_hz` is within
    `suggestion_tolerance` (default 5%) of any catalogue profile that
    lists this decoder. The 5% window is a soft heuristic — close
    enough to the canonical centre frequency that the matched-filter
    bandwidth will still cover it.

    All registered decoders are returned (suggested or not), so the
    user is never forced into the suggestion list.
    """
    from rfcensus.decoders.registry import get_registry

    suggested: set[str] = set()
    for prof in catalogue:
        if abs(prof.freq_hz - freq_hz) / max(prof.freq_hz, 1) < suggestion_tolerance:
            suggested.update(prof.decoders)
            # Don't break — multiple catalogue entries can land within
            # tolerance (e.g. 433.92 + 433.42 ISM packets).

    all_decoders = sorted(get_registry().names())

    out: list[DecoderOption] = []
    # Suggested decoders first, in catalogue order
    for name in all_decoders:
        if name in suggested:
            out.append(DecoderOption(
                name=name,
                label=f"{name}  (suggested)",
                suggested=True,
            ))
    # Then everything else
    for name in all_decoders:
        if name not in suggested:
            out.append(DecoderOption(name=name, label=name, suggested=False))
    return out


# ────────────────────────────────────────────────────────────────────
# Final validation + commit
# ────────────────────────────────────────────────────────────────────


def validate_pin(
    dongle,
    *,
    freq_hz: int,
    decoder: str,
    sample_rate: int | None = None,
    access_mode: str = "exclusive",
) -> ValidatedPin:
    """Last-mile validation before committing a pin.

    Checks that mirror PinSpec's parser plus the dongle/antenna
    constraints. Returns either a ValidatedPin with `spec` set (ready
    to feed to `_apply_pin_dicts_to_toml`) or a ValidatedPin with
    `errors` listing every problem found (so the TUI can show all of
    them at once instead of one-at-a-time).
    """
    errors: list[ValidationError] = []

    # Frequency must be a positive integer
    if not isinstance(freq_hz, int) or freq_hz <= 0:
        errors.append(ValidationError(
            field="freq_hz",
            message=f"freq_hz must be a positive integer, got {freq_hz!r}",
        ))
    else:
        # Antenna coverage (if antenna is set)
        if dongle.antenna is not None and not dongle.antenna.covers(freq_hz):
            errors.append(ValidationError(
                field="freq_hz",
                message=(
                    f"antenna {dongle.antenna.name!r} does not cover "
                    f"{freq_hz / 1e6:.3f} MHz"
                ),
            ))
        # Hardware range
        if not dongle.covers(freq_hz):
            errors.append(ValidationError(
                field="freq_hz",
                message=(
                    f"dongle {dongle.id} hardware range does not "
                    f"cover {freq_hz / 1e6:.3f} MHz"
                ),
            ))

    # Decoder must be registered
    from rfcensus.decoders.registry import get_registry
    if decoder not in get_registry().names():
        errors.append(ValidationError(
            field="decoder",
            message=f"decoder {decoder!r} is not registered",
        ))

    # Sample rate, if specified, must be positive
    if sample_rate is not None:
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            errors.append(ValidationError(
                field="sample_rate",
                message=(
                    f"sample_rate must be a positive integer, "
                    f"got {sample_rate!r}"
                ),
            ))

    # Access mode must be one of the supported values. The runtime
    # PinSpec accepts `"exclusive"` and `"shared"`. Anything else is
    # a typo we catch here rather than at session-start time.
    if access_mode not in ("exclusive", "shared"):
        errors.append(ValidationError(
            field="general",
            message=(
                f"access_mode must be 'exclusive' or 'shared', "
                f"got {access_mode!r}"
            ),
        ))

    if errors:
        return ValidatedPin(spec=None, errors=tuple(errors))

    spec: dict[str, object] = {
        "decoder": decoder,
        "freq_hz": freq_hz,
        "access_mode": access_mode,
    }
    if sample_rate is not None:
        spec["sample_rate"] = sample_rate
    return ValidatedPin(spec=spec)
