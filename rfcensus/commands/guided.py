"""Guided scan mode.

For users with limited hardware (typically one telescopic-whip dongle),
guided mode prompts for retuning between bands so a single antenna can
cover the whole enabled-band list without needing one whip per band.

Single-pass scans only — multi-pass / round-robin / indefinite would
prompt the user dozens of times and is hostile.

Flow:
  1. prepare_guided_scan(): identify the telescopic dongle, show the
     plan upfront, get user buy-in
  2. before_band_callback(): per-band, prompt for retune if the
     quarter-wave length differs meaningfully from the previous band
  3. after_session_callback(): post-scan, offer to restore the
     antenna to its config-stored length OR update the config to
     reflect the actual current length (no stale config)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import click

from rfcensus.config.schema import BandConfig, SiteConfig
from rfcensus.hardware.dongle import Dongle


log = logging.getLogger(__name__)

# Quarter-wave length difference below which we consider it "no
# meaningful retune needed" — within typical antenna section steps
SAME_TUNING_TOLERANCE_PCT = 0.10


@dataclass
class GuidedConfig:
    """Holds the dongle/antenna context for guided mode."""

    dongle_id: str
    antenna_id: str
    original_resonant_freq_hz: int
    original_quarter_wave_cm: float
    bands_in_order: list[BandConfig] = field(default_factory=list)
    # Mutable: tracks the last frequency we asked the user to tune to,
    # so we can skip the prompt when the next band is essentially the
    # same tuning.
    current_quarter_wave_cm: float = 0.0


def find_telescopic_dongle(
    dongles: list[Dongle], antennas_by_id: dict
) -> tuple[Dongle | None, dict | None, str]:
    """Find the dongle to use for guided mode.

    Returns (dongle, antenna_dict, message). If no telescopic dongle
    is detected, returns (None, None, message_for_user). The caller
    decides whether to error or proceed.

    Telescopic detection: id starts with `whip_telescopic_`. v0.5.10+
    creates these for any wizard-assigned telescopic. Pre-v0.5.10
    configs may not match — we fall back to the first dongle with any
    user-defined antenna in that case.
    """
    candidates: list[tuple[Dongle, dict]] = []
    for d in dongles:
        if not d.is_usable() or not d.antenna:
            continue
        ant_id = d.antenna.id
        ant = antennas_by_id.get(ant_id)
        if not ant:
            continue
        if ant_id.startswith("whip_telescopic_"):
            candidates.append((d, ant))

    if not candidates:
        return None, None, (
            "No telescopic-whip antenna detected in your config. Guided "
            "mode works best with a telescopic whip you can adjust between "
            "bands. Either fit one and re-run setup, or proceed knowing "
            "the prompts will tell you what length to set regardless."
        )

    if len(candidates) > 1:
        chosen = sorted(candidates, key=lambda x: x[0].id)[0]
        return chosen[0], chosen[1], (
            f"Multiple telescopic dongles available; using {chosen[0].id}. "
            f"Pass --guided-dongle ID if you want a different one."
        )

    return candidates[0][0], candidates[0][1], ""


def build_guided_config(
    dongle: Dongle, antenna_dict: dict | None, bands: list[BandConfig],
) -> GuidedConfig:
    """Build the GuidedConfig from a chosen dongle + the band list.

    `antenna_dict` is the raw dict from site.toml (so we can update
    its resonant_freq_hz later if the user reports a new length).
    Order bands by frequency so adjacent bands often share tuning.
    """
    if antenna_dict and "resonant_freq_hz" in antenna_dict:
        original_freq = antenna_dict["resonant_freq_hz"]
    elif dongle.antenna:
        original_freq = dongle.antenna.resonant_freq_hz or 100_000_000
    else:
        original_freq = 100_000_000  # arbitrary default
    original_qw_cm = quarter_wave_cm(original_freq)
    return GuidedConfig(
        dongle_id=dongle.id,
        antenna_id=dongle.antenna.id if dongle.antenna else "",
        original_resonant_freq_hz=original_freq,
        original_quarter_wave_cm=original_qw_cm,
        bands_in_order=sorted(bands, key=lambda b: b.center_hz),
        current_quarter_wave_cm=original_qw_cm,
    )


def quarter_wave_cm(freq_hz: int) -> float:
    """Quarter-wavelength in cm for a given frequency in Hz."""
    return (29979.2458 / (freq_hz / 1_000_000)) / 4


def show_plan(cfg: GuidedConfig) -> None:
    """Print the upfront plan so the user knows what they're in for."""
    click.echo()
    click.echo("─" * 72)
    click.echo(" Guided scan mode")
    click.echo("─" * 72)
    click.echo()
    click.echo(
        f"  I'll prompt between bands so you can retune your telescopic"
    )
    click.echo(
        f"  whip on dongle {cfg.dongle_id}. The whip is currently"
    )
    click.echo(
        f"  at {cfg.original_quarter_wave_cm:.1f} cm "
        f"(tuned for {cfg.original_resonant_freq_hz/1e6:.0f} MHz)."
    )
    click.echo()
    click.echo("  Bands to scan, in frequency order:")
    last_qw = -1.0
    for i, band in enumerate(cfg.bands_in_order, 1):
        qw = quarter_wave_cm(band.center_hz)
        delta_pct = (
            abs(qw - last_qw) / last_qw if last_qw > 0 else 1.0
        )
        retune_note = (
            "no retune needed"
            if delta_pct < SAME_TUNING_TOLERANCE_PCT
            else f"set to {qw:.1f} cm"
        )
        click.echo(
            f"    {i}) {band.id:<24s} "
            f"({band.center_hz/1e6:6.1f} MHz, {retune_note})"
        )
        last_qw = qw
    click.echo()


def make_before_band_callback(cfg: GuidedConfig):
    """Return an async callback suitable for SessionRunner's
    before_task_hook. Prompts the user to retune when needed."""

    async def callback(task) -> str:
        band = task.band
        target_qw = quarter_wave_cm(band.center_hz)
        delta_pct = abs(target_qw - cfg.current_quarter_wave_cm) / max(
            cfg.current_quarter_wave_cm, 0.01
        )

        click.echo()
        click.echo(f"  ─ Band: {band.id} ({band.center_hz/1e6:.1f} MHz) ─")

        if delta_pct < SAME_TUNING_TOLERANCE_PCT:
            click.echo(
                f"  Already at this tuning ({cfg.current_quarter_wave_cm:.1f} "
                f"cm covers {target_qw*4*1000/29979.2458:.0f} MHz "
                f"close enough). No retune needed."
            )
            return "go"

        click.echo(
            f"  ⏸ Please retune your telescopic whip on {cfg.dongle_id}:"
        )
        click.echo(
            f"     Set to {target_qw:.1f} cm (quarter-wave at "
            f"{band.center_hz/1e6:.0f} MHz)"
        )
        click.echo(
            f"     If you can't extend that far, the antenna will still "
            f"work but with reduced sensitivity."
        )
        response = click.prompt(
            "  Press Enter when ready, or type 'skip' to skip this band",
            default="", show_default=False,
        ).strip().lower()
        if response in ("skip", "s"):
            click.echo(f"  Skipping {band.id}.")
            return "skip"
        cfg.current_quarter_wave_cm = target_qw
        return "go"

    return callback


def make_after_session_callback(
    cfg: GuidedConfig, config_path: Path,
):
    """Return an async callback for SessionRunner's after_session_hook.
    Prompts the user to restore antenna state or update the config."""

    async def callback(strategy_results) -> None:
        # Compute current state for the prompt
        last_freq = (
            29979.2458 / (cfg.current_quarter_wave_cm * 4)
        )
        current_matches_original = (
            abs(cfg.current_quarter_wave_cm - cfg.original_quarter_wave_cm)
            / max(cfg.original_quarter_wave_cm, 0.01)
            < SAME_TUNING_TOLERANCE_PCT
        )

        click.echo()
        click.echo("─" * 72)
        click.echo(" Restore antenna state")
        click.echo("─" * 72)
        click.echo()
        click.echo(
            f"  Your config has {cfg.dongle_id} → antenna={cfg.antenna_id}"
        )
        click.echo(
            f"  ({cfg.original_quarter_wave_cm:.1f} cm quarter-wave for "
            f"{cfg.original_resonant_freq_hz/1e6:.0f} MHz)."
        )
        if current_matches_original:
            click.echo()
            click.echo(
                f"  Your whip is currently at "
                f"{cfg.current_quarter_wave_cm:.1f} cm — already matches "
                f"the config. Nothing to do."
            )
            return
        click.echo(
            f"  After this scan, your whip is at "
            f"{cfg.current_quarter_wave_cm:.1f} cm "
            f"(set for {last_freq:.0f} MHz)."
        )
        click.echo()
        click.echo("  Options:")
        click.echo(
            f"    1) Reset whip to {cfg.original_quarter_wave_cm:.1f} cm "
            f"to match config (recommended)"
        )
        click.echo(
            f"    2) Update config to reflect a different actual length "
            f"(you tell me cm)"
        )
        click.echo("    3) Skip — config stays as-is, you handle the antenna")
        choice = click.prompt(
            "  >", type=click.IntRange(1, 3), default=1, show_default=True,
        )
        if choice == 1:
            click.echo()
            click.echo(
                f"  Please set your whip back to "
                f"{cfg.original_quarter_wave_cm:.1f} cm."
            )
            click.prompt(
                "  Press Enter when done", default="", show_default=False,
            )
            click.echo("  ✓ Done. Config is consistent with your antenna.")
        elif choice == 2:
            new_length_str = click.prompt(
                "  Enter actual length in cm", type=str,
            ).strip()
            try:
                new_length_cm = float(new_length_str)
                if new_length_cm <= 0 or new_length_cm > 500:
                    raise ValueError("length must be between 0 and 500 cm")
            except ValueError as exc:
                click.echo(
                    f"  Couldn't parse '{new_length_str}': {exc}. "
                    f"Skipping config update."
                )
                return
            new_freq_hz = int((29979.2458 / (new_length_cm * 4)) * 1_000_000)
            click.echo(
                f"  At {new_length_cm:.1f} cm, your whip is tuned to "
                f"{new_freq_hz/1e6:.0f} MHz."
            )
            n_changed = update_antenna_in_config(
                config_path,
                old_antenna_id=cfg.antenna_id,
                dongle_id=cfg.dongle_id,
                new_resonant_hz=new_freq_hz,
                new_quarter_wave_cm=new_length_cm,
            )
            if n_changed:
                click.echo(
                    f"  ✓ Updated site.toml: antenna for {cfg.dongle_id} "
                    f"now reflects {new_freq_hz/1e6:.0f} MHz tuning."
                )
            else:
                click.echo(
                    "  ⚠ Could not write to site.toml. Update manually if needed."
                )
        else:
            click.echo("  OK. Config left as-is.")

    return callback


def update_antenna_in_config(
    config_path: Path,
    *,
    old_antenna_id: str,
    dongle_id: str,
    new_resonant_hz: int,
    new_quarter_wave_cm: float,
) -> int:
    """Update site.toml to reflect a new antenna tuning.

    Creates a new custom antenna stanza for the new resonant frequency
    and updates the dongle's antenna assignment to point to it. Leaves
    the old stanza in place if other dongles still reference it; removes
    it otherwise.
    """
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        import tomli_w
    except ImportError:
        log.warning("tomli/tomli_w not available; can't update config")
        return 0

    if not config_path.exists():
        return 0
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    new_freq_mhz = int(new_resonant_hz / 1_000_000)
    new_id = f"whip_telescopic_{new_freq_mhz}mhz"
    new_stanza = {
        "id": new_id,
        "name": (
            f"Telescopic whip @ {new_quarter_wave_cm:.1f} cm "
            f"(tuned for {new_freq_mhz} MHz)"
        ),
        "antenna_type": "whip",
        "resonant_freq_hz": new_resonant_hz,
        "usable_range": [
            int(new_resonant_hz * 0.5), int(new_resonant_hz * 1.5)
        ],
        "gain_dbi": 2.15,
        "polarization": "vertical",
    }

    # Add new antenna stanza if not already present
    antennas = data.get("antennas", [])
    if not any(a.get("id") == new_id for a in antennas):
        antennas.append(new_stanza)

    # Update the dongle's antenna assignment
    n_changed = 0
    for stanza in data.get("dongles", []):
        if stanza.get("id") == dongle_id:
            stanza["antenna"] = new_id
            n_changed += 1

    if n_changed == 0:
        return 0  # didn't find the dongle; nothing to change

    # Remove the old antenna stanza if no dongle still references it
    still_referenced = any(
        d.get("antenna") == old_antenna_id
        for d in data.get("dongles", [])
    )
    if not still_referenced:
        antennas = [a for a in antennas if a.get("id") != old_antenna_id]

    data["antennas"] = antennas
    config_path.write_text(tomli_w.dumps(data), encoding="utf-8")
    return n_changed
