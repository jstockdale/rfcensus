"""`rfcensus monitor` — sustained monitoring of a single band.

Motivation
==========

After an `inventory` scan has surveyed the RF landscape and surfaced
interesting bands (via emitters, detections, and the v0.5.36 mystery-
carriers section), the natural next step is to point a dongle at one
specific band and watch it continuously. Monitor mode is that.

Differences from inventory
==========================

                       inventory          monitor
---------------------  ---------------    ----------------
scope                  many bands         exactly one band
wave structure         multiple waves     one wave, one task
duration               finite (30m def.)  infinite (until Ctrl-C)
output                 final report       live stream to stdout
DB writes              always             opt-in via --save

Stream-only by default
======================

We don't write monitor-session data to the SQLite DB by default. Two
reasons:

  1. Monitor sessions can run for hours or days and would bloat the
     DB with repetitive activity from the same few frequencies.

  2. Inventory-session statistics (most-active bands, emitter-per-band
     counts, etc.) are computed against the DB. A 72-hour monitor run
     dumping 100,000 decodes of a single LoRaWAN gateway would skew
     those stats heavily if commingled. Segregating monitor data by
     default avoids the skew; users who DO want monitor data persisted
     (e.g., for longitudinal analysis of a specific gateway) opt in
     with --save, and we tag the session with command="monitor" so
     downstream analysis can filter.

For --save=false runs, we use an in-memory SQLite database. All the
normal event → repo → writer plumbing still runs; it just writes to
memory and the data vanishes when the process exits. Keeps the code
path identical to --save=true and avoids special casing.

Output format
=============

`--format text` (default): human-readable, one line per decode.
`--format json`: JSON Lines (one JSON object per decode) — pipe this
into jq / a log aggregator / another tool. Useful for scripting.

Usage examples
==============

    # Monitor 915 MHz ISM activity with all applicable decoders
    rfcensus monitor --band 915_ism

    # Monitor just rtl_433 on interlogix_security
    rfcensus monitor --band interlogix_security --decoder rtl_433

    # Monitor with a specific dongle, duration-limited, saving to DB
    rfcensus monitor --band aprs_2m --dongle 07262454 \\
        --duration 1h --save

    # Machine-readable output for piping
    rfcensus monitor --band pocsag_929 --format json | jq '.payload'
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import click

from rfcensus.commands.base import bootstrap, run_async
from rfcensus.commands.inventory import _parse_duration, _parse_gain
from rfcensus.engine.session import SessionRunner
from rfcensus.events import DecodeEvent, DetectionEvent, WideChannelEvent
from rfcensus.storage.db import Database
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


def _build_cli():
    @click.command(
        name="monitor",
        help=(
            "Continuously monitor one band. Streams decodes live to stdout; "
            "optionally saves to the database. Stops on Ctrl-C."
        ),
    )
    @click.option(
        "--config",
        "config_path",
        type=click.Path(path_type=Path),
        help="Path to site.toml (defaults to standard location).",
    )
    @click.option(
        "--band",
        required=True,
        help="Single band ID to monitor (e.g. '915_ism', 'aprs_2m').",
    )
    @click.option(
        "--decoder",
        default=None,
        help=(
            "Optional: restrict to a specific decoder (e.g. 'rtl_433', "
            "'direwolf'). Default: run all decoders that match the band."
        ),
    )
    @click.option(
        "--dongle",
        default=None,
        help=(
            "Optional: pin to a specific dongle by serial/id. Default: let "
            "the broker pick any suitable dongle."
        ),
    )
    @click.option(
        "--duration",
        default="forever",
        help=(
            "Run duration (e.g. '30m', '2h', '600'). Default 'forever' "
            "runs until Ctrl-C."
        ),
    )
    @click.option(
        "--save/--no-save",
        default=False,
        help=(
            "Persist decodes and detections to the DB (tagged with "
            "command='monitor'). Default: stream-only (no DB writes) "
            "to avoid bloating stats for inventory-style queries."
        ),
    )
    @click.option(
        "--format",
        "output_format",
        type=click.Choice(["text", "json"], case_sensitive=False),
        default="text",
        help=(
            "Output format for live decodes. 'text' is human-readable; "
            "'json' emits one JSON object per line (good for piping into "
            "jq or log aggregators)."
        ),
    )
    @click.option(
        "--gain",
        default="auto",
        help="Tuner gain: 'auto' or a dB value (0-50). Default: auto.",
    )
    def cli(
        config_path: Path | None,
        band: str,
        decoder: str | None,
        dongle: str | None,
        duration: str,
        save: bool,
        output_format: str,
        gain: str,
    ) -> None:
        run_async(
            _run(
                config_path=config_path,
                band_id=band,
                decoder_name=decoder,
                dongle_id=dongle,
                duration=duration,
                save=save,
                output_format=output_format.lower(),
                gain=gain,
            )
        )

    return cli


async def _run(
    *,
    config_path: Path | None,
    band_id: str,
    decoder_name: str | None,
    dongle_id: str | None,
    duration: str,
    save: bool,
    output_format: str,
    gain: str,
) -> None:
    duration_s = _parse_duration(duration, default_s=0.0)
    indefinite = duration_s == 0.0
    gain_normalized = _parse_gain(gain)

    rt = await bootstrap(config_path=config_path, detect=True)

    # Restrict config to the single requested band
    _validate_and_restrict_band(rt.config, band_id)

    # If --decoder, disable all others. Otherwise keep defaults.
    if decoder_name is not None:
        _restrict_to_decoder(rt.config, decoder_name)

    # If --dongle specified, filter the registry to just that one
    if dongle_id is not None:
        _filter_to_dongle(rt.registry, dongle_id)

    if not rt.registry.usable():
        click.echo(
            "No usable dongles for this monitor session. Run "
            "`rfcensus doctor` for diagnostics.",
            err=True,
        )
        raise SystemExit(1)

    # Substitute in-memory DB if not saving. The session still goes
    # through the normal repository + writer path — it just writes to
    # a DB that gets discarded when the process exits.
    if not save:
        rt.db = Database(":memory:")
        log.info(
            "monitor mode: --no-save, using in-memory DB (nothing persisted)"
        )
    else:
        log.info(
            "monitor mode: --save, persisting to DB at %s with "
            "command='monitor' tag",
            rt.db.path,
        )

    runner = SessionRunner(
        command="monitor",
        config=rt.config,
        registry=rt.registry,
        db=rt.db,
        duration_s=duration_s,
        indefinite=indefinite,
        gain=gain_normalized,
        skip_health_check=False,
    )

    # Attach live output printers BEFORE runner.run() so we catch the
    # first events. Runner populates event_bus in __init__.
    _attach_printers(runner, output_format=output_format)

    header = (
        f"Monitoring band '{band_id}'"
        + (f" with decoder '{decoder_name}'" if decoder_name else "")
        + (f" on dongle '{dongle_id}'" if dongle_id else "")
        + (f" for {duration_s:.0f}s" if not indefinite else " indefinitely")
        + (" (saving to DB)" if save else " (stream-only, --no-save)")
        + ".  Ctrl-C to stop."
    )
    click.echo(header, err=True)

    try:
        await runner.run()
    finally:
        # Close in-memory DB explicitly so sqlite frees its pages.
        # File DBs use the global singleton and shouldn't be closed here.
        if not save and hasattr(rt.db, "close"):
            try:
                rt.db.close()
            except Exception:
                pass


# ------------------------------------------------------------------
# Validation helpers (pure; easy to unit test)
# ------------------------------------------------------------------


def _validate_and_restrict_band(config, band_id: str) -> None:
    """Ensure band_id is known; restrict config.bands.enabled to just
    this one band. Raises SystemExit(2) with a helpful error if the
    band is unknown.

    Mutates `config` in place — the monitor session will only plan
    this single band.
    """
    known_ids = {b.id for b in config.band_definitions}
    if band_id not in known_ids:
        known = ", ".join(sorted(known_ids))
        click.echo(
            f"Unknown band '{band_id}'. Known bands: {known}", err=True
        )
        raise SystemExit(2)
    config.bands.enabled = [band_id]


def _restrict_to_decoder(config, decoder_name: str) -> None:
    """Disable every decoder in the site config except the chosen one.
    Validates that the decoder name is registered. Raises SystemExit(2)
    on unknown name.

    Decoders not listed in site.toml are enabled by default; we have
    to explicitly write entries to disable them. This mutation only
    affects the in-memory config for this session — site.toml on disk
    is unchanged.
    """
    from rfcensus.decoders.registry import get_registry

    reg = get_registry()
    if reg.get(decoder_name) is None:
        available = ", ".join(sorted(reg.names()))
        click.echo(
            f"Unknown decoder '{decoder_name}'. Available: {available}",
            err=True,
        )
        raise SystemExit(2)
    for name in reg.names():
        dec = config.decoders.setdefault(
            name, _new_decoder_config(enabled=True)
        )
        dec.enabled = (name == decoder_name)


def _filter_to_dongle(registry, dongle_id: str) -> None:
    """Restrict registry to just one dongle (matched by id or serial).
    Raises SystemExit(2) if no match.

    Simpler than marking the others unavailable because the broker
    queries registry.dongles directly. Keeps the session's hardware
    view narrow and deterministic.
    """
    matches = [
        d for d in registry.dongles
        if d.id == dongle_id or d.serial == dongle_id
    ]
    if not matches:
        found = ", ".join(sorted(d.id for d in registry.dongles))
        click.echo(
            f"No dongle matches '{dongle_id}'. Detected: {found}",
            err=True,
        )
        raise SystemExit(2)
    registry.dongles = matches


# ------------------------------------------------------------------
# Output printers
# ------------------------------------------------------------------


def _attach_printers(runner: SessionRunner, *, output_format: str) -> None:
    """Subscribe live decoders + detection printers to the bus."""

    if output_format == "json":
        printer = _JsonLinesPrinter()
    else:
        printer = _TextPrinter()

    async def _on_decode(event: DecodeEvent) -> None:
        printer.print_decode(event)

    async def _on_detection(event: DetectionEvent) -> None:
        printer.print_detection(event)

    async def _on_wide_channel(event: WideChannelEvent) -> None:
        printer.print_wide_channel(event)

    runner.event_bus.subscribe(DecodeEvent, _on_decode)
    runner.event_bus.subscribe(DetectionEvent, _on_detection)
    runner.event_bus.subscribe(WideChannelEvent, _on_wide_channel)


class _TextPrinter:
    """Human-readable single-line format for live events."""

    def print_decode(self, event: DecodeEvent) -> None:
        ts = event.timestamp.astimezone().strftime("%H:%M:%S")
        freq_mhz = event.freq_hz / 1e6
        rssi = (
            f"{event.rssi_dbm:.1f}dBm"
            if event.rssi_dbm is not None else "----"
        )
        payload_summary = _summarize_payload(event.payload)
        click.echo(
            f"{ts} decode  {event.protocol:<14s} {freq_mhz:8.3f} MHz "
            f"{rssi:<10s} {payload_summary}"
        )

    def print_detection(self, event: DetectionEvent) -> None:
        ts = event.timestamp.astimezone().strftime("%H:%M:%S")
        freq_mhz = event.freq_hz / 1e6
        bw_khz = (event.bandwidth_hz or 0) // 1000
        click.echo(
            f"{ts} detect  {event.technology:<14s} {freq_mhz:8.3f} MHz "
            f"bw={bw_khz}kHz conf={event.confidence:.2f} -- {event.evidence}"
        )

    def print_wide_channel(self, event: WideChannelEvent) -> None:
        ts = event.timestamp.astimezone().strftime("%H:%M:%S")
        freq_mhz = event.freq_center_hz / 1e6
        bw_khz = event.bandwidth_hz // 1000
        tmpl_khz = event.matched_template_hz // 1000
        click.echo(
            f"{ts} wide    template={tmpl_khz}kHz "
            f"{freq_mhz:8.3f} MHz span={bw_khz}kHz "
            f"bins={event.constituent_bin_count} "
            f"coverage={event.coverage_ratio*100:.0f}%"
        )


class _JsonLinesPrinter:
    """JSON Lines format — one JSON object per event, no indentation.
    Useful for piping into jq or for log aggregators that parse JSON."""

    def print_decode(self, event: DecodeEvent) -> None:
        self._emit("decode", {
            "protocol": event.protocol,
            "freq_hz": event.freq_hz,
            "rssi_dbm": event.rssi_dbm,
            "dongle_id": event.dongle_id,
            "decoder": event.decoder_name,
            "payload": event.payload,
            "timestamp": event.timestamp.isoformat(),
        })

    def print_detection(self, event: DetectionEvent) -> None:
        self._emit("detection", {
            "detector": event.detector_name,
            "technology": event.technology,
            "freq_hz": event.freq_hz,
            "bandwidth_hz": event.bandwidth_hz,
            "confidence": event.confidence,
            "evidence": event.evidence,
            "metadata": event.metadata,
            "timestamp": event.timestamp.isoformat(),
        })

    def print_wide_channel(self, event: WideChannelEvent) -> None:
        self._emit("wide_channel", {
            "freq_center_hz": event.freq_center_hz,
            "bandwidth_hz": event.bandwidth_hz,
            "matched_template_hz": event.matched_template_hz,
            "constituent_bin_count": event.constituent_bin_count,
            "coverage_ratio": event.coverage_ratio,
            "peak_power_dbm": event.peak_power_dbm,
            "avg_power_dbm": event.avg_power_dbm,
            "timestamp": event.timestamp.isoformat(),
        })

    def _emit(self, kind: str, payload: dict) -> None:
        payload = {"kind": kind, **payload}
        sys.stdout.write(
            json.dumps(payload, default=_json_fallback) + "\n"
        )
        sys.stdout.flush()


def _summarize_payload(payload) -> str:
    """One-line summary of a decode payload for the text formatter."""
    if not payload:
        return ""
    if isinstance(payload, dict):
        # Heuristic: show _device_id first if present, then first 2-3
        # other key=value pairs.
        parts = []
        device_id = payload.get("_device_id")
        if device_id is not None:
            parts.append(f"id={device_id}")
        for k, v in list(payload.items())[:4]:
            if k.startswith("_") or k == "message":
                continue
            parts.append(f"{k}={v}")
        message = payload.get("message")
        if message:
            msg_str = str(message)
            if len(msg_str) > 40:
                msg_str = msg_str[:37] + "..."
            parts.append(f'msg="{msg_str}"')
        return " ".join(parts)
    return str(payload)


def _json_fallback(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__dict__"):
        return asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj.__dict__
    return str(obj)


def _new_decoder_config(*, enabled: bool):
    """Create a fresh DecoderConfig with given enabled state. Imported
    inside the function to avoid a module-level import of a type that
    may be moved later."""
    from rfcensus.config.schema import DecoderConfig
    return DecoderConfig(enabled=enabled)


# Public entrypoint for registering with the top-level CLI group
monitor_cli = _build_cli()
