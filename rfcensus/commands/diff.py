"""`rfcensus diff` — compare two sessions.

Shows what's new, what's gone, and what's changed between two sessions.
Useful for: "what's different from last week" or "what did my trip to
the coffee shop surface that home didn't?"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import click

from rfcensus.commands.base import bootstrap, run_async
from rfcensus.storage.models import DetectionRecord, EmitterRecord
from rfcensus.storage.repositories import (
    DetectionRepo,
    EmitterRepo,
    SessionRepo,
)


@dataclass
class _Diff:
    added: list[EmitterRecord]
    removed: list[EmitterRecord]
    shared: list[tuple[EmitterRecord, EmitterRecord]]  # (prev, curr)
    detections_added: list[DetectionRecord]
    detections_removed: list[DetectionRecord]


@click.command(name="diff")
@click.argument("a_id", type=int)
@click.argument("b_id", type=int)
@click.option("--config", "config_path", type=click.Path(path_type=Path))
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
)
def cli(a_id: int, b_id: int, config_path: Path | None, fmt: str) -> None:
    """Compare two sessions: show emitters and detections added/removed.

    A_ID is treated as the baseline, B_ID is the comparison. Output
    groups into: added (present in B but not A), removed (present in A
    but not B), and shared (in both).
    """
    run_async(_diff(a_id, b_id, config_path, fmt))


async def _diff(a_id: int, b_id: int, config_path: Path | None, fmt: str) -> None:
    rt = await bootstrap(config_path=config_path, detect=False)

    session_repo = SessionRepo(rt.db)
    a = await session_repo.by_id(a_id)
    b = await session_repo.by_id(b_id)
    if a is None or b is None:
        click.echo(
            f"One or both sessions not found (a_id={a_id}, b_id={b_id}). "
            f"Run `rfcensus list sessions`.",
            err=True,
        )
        raise SystemExit(1)

    emitter_repo = EmitterRepo(rt.db)
    detection_repo = DetectionRepo(rt.db)

    a_emitters = await emitter_repo.for_session(a_id)
    b_emitters = await emitter_repo.for_session(b_id)
    a_detections = await detection_repo.for_session(a_id)
    b_detections = await detection_repo.for_session(b_id)

    # Diff emitters by (protocol, device_id_hash) — hashes, not raw IDs
    a_map = {(e.protocol, e.device_id_hash): e for e in a_emitters}
    b_map = {(e.protocol, e.device_id_hash): e for e in b_emitters}

    a_keys = set(a_map.keys())
    b_keys = set(b_map.keys())

    added_keys = b_keys - a_keys
    removed_keys = a_keys - b_keys
    shared_keys = a_keys & b_keys

    diff = _Diff(
        added=[b_map[k] for k in added_keys],
        removed=[a_map[k] for k in removed_keys],
        shared=[(a_map[k], b_map[k]) for k in shared_keys],
        detections_added=_detections_diff(b_detections, a_detections),
        detections_removed=_detections_diff(a_detections, b_detections),
    )

    if fmt == "json":
        _render_json(diff, a_id, b_id)
    else:
        _render_text(diff, a, b)


def _detections_diff(
    in_these: list[DetectionRecord], not_in_those: list[DetectionRecord]
) -> list[DetectionRecord]:
    """Detections in `in_these` but not in `not_in_those`, keyed by (tech, freq_mhz)."""
    other_keys = {(d.technology, d.freq_hz // 1_000_000) for d in not_in_those}
    return [
        d for d in in_these
        if (d.technology, d.freq_hz // 1_000_000) not in other_keys
    ]


def _render_text(diff: _Diff, a, b) -> None:
    click.echo("═" * 72)
    click.echo(f" Session diff: {a.id} ({a.command}) → {b.id} ({b.command})")
    click.echo(f"   A started: {a.started_at.astimezone().strftime('%Y-%m-%d %H:%M')}")
    click.echo(f"   B started: {b.started_at.astimezone().strftime('%Y-%m-%d %H:%M')}")
    click.echo("═" * 72)

    click.echo("")
    click.echo(f"Emitters in B but not A (NEW): {len(diff.added)}")
    for e in sorted(diff.added, key=lambda r: r.confidence, reverse=True)[:20]:
        click.echo(
            f"  + {e.protocol:<20s} hash={e.device_id_hash}  "
            f"conf={e.confidence:.2f}  obs={e.observation_count}"
        )
    if len(diff.added) > 20:
        click.echo(f"    ...and {len(diff.added) - 20} more")

    click.echo("")
    click.echo(f"Emitters in A but not B (GONE): {len(diff.removed)}")
    for e in sorted(diff.removed, key=lambda r: r.confidence, reverse=True)[:20]:
        click.echo(
            f"  - {e.protocol:<20s} hash={e.device_id_hash}  "
            f"conf={e.confidence:.2f}"
        )
    if len(diff.removed) > 20:
        click.echo(f"    ...and {len(diff.removed) - 20} more")

    click.echo("")
    click.echo(f"Emitters in both: {len(diff.shared)}")

    if diff.detections_added or diff.detections_removed:
        click.echo("")
        click.echo("─" * 72)
        click.echo(" Technology detections")
        click.echo("─" * 72)
        for d in diff.detections_added:
            click.echo(
                f"  + {d.technology:<24s} {d.freq_hz/1e6:.3f} MHz "
                f"conf={d.confidence:.2f}"
            )
        for d in diff.detections_removed:
            click.echo(
                f"  - {d.technology:<24s} {d.freq_hz/1e6:.3f} MHz"
            )

    click.echo("")
    click.echo("═" * 72)


def _render_json(diff: _Diff, a_id: int, b_id: int) -> None:
    payload = {
        "a_session": a_id,
        "b_session": b_id,
        "added_emitters": [
            {"protocol": e.protocol, "device_id_hash": e.device_id_hash,
             "confidence": e.confidence, "observation_count": e.observation_count}
            for e in diff.added
        ],
        "removed_emitters": [
            {"protocol": e.protocol, "device_id_hash": e.device_id_hash,
             "confidence": e.confidence, "observation_count": e.observation_count}
            for e in diff.removed
        ],
        "shared_emitters_count": len(diff.shared),
        "detections_added": [
            {"technology": d.technology, "freq_hz": d.freq_hz,
             "confidence": d.confidence, "evidence": d.evidence}
            for d in diff.detections_added
        ],
        "detections_removed": [
            {"technology": d.technology, "freq_hz": d.freq_hz}
            for d in diff.detections_removed
        ],
    }
    click.echo(json.dumps(payload, indent=2))
