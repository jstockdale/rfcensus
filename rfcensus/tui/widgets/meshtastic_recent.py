"""Recent-Meshtastic-decodes widget.

Shows the last ~25 Meshtastic packets seen in the current session as
a compact rolling list:

    21:03:45  913.5  MED_FAST  ✓✓  -65dBm  0x99BC7160→BCAST TEXT 'anyone copy?'
    21:03:42  913.5  MED_FAST  ✓·  -71dBm  [encrypted; ch=0x42 preset=MEDIUM_FAST]
    21:03:38  913.5  MED_FAST  ✓✓  -68dBm  0xBEE97208→BCAST POSITION (24 B)

Columns:
  • time of receipt (HH:MM:SS)
  • freq in MHz
  • preset (truncated)
  • status: ✓✓ = CRC ok and decrypted; ✓· = CRC ok but no PSK; ·· = CRC fail
  • RSSI
  • per-protocol formatter output (the from→to and text/payload preview)

The widget reads ``state.meshtastic_recent`` once per render tick.
Newest entry at the bottom (tail-style), so the stream feels like a
chat log.
"""

from __future__ import annotations

from textual.app import RenderResult
from textual.widget import Widget

from rfcensus.tui.state import MeshtasticDecodeEntry


class MeshtasticRecentWidget(Widget):
    """Live tail of recent Meshtastic decodes."""

    DEFAULT_CSS = """
    MeshtasticRecentWidget {
        background: $background;
        padding: 0 1;
        border: round $surface;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._entries: list[MeshtasticDecodeEntry] = []
        self.border_title = "Meshtastic (recent)"

    def update_entries(
        self, entries: list[MeshtasticDecodeEntry],
    ) -> None:
        """Replace the entries list and refresh ONLY if currently
        visible.

        v0.7.3: previously called ``self.refresh()`` unconditionally,
        which queued render messages even when ``display=False``.
        With four sibling panes in the same Horizontal container
        (EventStream + DongleDetail + PlanTree + this), the rapid
        refresh queue caused the meshtastic widget to briefly bleed
        through other panes after the user toggled it off — it kept
        "fighting the default log" per the bug report. Skip the
        refresh when hidden; the data is still stored in
        ``self._entries`` and will render correctly the next time
        the user toggles the pane back on."""
        self._entries = entries
        if self.display:
            self.refresh()

    def render(self) -> RenderResult:
        if not self._entries:
            # v0.7.4: be specific. Old message said
            #   "ensure decoders.meshtastic.enabled = true in site.toml"
            # which was misleading — `enabled` already defaults to True
            # and the actual gate was the band's suggested_decoders list.
            # New message shows valid TOML the user can paste verbatim
            # and uses the v0.7.4 auto_attach mechanism so the decoder
            # gets wired into all matching bands automatically.
            return (
                "[dim]No Meshtastic packets seen yet.[/dim]\n"
                "\n"
                "[dim]To enable the Meshtastic decoder, add this to "
                "your site.toml:[/dim]\n"
                "\n"
                "[bold]    [decoders.meshtastic][/bold]\n"
                "[bold]    region = \"US\"[/bold]    "
                "[dim]# or NZ_865, EU_868, EU_433, ...[/dim]\n"
                "\n"
                "[dim]That's it — `auto_attach` defaults to true so "
                "the decoder will run on every band whose frequency "
                "overlaps Meshtastic's (902-928 MHz, 868 MHz, "
                "433 MHz). Restart the scan to pick up the change.[/]"
                "\n\n"
                "[dim]For private channels, append a `psks` list:[/]\n"
                "[bold]    psks = [\n"
                "      { name = \"MyChannel\", "
                "psk_b64 = \"...base64...\" },\n"
                "    ][/bold]"
            )

        # Determine viewport height — Widget.size is available after
        # mount; if not yet sized, default to 25 (matches buffer cap).
        try:
            max_lines = max(1, self.size.height - 2)    # account for border
        except Exception:
            max_lines = 25

        visible = self._entries[-max_lines:]

        lines = []
        for entry in visible:
            ts = entry.timestamp.strftime("%H:%M:%S")
            freq_mhz = f"{entry.freq_hz / 1e6:.3f}"
            # Truncate preset to fit. e.g. "MEDIUM_FAST" → "MED_FAST"
            preset = entry.preset.replace("MEDIUM", "MED").replace(
                "LONG", "LNG").replace("SHORT", "SHT")[:10]
            # Status glyphs encode the validation tier at a glance
            if not entry.crc_ok:
                status = "[red]··[/red]"
            elif not entry.decrypted:
                status = "[yellow]✓·[/yellow]"
            else:
                status = "[green]✓✓[/green]"
            rssi = (
                f"{entry.rssi_dbm:>4.0f}dBm"
                if entry.rssi_dbm is not None
                else "  -dBm"
            )
            # The per-protocol summary is the meat of the row; let it
            # be the longest column.
            summary = entry.summary
            lines.append(
                f"[dim]{ts}[/dim] {freq_mhz:>7s}  "
                f"[blue]{preset:<10s}[/blue] {status} {rssi}  "
                f"{summary}"
            )

        # Show how many entries are out of view (above)
        n_hidden = max(0, len(self._entries) - len(visible))
        if n_hidden > 0:
            lines.insert(
                0,
                f"[dim]…{n_hidden} earlier entries scrolled off "
                "(use `rfcensus list decodes` for full history)[/dim]",
            )

        return "\n".join(lines)
