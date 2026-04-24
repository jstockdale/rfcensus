# rfcensus

A site survey and inventory tool for the RF environment around you.

rfcensus coordinates one or more SDR dongles – RTL-SDR, HackRF, NESDR, etc. – to
discover, characterize, and identify the radio transmitters in your local
environment. Point it at your bands of interest and it will tell you what's
out there, what protocol each transmitter is speaking, and flag signals it
doesn't recognize for you to investigate.

## Status

Alpha. Being built in the open. Expect rough edges, sharp corners, and occasional
bleeding.

## What it does today

• Enumerates RTL-SDR and HackRF hardware on your system
• Runs multiple decoders in parallel across multiple dongles (rtl_433, rtlamr,
  rtl-ais, multimon-ng, direwolf)
• Power scans with rtl_power or hackrf_sweep depending on hardware
• Identifies active channels, classifies them, tracks them over time
• Deduplicates decodes into persistent "emitter" records with confidence scoring
• Flags unknown persistent carriers for manual investigation
• Privacy-preserving by default (device IDs are hashed in reports)

## What it doesn't do yet

• No TUI / web UI (coming next)
• No satellite pass prediction + SatDump hand-off
• No P25 trunk-following (identifies P25 control channels and recommends SDRTrunk)
• No detection-only modules yet (LoRa, P25 CC, TETRA, LTE fingerprinting)

## Quick start

```bash
# Install the Python package
pip install -e .

# Install the external decoder binaries (Debian/Ubuntu)
./scripts/install_decoders.sh
# Or --all to include optional HackRF and direwolf:
./scripts/install_decoders.sh --all

# Set up your site config
rfcensus init

# Verify everything works
rfcensus doctor

# Run a quick inventory
rfcensus inventory --duration 10m

# See what's been identified
rfcensus list emitters
```

## Hardware prerequisites

You need at least one of the following:

• RTL-SDR dongle (V3, V4, NESDR Nano 3, NESDR Smart V5, etc.)
• HackRF One
• Anything that speaks to librtlsdr or hackrf tools

You'll also need the underlying decoder tools installed. The easiest
path is `./scripts/install_decoders.sh` (see above), which handles all
of this. To install manually:

```bash
# Debian/Ubuntu
sudo apt install rtl-433 rtl-sdr multimon-ng
# Optional for more protocols:
sudo apt install direwolf

# rtlamr (Go) — we recommend the jstockdale fork, which carries
# a small r900 performance patch (~35× faster in the no-traffic
# case). The patch is intended for upstream; until then, the fork
# is what rfcensus tests against.
go install github.com/jstockdale/rtlamr@latest
# (upstream github.com/bemasher/rtlamr works too, just slower for
# r900 coverage — see scripts/install_decoders.sh for why)

# rtl-ais – install separately, or build from source:
# https://github.com/dgiardini/rtl-ais

# hackrf tools if using HackRF:
sudo apt install hackrf
```

Run `rfcensus doctor` and it will tell you what's missing.

## Design principles

1. **Progressive disclosure**: Default usage is simple. Expert users can drill in.
2. **Hardware-adaptive**: Does best job possible with whatever hardware is
   available. One RTL-SDR or a rack of gear, both work.
3. **Privacy-preserving by default**: IDs are hashed. Opt in to raw values.
4. **Legally-conservative**: Passive monitoring only. Encrypted protocols are
   identified but not decoded.
5. **Specialization hand-off**: For things that need specialized tooling
   (satellite imagery, trunked P25), we identify and hand off rather than
   reimplement.

## License

BSD-3-Clause. See LICENSE.

Underlying decoder tools have their own licenses (rtl_433 is GPL-2.0, rtlamr is
AGPL-3.0, multimon-ng is GPL-2.0, etc.). rfcensus subprocess-invokes these tools
and does not link against them, so the BSD license applies to rfcensus itself.

## Project

rfcensus is an Off by One project.
