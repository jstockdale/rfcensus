"""rtl_433 decoder.

rtl_433 is the workhorse for OOK/FSK consumer protocols: TPMS, weather
stations, security sensors, remote switches, doorbells, keyfobs, and so
on. It supports ~250 distinct protocols out of the box.

We invoke it in JSON-output mode and parse one decode per line. The
reported freq/RSSI/SNR come from `-M level -M time:iso` flags.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from rfcensus.decoders.base import (
    DecoderAvailability,
    DecoderBase,
    DecoderCapabilities,
    DecoderResult,
    DecoderRunSpec,
)
from rfcensus.events import DecodeEvent, EventBus
from rfcensus.hardware.broker import DongleLease
from rfcensus.utils.async_subprocess import (
    BinaryNotFoundError,
    ManagedProcess,
    ProcessConfig,
    which,
)
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


class Rtl433Decoder(DecoderBase):
    capabilities = DecoderCapabilities(
        name="rtl_433",
        protocols=[
            "tpms",
            "weather_station",
            "remote_switch",
            "doorbell",
            "keyfob",
            "security_sensor",
            "interlogix",
            "honeywell",
            "acurite",
            "oregon_scientific",
            "ambient_weather",
            "lacrosse",
            "generic_ook",
        ],
        freq_ranges=(
            (300_000_000, 470_000_000),
            (850_000_000, 960_000_000),
        ),
        min_sample_rate=250_000,
        # 2.4 Msps matches rtlamr's preferred rate, which enables both
        # decoders to share a single rtl_tcp server on the same dongle.
        # rtl_433 works fine at 2.4 Msps (uses min_sample_rate as floor,
        # not ceiling). Modern systems handle the CPU load easily.
        preferred_sample_rate=2_400_000,
        # rtl_433 can open the dongle directly (exclusive) OR connect
        # to a running rtl_tcp server (shared). We default to SHARED
        # so multiple decoders on the same band (e.g. rtl_433 + rtlamr
        # at 915 MHz) can co-exist on a single physical dongle,
        # dramatically improving dongle utilization in multi-decoder
        # setups. The broker transparently spins up rtl_tcp on the
        # first shared lease and shuts it down when the last client
        # releases — the overhead is negligible on localhost.
        requires_exclusive_dongle=False,
        external_binary="rtl_433",
        cpu_cost="cheap",
        description="Decodes ~250 OOK/FSK consumer protocols on ISM bands",
    )

    async def check_available(self) -> DecoderAvailability:
        binary = self.settings.binary or "rtl_433"
        path = which(binary)
        if path is None:
            return DecoderAvailability(
                name=self.name,
                available=False,
                reason=f"{binary} not on PATH",
            )
        return DecoderAvailability(
            name=self.name,
            available=True,
            binary_path=path,
        )

    async def run(self, spec: DecoderRunSpec) -> DecoderResult:
        lease = spec.lease
        freq_hz = spec.freq_hz
        binary = self.settings.binary or "rtl_433"
        args: list[str] = [
            binary,
            "-f", str(freq_hz),
            "-s", str(max(spec.sample_rate, self.capabilities.min_sample_rate)),
            "-F", "json",
            "-M", "level",
            "-M", "time:iso",
        ]
        # Dongle access: prefer rtl_tcp if this lease is shared (multiple
        # decoders on the same dongle); fall back to direct driver_index
        # for exclusive leases. rtl_433's -d arg accepts either form.
        endpoint = lease.endpoint()
        if endpoint is not None:
            host, port = endpoint
            args += ["-d", f"rtl_tcp:{host}:{port}"]
        elif lease.dongle.driver_index is not None:
            args += ["-d", str(lease.dongle.driver_index)]
        if spec.duration_s is not None:
            args += ["-T", str(int(spec.duration_s))]
        # rtl_433 -g sets tuner gain. "0" means auto/AGC; numeric values are
        # in dB. We default to auto unless the caller explicitly overrides.
        if spec.gain and spec.gain != "auto":
            args += ["-g", str(spec.gain)]
        args += list(self.settings.extra_args)

        result = DecoderResult(name=self.name)

        proc = ManagedProcess(
            ProcessConfig(
                name=f"rtl_433[{lease.dongle.id}@{freq_hz}]",
                args=args,
                log_stderr=True,
                stderr_log_level="DEBUG",
            )
        )
        try:
            await proc.start()
        except BinaryNotFoundError as exc:
            log.warning(
                "rtl_433 NOT INSTALLED or not on PATH: %s. "
                "Install via `apt install rtl-433` (Debian/Ubuntu) or "
                "`brew install rtl_433` (macOS), or set the decoder "
                "`binary` setting to an absolute path. Skipping "
                "rtl_433 for this band.",
                exc,
            )
            result.errors.append(str(exc))
            result.ended_reason = "binary_missing"
            return result

        try:
            async for line in proc.stdout_lines():
                if not line.strip():
                    continue
                event = _parse_line(
                    line, freq_hz=freq_hz, dongle_id=lease.dongle.id,
                    session_id=spec.session_id, decoder_name=self.name,
                )
                if event is not None:
                    await spec.event_bus.publish(event)
                    result.decodes_emitted += 1
        finally:
            await proc.stop()
        return result


def _parse_line(
    line: str,
    *,
    freq_hz: int,
    dongle_id: str,
    session_id: int,
    decoder_name: str,
) -> DecodeEvent | None:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        log.debug("rtl_433 non-JSON line: %s", line[:100])
        return None

    # rtl_433 reports frequency in MHz (float) on some versions, Hz on others.
    reported_freq = _extract_freq_hz(data) or freq_hz
    model = data.get("model", "unknown")
    # Synthesize a device ID from whatever identifying fields are present.
    device_id = _extract_device_id(data)
    protocol = _classify_protocol(model, data)

    rssi = _coerce_float(data.get("rssi"))
    snr = _coerce_float(data.get("snr"))
    # rtl_433 also sometimes uses "rssi_dB"; be lenient
    if rssi is None:
        rssi = _coerce_float(data.get("rssi_dB"))

    payload = dict(data)
    payload.setdefault("model", model)
    if device_id is not None:
        payload["_device_id"] = device_id

    return DecodeEvent(
        session_id=session_id,
        decoder_name=decoder_name,
        protocol=protocol,
        dongle_id=dongle_id,
        freq_hz=reported_freq,
        rssi_dbm=rssi,
        snr_db=snr,
        payload=payload,
        timestamp=datetime.now(timezone.utc),
    )


def _extract_freq_hz(data: dict) -> int | None:
    if "freq" in data:
        value = data["freq"]
        if isinstance(value, (int, float)):
            # Usually MHz; convert
            if value < 10_000:
                return int(value * 1_000_000)
            return int(value)
    if "freq_hz" in data:
        try:
            return int(data["freq_hz"])
        except (TypeError, ValueError):
            return None
    return None


def _extract_device_id(data: dict) -> str | None:
    for key in ("id", "ID", "serial", "serial_number", "device", "sid"):
        if key in data and data[key] not in (None, ""):
            return str(data[key])
    return None


def _classify_protocol(model: str, _data: dict) -> str:
    """Map rtl_433's free-form model strings to our rough protocol taxonomy."""
    lowered = model.lower()
    if "tpms" in lowered:
        return "tpms"
    if "interlogix" in lowered:
        return "interlogix_security"
    if "honeywell" in lowered:
        return "honeywell_security"
    if "acurite" in lowered or "oregon" in lowered or "ambient" in lowered or "lacrosse" in lowered:
        return "weather_station"
    if "doorbell" in lowered or "chime" in lowered:
        return "doorbell"
    if "keyfob" in lowered or "remote" in lowered:
        return "keyfob"
    return "generic_ook"


def _coerce_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
