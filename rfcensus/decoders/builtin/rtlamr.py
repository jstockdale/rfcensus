"""rtlamr decoder.

rtlamr decodes Itron ERT compatible utility meter broadcasts: SCM, SCM+,
IDM, NetIDM, and R900 (Neptune water meters). rtlamr connects to a
running `rtl_tcp` server rather than opening the device directly, which
makes it a good candidate for shared-dongle scheduling.
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


class RtlamrDecoder(DecoderBase):
    capabilities = DecoderCapabilities(
        name="rtlamr",
        protocols=["ert_scm", "ert_scm_plus", "ert_idm", "ert_netidm", "r900", "r900_bcd"],
        freq_ranges=((902_000_000, 928_000_000),),
        min_sample_rate=2_048_000,
        preferred_sample_rate=2_400_000,
        requires_exclusive_dongle=False,  # Connects via rtl_tcp
        external_binary="rtlamr",
        cpu_cost="cheap",
        description="Decodes Itron ERT utility meter broadcasts on 900 MHz ISM",
    )

    async def check_available(self) -> DecoderAvailability:
        binary = self.settings.binary or "rtlamr"
        path = which(binary)
        if path is None:
            return DecoderAvailability(
                name=self.name,
                available=False,
                reason=f"{binary} not on PATH (install via `go install github.com/bemasher/rtlamr@latest`)",
            )
        return DecoderAvailability(name=self.name, available=True, binary_path=path)

    async def run(self, spec: DecoderRunSpec) -> DecoderResult:
        result = DecoderResult(name=self.name)
        lease = spec.lease
        endpoint = lease.endpoint()
        if endpoint is None:
            result.errors.append(
                "rtlamr requires a shared (rtl_tcp) dongle lease, but got exclusive"
            )
            result.ended_reason = "wrong_lease_type"
            return result

        host, port = endpoint
        # Note: in v0.5.20+ the broker starts the fanout via
        # `await fanout.start()` which only returns after
        # asyncio.start_server has completed the bind. So by the
        # time the broker returns a lease pointing at host:port,
        # the listener is already accepting. We previously called
        # wait_for_tcp_ready(host, port) here as a belt-and-
        # suspenders probe against an older rtl_tcp spawn race —
        # but that probe opened a TCP connection and closed it,
        # which showed up in fanout logs as a ghost client
        # (0.03s connection, 0 bytes, ended_by=cmd_reader) and
        # was indistinguishable from rtlamr itself fast-exiting.
        # Removed in v0.5.24+; if we see real bind races again,
        # fix them at the broker layer.

        binary = self.settings.binary or "rtlamr"
        # IMPORTANT: rtlamr uses Go's flag package, which has two quirks
        # we MUST work around:
        #   1. Bool flags only accept '-flag' or '-flag=value', NEVER
        #      '-flag value'. The latter parses as: bool flag set to
        #      true (default), then 'value' as a positional arg.
        #   2. flag.Parse() stops at the first non-flag positional. So
        #      if we pass '-unique true', then 'true' becomes a
        #      positional and -centerfreq/-duration after it are
        #      silently ignored — rtlamr falls back to defaults
        #      (centerfreq=912600155, duration=0s/infinite) without
        #      ever telling us. Diagnosed by 0.02s connection then
        #      silent exit, no stderr output. The fix is to use
        #      '-flag=value' syntax everywhere so each token is
        #      unambiguously a flag.
        #
        # Go's flag.Parse() also errors out hard when given an
        # UNKNOWN flag ("flag provided but not defined: -xxx"),
        # printing the usage banner and exiting 2. We learned this
        # the hard way in v0.5.29 by adding -decimation=2 — that
        # flag exists in rtl_sdr/rtl_power but NOT in rtlamr, so
        # every scan immediately bombed with a usage dump.
        # Any flag added here must be verified against `rtlamr -h`.
        #
        # MESSAGE TYPE SELECTION (the v0.5.33 fix):
        #
        # Default is `-msgtype=all`, which expands inside rtlamr to
        # (scm, scm+, idm, r900). This matches v0.5.27's working
        # behavior and gives balanced CPU cost.
        #
        # Per-band override via BandConfig.decoder_options["rtlamr"]
        # lets a band supply a different msgtype list. We use this
        # for 915_ism_r900 (second-pass band centered at 912.6 MHz)
        # which sets msgtype="r900,r900bcd,idm,netidm" to catch the
        # newer protocol variants on that dedicated pass.
        #
        # Why NOT just list all 6 protocols on every call: it turns
        # out r900 and r900bcd's Parse functions do a full filter +
        # quantize pass on every IQ block, independent of whether
        # any preamble matched. Adding r900bcd effectively runs a
        # second r900 DSP pipeline on every block, roughly DOUBLING
        # r900's CPU share. v0.5.31 learned this the hard way —
        # drops jumped from 0.8% to ~53% of the IQ stream. netidm
        # doesn't have this issue (its Parse returns fast when
        # pkts is empty) but including it with r900bcd was enough
        # to push rtlamr past the "can keep up" threshold under
        # normal CPU contention (Chrome, IDE, etc).
        rtlamr_opts = spec.decoder_options.get("rtlamr", {})
        msgtype = rtlamr_opts.get("msgtype", "all")
        args = [
            binary,
            f"-server={host}:{port}",
            f"-msgtype={msgtype}",
            "-format=json",
            "-unique=true",
            f"-centerfreq={spec.freq_hz}",
        ]
        if spec.duration_s is not None:
            # rtlamr takes Go duration strings
            args.append(f"-duration={int(spec.duration_s)}s")
        args += list(self.settings.extra_args)

        proc = ManagedProcess(
            ProcessConfig(
                name=f"rtlamr[{host}:{port}]",
                args=args,
                # rtlamr is terse — it emits little on stderr unless
                # something is wrong. When it DOES print something,
                # we want to see it (wrong tuner type, header parse
                # error, connection drop), since rtlamr silently
                # exiting with 0 decodes is one of our top bugs.
                log_stderr=True,
                stderr_log_level="WARNING",
            )
        )
        try:
            await proc.start()
        except BinaryNotFoundError as exc:
            # Loudly log missing-binary errors. Previously this was
            # silent and the operator had no clue their scan was a
            # no-op for that decoder. rtlamr specifically installs
            # to ~/go/bin/rtlamr via `go install`, which isn't on
            # most systems' default PATH.
            log.warning(
                "rtlamr NOT INSTALLED or not on PATH: %s. "
                "Install with `go install github.com/bemasher/rtlamr@latest` "
                "then either add ~/go/bin to PATH or set the decoder "
                "`binary` setting to the absolute path "
                "(e.g. ~/go/bin/rtlamr). Skipping rtlamr for this band.",
                exc,
            )
            result.errors.append(str(exc))
            result.ended_reason = "binary_missing"
            return result

        # Surface the first few non-JSON stdout lines at WARNING. rtlamr
        # writes its decodes as JSON to stdout, but it also writes any
        # error messages, usage hints, or non-decode chatter to stdout
        # too (Go's default log destination is stderr but some setups
        # mix). Without this, rtlamr exiting cleanly because of an
        # unknown flag (`-unique` was renamed to `-single` in newer
        # builds, for instance) is invisible — exit code 0, no stderr,
        # 0 decodes, no clue why.
        nonjson_logged = 0
        stdout_lines_seen = 0
        try:
            async for line in proc.stdout_lines():
                stdout_lines_seen += 1
                event = _parse_line(
                    line,
                    freq_hz=spec.freq_hz,
                    dongle_id=lease.dongle.id,
                    session_id=spec.session_id,
                    decoder_name=self.name,
                )
                if event is not None:
                    await spec.event_bus.publish(event)
                    result.decodes_emitted += 1
                else:
                    stripped = line.strip()
                    if stripped and nonjson_logged < 5:
                        log.warning(
                            "rtlamr[%s:%d] stdout: %s",
                            host, port, stripped[:200],
                        )
                        nonjson_logged += 1
        finally:
            # Post-mortem runs UNCONDITIONALLY, before AND after
            # proc.stop(), so that even if stop() raises (drain
            # bug, cancellation, etc.) we still see state. Previous
            # versions put this inside `if decodes_emitted == 0`
            # after proc.stop(); user reports showed the warning
            # never fired even on 0-decode fast exits, suggesting
            # an exception path we weren't catching.
            pre_stop_rc = None
            if hasattr(proc, "_proc") and proc._proc is not None:
                pre_stop_rc = proc._proc.returncode
            pre_stop_stderr = getattr(proc, "stderr_lines_logged", "?")
            log.warning(
                "rtlamr[%s:%d] pre-stop: rc=%s stderr_lines=%s "
                "stdout_lines=%d decodes=%d",
                host, port, pre_stop_rc, pre_stop_stderr,
                stdout_lines_seen, result.decodes_emitted,
            )
            try:
                await proc.stop()
            except Exception as stop_exc:
                log.error(
                    "rtlamr[%s:%d] proc.stop() raised: %r",
                    host, port, stop_exc,
                )
            post_stop_rc = None
            if hasattr(proc, "_proc") and proc._proc is not None:
                post_stop_rc = proc._proc.returncode
            post_stop_stderr = getattr(proc, "stderr_lines_logged", "?")
            # Failure mode guide:
            #   1. stderr=0, stdout=0, rc=0   → "silent exit" (rtlamr
            #      never wrote ANYTHING before terminating cleanly;
            #      likely an environment-level issue, try `rtlamr -h`
            #      manually)
            #   2. stderr>0, stdout=0, rc!=0  → "startup failure"
            #      (rtlamr started, logged, then bailed; read stderr
            #      to see why)
            #   3. stderr>0, stdout>0, rc=0   → "ran but no decodes"
            #      (rtlamr worked but found nothing; check antenna /
            #      frequency / environment)
            log.warning(
                "rtlamr[%s:%d] post-mortem: rc=%s stderr_lines=%s "
                "stdout_lines=%d decodes=%d args=%s — if "
                "stderr_lines=0 AND rc=0, rtlamr exited silently; "
                "try running those args manually. "
                "Go rc: 0=clean, 1=log.Fatal, 2=flag.Parse error.",
                host, port, post_stop_rc, post_stop_stderr,
                stdout_lines_seen, result.decodes_emitted,
                " ".join(args),
            )
        return result


def _parse_line(
    line: str,
    *,
    freq_hz: int,
    dongle_id: str,
    session_id: int,
    decoder_name: str,
) -> DecodeEvent | None:
    line = line.strip()
    if not line or not line.startswith("{"):
        return None
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        log.debug("rtlamr non-JSON line: %s", line[:100])
        return None

    message = data.get("Message") or {}
    msg_type = data.get("Type", "")
    device_id = (
        message.get("ID")
        or message.get("ERTSerialNumber")
        or message.get("EndpointID")
        or message.get("Meter ID")
    )
    commodity = message.get("Type") or message.get("ERTType")
    consumption = (
        message.get("Consumption")
        or message.get("LastConsumption")
        or message.get("LastConsumptionCount")
    )

    protocol = _classify_msg_type(msg_type)
    payload = {
        "msg_type": msg_type,
        "commodity": commodity,
        "consumption": consumption,
        "raw": message,
    }
    if device_id is not None:
        payload["_device_id"] = str(device_id)

    return DecodeEvent(
        session_id=session_id,
        decoder_name=decoder_name,
        protocol=protocol,
        dongle_id=dongle_id,
        freq_hz=freq_hz,
        payload=payload,
        timestamp=datetime.now(timezone.utc),
    )


def _classify_msg_type(msg_type: str) -> str:
    lowered = msg_type.lower()
    if lowered == "scm":
        return "ert_scm"
    if "scm+" in lowered or "scm_plus" in lowered:
        return "ert_scm_plus"
    if "idm" in lowered and "net" in lowered:
        return "ert_netidm"
    if "idm" in lowered:
        return "ert_idm"
    if "r900bcd" in lowered or "r900_bcd" in lowered:
        return "r900_bcd"
    if "r900" in lowered:
        return "r900"
    return "ert_generic"
