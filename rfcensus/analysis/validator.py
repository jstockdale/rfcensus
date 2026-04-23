"""Decode validator."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rfcensus.config.schema import ValidationConfig
from rfcensus.events import DecodeEvent
from rfcensus.utils.logging import get_logger
from rfcensus.utils.rate_limiter import RateLimiter

log = get_logger(__name__)


@dataclass
class ValidationContext:
    decoder_rates: dict[str, RateLimiter] = field(default_factory=dict)
    suspicious_pattern_res: list[re.Pattern] = field(default_factory=list)


@dataclass
class ValidationResult:
    accept: bool
    reasons: list[str] = field(default_factory=list)
    confidence_delta: float = 0.0


class DecodeValidator:
    """Evaluates DecodeEvents against configurable thresholds and patterns."""

    def __init__(
        self,
        config: ValidationConfig,
        decoder_names: list[str] | None = None,
    ):
        self.config = config
        patterns = [re.compile(p, re.IGNORECASE) for p in config.suspicious_id_patterns]
        self.context = ValidationContext(suspicious_pattern_res=patterns)
        rate = config.max_decodes_per_minute_per_decoder / 60
        burst = config.max_decodes_per_minute_per_decoder
        for name in decoder_names or []:
            self.context.decoder_rates[name] = RateLimiter(rate=rate, burst=burst)

    def _limiter_for(self, decoder_name: str) -> RateLimiter:
        limiter = self.context.decoder_rates.get(decoder_name)
        if limiter is None:
            limiter = RateLimiter(
                rate=self.config.max_decodes_per_minute_per_decoder / 60,
                burst=self.config.max_decodes_per_minute_per_decoder,
            )
            self.context.decoder_rates[decoder_name] = limiter
        return limiter

    def validate(self, event: DecodeEvent) -> ValidationResult:
        reasons: list[str] = []
        confidence_delta = 0.1

        limiter = self._limiter_for(event.decoder_name)
        if not limiter.try_acquire():
            reasons.append(
                f"{event.decoder_name} exceeding {self.config.max_decodes_per_minute_per_decoder}/min"
            )
            return ValidationResult(accept=False, reasons=reasons, confidence_delta=-0.1)

        if event.rssi_dbm is not None:
            if event.rssi_dbm < self.config.min_rssi_dbm:
                reasons.append(f"RSSI {event.rssi_dbm:.1f} below floor")
                return ValidationResult(accept=False, reasons=reasons, confidence_delta=0.0)
            if event.rssi_dbm > self.config.max_rssi_dbm:
                reasons.append(
                    f"RSSI {event.rssi_dbm:.1f} indicates receiver compression"
                )
                return ValidationResult(accept=False, reasons=reasons, confidence_delta=-0.1)

        device_id = _device_id_from_payload(event.payload)
        if device_id:
            for pattern in self.context.suspicious_pattern_res:
                if pattern.search(device_id):
                    reasons.append(f"device ID '{device_id}' matches suspicious pattern")
                    return ValidationResult(
                        accept=False, reasons=reasons, confidence_delta=-0.05
                    )
            stripped = device_id.lower().replace("-", "").replace(":", "")
            if len(stripped) >= 4 and len(set(stripped)) == 1:
                reasons.append(f"device ID '{device_id}' is constant-digit")
                return ValidationResult(
                    accept=False, reasons=reasons, confidence_delta=-0.05
                )

        # SNR: soft. Low SNR scales confidence down but doesn't reject.
        if event.snr_db is not None:
            if event.snr_db < self.config.min_snr_db:
                shortfall = self.config.min_snr_db - event.snr_db
                confidence_delta = max(0.0, 0.1 - 0.03 * shortfall)
                reasons.append(
                    f"SNR {event.snr_db:.1f} below target {self.config.min_snr_db:.1f} "
                    f"(accepted with reduced confidence)"
                )
            elif event.snr_db > 15.0:
                confidence_delta = 0.15

        reasons.append("passed checks")
        return ValidationResult(accept=True, reasons=reasons, confidence_delta=confidence_delta)


def _device_id_from_payload(payload: dict) -> str | None:
    if not payload:
        return None
    if "_device_id" in payload and payload["_device_id"] is not None:
        return str(payload["_device_id"])
    for key in ("id", "ID", "mmsi", "serial", "address", "callsign", "capcode"):
        if key in payload and payload[key] is not None:
            return str(payload[key])
    return None
