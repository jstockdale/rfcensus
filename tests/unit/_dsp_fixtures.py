"""Shared test fixtures for DSP / signal-synthesis tests.

Used by tests that exercise chirp detection, in-window survey, or
LoRa-related DSP code without needing a real SDR. Lives here (not in
a deleted test file) because multiple test modules need the same
synthesis primitives — duplication would invite drift.
"""

from __future__ import annotations

import numpy as np


def synthesize_lora_chirp(
    *,
    sample_rate: int,
    duration_s: float,
    bandwidth_hz: int,
    center_shift_hz: float = 0.0,
    sf: int = 11,
) -> np.ndarray:
    """Generate synthetic LoRa-style chirp bursts the analyzer can detect.

    Each chirp sweeps linearly from -BW/2 to +BW/2 over its duration.
    Between chirps we insert short silent gaps (below the analyzer's
    amplitude threshold) so analyze_chirps() segments the signal into
    individual chirps and fits a linear model to each.

    For SF=11 at 250 kHz BW: one chirp = 8.192 ms. A 0.5 s capture
    holds ~30 full chirps with gaps, plenty for the analyzer to find
    multiple linear segments.
    """
    chirp_duration_s = (2 ** sf) / bandwidth_hz
    slope_hz_per_sec = bandwidth_hz / chirp_duration_s
    gap_duration_s = chirp_duration_s * 0.2  # 20% gap between chirps

    total_samples = int(sample_rate * duration_s)
    signal = np.zeros(total_samples, dtype=np.complex64)

    np.random.seed(42)
    noise_floor = (
        np.random.randn(total_samples) + 1j * np.random.randn(total_samples)
    ).astype(np.complex64) * np.float32(0.02)
    signal += noise_floor

    # Build successive chirps
    chirp_samples = int(sample_rate * chirp_duration_s)
    gap_samples = int(sample_rate * gap_duration_s)
    period = chirp_samples + gap_samples

    idx = 0
    while idx + chirp_samples < total_samples:
        # Linear sweep from -BW/2 + center_shift to +BW/2 + center_shift
        t = np.arange(chirp_samples, dtype=np.float64) / sample_rate
        inst_freq = center_shift_hz + slope_hz_per_sec * t - bandwidth_hz / 2
        phase = 2 * np.pi * np.cumsum(inst_freq) / sample_rate
        burst = np.exp(1j * phase).astype(np.complex64)
        # Superimpose on noise floor (full amplitude during chirp)
        signal[idx : idx + chirp_samples] = burst + noise_floor[
            idx : idx + chirp_samples
        ] * np.float32(0.1)  # weaker noise during chirp
        # Gap region keeps noise floor as-is (below amplitude threshold)
        idx += period

    return signal


# Backward-compat alias for tests that still use the underscore-prefixed
# import name from the deleted test_confirmation_task module.
_synthesize_lora_chirp = synthesize_lora_chirp


def synthesize_realistic_lora(
    *,
    sample_rate: int,
    duration_s: float,
    bandwidth_hz: int,
    sf: int = 9,
    snr_db: float = 20.0,
    num_preamble: int = 8,
    num_data_chirps: int = 20,
    data_symbols: list[int] | None = None,
    packet_start_s: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Generate a realistic gap-free LoRa packet for v0.6.8 dechirp tests.

    Real LoRa packets are *contiguous* chirps with no inter-chirp
    silence — this is the single most important difference from the
    older synthesize_lora_chirp() fixture, which inserts 20% silent
    gaps so analyze_chirps()'s segment finder can fit individual
    chirps.  The reference-dechirp SF classifier relies on the LoRa
    cyclic-shift invariant: any contiguous chirp-length window of an
    upchirp dechirps to a single tone (whose frequency is the symbol
    value), which only holds when chirps butt directly together.

    Packet structure mirrors a real LoRa frame at the modem level:

      • num_preamble back-to-back upchirps (the LoRa preamble that
        receivers use for symbol-timing recovery)
      • num_data_chirps cyclic-shifted upchirps (each shifted by
        data_symbols[i] samples, modeling actual modulated payload)

    Skipped vs the real protocol: SFD downchirps, sync word, header.
    The dechirp classifier only needs SF discrimination, not packet
    boundary recovery, so this simplification is fine.

    SNR is set by scaling AWGN noise relative to a unit-amplitude
    chirp signal. Noise persists across the entire capture; the
    pre/post-packet regions contain noise only (above-noise masking
    in the classifier handles this correctly).
    """
    if sf < 5 or sf > 13:
        raise ValueError(f"SF must be in [5, 13], got {sf}")
    if bandwidth_hz <= 0 or sample_rate <= 0:
        raise ValueError("bandwidth and sample rate must be positive")
    if sample_rate < bandwidth_hz:
        raise ValueError(
            f"sample_rate {sample_rate} below Nyquist for "
            f"bandwidth {bandwidth_hz}"
        )

    chirp_duration_s = (2 ** sf) / bandwidth_hz
    slope_hz_per_sec = bandwidth_hz / chirp_duration_s
    chirp_samples = int(round(sample_rate * chirp_duration_s))
    total_samples = int(sample_rate * duration_s)
    if chirp_samples < 8:
        raise ValueError(
            f"chirp too short ({chirp_samples} samples) — increase "
            f"sample_rate or SF/BW combination"
        )

    rng = np.random.default_rng(seed)

    # AWGN noise sized to give the requested SNR vs unit-amplitude signal
    sig_amp = 1.0
    noise_amp = sig_amp / (10 ** (snr_db / 20.0)) / np.sqrt(2.0)
    noise = (
        rng.standard_normal(total_samples)
        + 1j * rng.standard_normal(total_samples)
    ).astype(np.complex64) * np.float32(noise_amp)
    signal = noise.copy()

    # Build the canonical base upchirp (one full chirp period)
    t = np.arange(chirp_samples, dtype=np.float64) / sample_rate
    inst_freq = slope_hz_per_sec * t - bandwidth_hz / 2.0
    phase = 2.0 * np.pi * np.cumsum(inst_freq) / sample_rate
    base_chirp = np.exp(1j * phase).astype(np.complex64)

    # Place packet
    start = int(sample_rate * packet_start_s)
    total_chirps = num_preamble + num_data_chirps
    end = start + total_chirps * chirp_samples
    if end > total_samples:
        # Truncate gracefully — caller asked for more chirps than fit
        # in the capture; we ship what we can
        end = total_samples
        total_chirps = (end - start) // chirp_samples
        if total_chirps <= 0:
            return signal
        end = start + total_chirps * chirp_samples

    # Default data symbols are 0 (= same as preamble upchirps);
    # generate random ones if the caller wants packet-shape realism
    if data_symbols is None:
        data_symbols = [0] * num_data_chirps

    pieces = [base_chirp] * num_preamble
    n_data_to_emit = max(0, total_chirps - num_preamble)
    for i in range(n_data_to_emit):
        sym = data_symbols[i % len(data_symbols)] % chirp_samples
        # Cyclic-shift the base chirp by `sym` samples (rolling left
        # produces a chirp that starts at frequency
        # +slope*sym/sample_rate - BW/2)
        pieces.append(np.roll(base_chirp, -sym))

    packet = np.concatenate(pieces)[: end - start].astype(np.complex64)
    # Replace noise within packet region with packet + reduced noise
    # (real receivers see the packet at the same noise level as the
    # rest of the band, but for SNR setup it's cleaner to model the
    # packet as drowning out the underlying noise)
    signal[start:end] = packet * sig_amp + noise[start:end] * np.float32(0.1)

    return signal
