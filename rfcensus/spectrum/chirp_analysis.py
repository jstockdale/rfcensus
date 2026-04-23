"""Chirp detection via instantaneous frequency analysis.

LoRa (and other chirp-spread-spectrum protocols) sweeps the
instantaneous frequency linearly through time. A "chirp" is a tone
whose frequency moves in a straight line in the time-frequency plane.

Our detector:

1. Compute instantaneous phase: `angle(samples)`
2. Compute instantaneous frequency: `diff(unwrap(phase)) * sample_rate / (2π)`
3. Look for segments where inst. frequency changes approximately linearly
4. Return confidence [0, 1] that chirp-like behavior is present

Not a full LoRa demodulator — doesn't decode symbols, doesn't identify
spreading factor. Just "this looks chirp-shaped," which is enough
confirmation that a heuristic LoRa suspicion is probably real.

Cheap: ~1 ms of numpy per 500 ms of IQ at 1 MHz sample rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChirpAnalysis:
    chirp_confidence: float  # 0..1
    num_chirp_segments: int
    mean_segment_length_samples: float
    mean_slope_hz_per_sec: float
    reasoning: str = ""


def analyze_chirps(
    samples: np.ndarray,
    sample_rate: int,
    *,
    min_segment_samples: int = 64,
    linearity_threshold: float = 0.85,
) -> ChirpAnalysis:
    """Analyze an IQ chunk for chirp-like patterns."""
    if samples.size < min_segment_samples * 2:
        return ChirpAnalysis(0.0, 0, 0.0, 0.0, "insufficient samples")

    phase = np.angle(samples)
    dphase = np.diff(np.unwrap(phase))
    inst_freq = dphase * sample_rate / (2 * np.pi)

    amplitude = np.abs(samples[1:])
    amp_threshold = max(1e-3, float(np.median(amplitude)) * 0.5)
    mask = amplitude > amp_threshold

    segments: list[tuple[int, int]] = []
    in_run = False
    run_start = 0
    for i, m in enumerate(mask):
        if m and not in_run:
            in_run = True
            run_start = i
        elif not m and in_run:
            in_run = False
            if i - run_start >= min_segment_samples:
                segments.append((run_start, i))
    if in_run and len(mask) - run_start >= min_segment_samples:
        segments.append((run_start, len(mask)))

    if not segments:
        return ChirpAnalysis(0.0, 0, 0.0, 0.0, "no above-noise segments")

    chirp_segments = 0
    slopes: list[float] = []
    lengths: list[int] = []

    for start, end in segments:
        freqs = inst_freq[start:end]
        if freqs.size < min_segment_samples:
            continue
        freqs = _median_filter(freqs, 5)
        t = np.arange(freqs.size, dtype=np.float32)
        a, b = np.polyfit(t, freqs, 1)
        predicted = a * t + b
        ss_res = float(np.sum((freqs - predicted) ** 2))
        ss_tot = float(np.sum((freqs - np.mean(freqs)) ** 2))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        slope_hz_per_sec = float(a * sample_rate)
        freq_span = abs(slope_hz_per_sec) * (freqs.size / sample_rate)
        if r_squared > linearity_threshold and freq_span > 5_000:
            chirp_segments += 1
            slopes.append(slope_hz_per_sec)
            lengths.append(freqs.size)

    if chirp_segments == 0:
        return ChirpAnalysis(
            0.0, 0, 0.0, 0.0,
            f"{len(segments)} candidate segments, none were chirp-linear",
        )

    mean_len = float(np.mean(lengths))
    mean_slope = float(np.mean(np.abs(slopes)))
    confidence = min(1.0, 0.4 + 0.15 * chirp_segments)
    return ChirpAnalysis(
        chirp_confidence=confidence,
        num_chirp_segments=chirp_segments,
        mean_segment_length_samples=mean_len,
        mean_slope_hz_per_sec=mean_slope,
        reasoning=(
            f"{chirp_segments} linear chirp segment(s), mean slope {mean_slope/1e3:.1f} kHz/s"
        ),
    )


def _median_filter(arr: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return arr
    half = kernel // 2
    padded = np.pad(arr, half, mode="edge")
    out = np.empty_like(arr)
    for i in range(arr.size):
        out[i] = np.median(padded[i : i + kernel])
    return out
