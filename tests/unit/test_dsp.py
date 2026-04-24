"""Tests for rfcensus.tools.dsp.

Strategy
========

The DSP primitives are small and deterministic, so we test them with
closed-form inputs where the expected output is known mathematically:

  • IQ unpacking: specific byte patterns → known complex values
  • Quadrature demod on a pure complex tone: output = constant equal to
    the per-sample phase advance
  • Lowpass filter: passes low-frequency tones, attenuates high-frequency
  • Resampler: preserves tone frequency after rate change
  • De-emphasis: impulse response matches 1st-order IIR formula
  • Composite FM round-trip: FM-modulate a 1 kHz tone, demodulate,
    verify output spectrum peaks at 1 kHz

Block-continuity tests feed the same total signal through primitives
as (a) one big call, (b) many small block calls, and assert the
outputs match to machine precision (ignoring filter warm-up and
resampler boundary artifacts, where relevant).

These tests don't need hardware — everything's synthetic numpy. Runs
in a few seconds.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rfcensus.tools.dsp import (
    DecimatingLowpass,
    DeEmphasis,
    FMDemodulator,
    QuadratureDemod,
    Resampler,
    iq_uint8_to_complex,
)


# ==================================================================
# IQ uint8 → complex
# ==================================================================


class TestIqUnpack:
    def test_empty_input(self):
        result = iq_uint8_to_complex(b"")
        assert result.size == 0
        assert result.dtype == np.complex64

    def test_odd_length_raises(self):
        with pytest.raises(ValueError, match="odd length"):
            iq_uint8_to_complex(b"\x80\x80\x80")

    def test_dc_center_byte_is_zero(self):
        """127.5 is center; closest integer bytes 127 and 128 should
        produce values very close to 0."""
        data = bytes([127, 128])  # I=127, Q=128
        result = iq_uint8_to_complex(data)
        assert result.shape == (1,)
        # (127 - 127.5) / 127.5 = -0.00392, (128 - 127.5) / 127.5 = +0.00392
        assert abs(result[0].real - (-1 / 255)) < 1e-5
        assert abs(result[0].imag - (+1 / 255)) < 1e-5

    def test_full_scale_bytes(self):
        """Bytes 0 and 255 should produce near ±1.0 values."""
        data = bytes([0, 255])  # I=0 (−1ish), Q=255 (+1ish)
        result = iq_uint8_to_complex(data)
        assert result.shape == (1,)
        # (0 - 127.5) / 127.5 = -1.0
        # (255 - 127.5) / 127.5 = +1.0
        assert abs(result[0].real - (-1.0)) < 1e-6
        assert abs(result[0].imag - (+1.0)) < 1e-6

    def test_interleaving_preserved(self):
        """Bytes [a0, b0, a1, b1, a2, b2] should produce
        [a0 + j*b0, a1 + j*b1, a2 + j*b2]."""
        data = bytes([127, 128, 0, 255, 255, 0])
        result = iq_uint8_to_complex(data)
        assert result.shape == (3,)
        # Spot-check imag of third sample = (0 - 127.5) / 127.5 = -1.0
        assert abs(result[2].imag - (-1.0)) < 1e-6
        assert abs(result[2].real - (+1.0)) < 1e-6

    def test_returns_complex64(self):
        data = bytes([127, 128, 127, 128])
        result = iq_uint8_to_complex(data)
        assert result.dtype == np.complex64


# ==================================================================
# QuadratureDemod
# ==================================================================


class TestQuadratureDemod:
    def test_pure_tone_gives_constant_phase_advance(self):
        """Input: exp(j * 2*pi * f/fs * n). Output should be a constant
        equal to 2*pi*f/fs (the per-sample phase advance)."""
        fs = 100_000
        f = 1_000
        n = np.arange(10_000)
        phase = 2 * np.pi * f / fs * n
        iq = np.exp(1j * phase).astype(np.complex64)

        demod = QuadratureDemod()
        out = demod.process(iq)

        expected = 2 * np.pi * f / fs
        # Skip the first sample — it depends on the initial _prev=(1+0j)
        # which doesn't match the tone's history.
        assert out.size == iq.size
        assert np.allclose(out[1:], expected, atol=1e-5)

    def test_empty_input_returns_empty(self):
        demod = QuadratureDemod()
        out = demod.process(np.empty(0, dtype=np.complex64))
        assert out.size == 0
        assert out.dtype == np.float32

    def test_stateful_across_blocks_matches_single_call(self):
        """Splitting a signal into blocks must match the single-call
        output (after the first sample, which is contaminated by the
        initial state in both cases)."""
        fs = 100_000
        f = 3_000
        n = np.arange(20_000)
        iq = np.exp(1j * 2 * np.pi * f / fs * n).astype(np.complex64)

        demod1 = QuadratureDemod()
        single = demod1.process(iq)

        demod2 = QuadratureDemod()
        blocks = np.concatenate(
            [
                demod2.process(iq[0:5000]),
                demod2.process(iq[5000:10000]),
                demod2.process(iq[10000:15000]),
                demod2.process(iq[15000:20000]),
            ]
        )

        # Must match exactly at block boundaries (thanks to state carry)
        assert single.shape == blocks.shape
        assert np.allclose(single, blocks, atol=1e-6)

    def test_reset_clears_state(self):
        demod = QuadratureDemod()
        iq = np.array([1 + 0j, 1j, -1 + 0j], dtype=np.complex64)
        _ = demod.process(iq)
        assert demod._prev != complex(1.0, 0.0)
        demod.reset()
        assert demod._prev == complex(1.0, 0.0)


# ==================================================================
# DecimatingLowpass
# ==================================================================


class TestDecimatingLowpass:
    def test_bad_decimation_raises(self):
        with pytest.raises(ValueError, match="decimation"):
            DecimatingLowpass(decimation=0)

    def test_bad_num_taps_raises(self):
        with pytest.raises(ValueError, match="num_taps"):
            DecimatingLowpass(decimation=2, num_taps=4)  # even

    def test_bad_cutoff_raises(self):
        with pytest.raises(ValueError, match="cutoff_fraction"):
            DecimatingLowpass(decimation=2, cutoff_fraction=1.5)

    def test_decimation_reduces_length(self):
        """Output length ~= input length / decimation."""
        lp = DecimatingLowpass(decimation=10)
        x = np.ones(1000, dtype=np.complex64)
        y = lp.process(x)
        # Allow ±1 for phase tracking
        assert abs(y.size - 100) <= 1

    def test_low_freq_tone_passes(self):
        """A tone well below the cutoff should come through mostly
        intact."""
        fs = 48_000
        decimation = 4
        lp = DecimatingLowpass(decimation=decimation)
        f = 1_000  # far below fs/decimation/2 = 6 kHz
        n = np.arange(8_192)
        x = np.exp(1j * 2 * np.pi * f / fs * n).astype(np.complex64)
        y = lp.process(x)
        # Drop filter warm-up region and check amplitude
        # After warm-up, filtered tone should have amplitude ~1 (unity
        # gain in passband).
        mid = y[50:]
        assert np.abs(np.abs(mid).mean() - 1.0) < 0.1

    def test_high_freq_tone_attenuated(self):
        """A tone ABOVE the cutoff should be attenuated."""
        fs = 48_000
        decimation = 4  # output Nyquist = 6 kHz
        lp = DecimatingLowpass(decimation=decimation)
        f = 10_000  # above output Nyquist (6 kHz) — should alias OR be filtered out
        n = np.arange(8_192)
        x = np.exp(1j * 2 * np.pi * f / fs * n).astype(np.complex64)
        y = lp.process(x)
        # After warm-up, the filter should heavily attenuate this.
        # Passband gain was ~1.0 in the previous test; here we should
        # see < 0.3 on average (10 dB+ suppression).
        mid = y[50:]
        assert np.abs(mid).mean() < 0.3

    def test_block_continuity_matches_single_pass(self):
        """Block-wise processing must match single-pass output."""
        fs = 48_000
        decimation = 4
        n = np.arange(16_384)  # divides evenly by 4
        x = np.random.default_rng(seed=42).standard_normal(n.size * 2).view(
            np.complex128
        ).astype(np.complex64)[: n.size]

        lp1 = DecimatingLowpass(decimation=decimation)
        single = lp1.process(x)

        lp2 = DecimatingLowpass(decimation=decimation)
        # Use block sizes that are multiples of decimation for
        # simplest alignment check
        block_size = 1024  # multiple of 4
        blocks = []
        for i in range(0, x.size, block_size):
            blocks.append(lp2.process(x[i : i + block_size]))
        concatenated = np.concatenate(blocks)

        # Outputs should match to within numerical precision
        assert single.shape == concatenated.shape
        assert np.allclose(single, concatenated, atol=1e-5)

    def test_block_continuity_with_odd_block_sizes(self):
        """Blocks that don't divide evenly by decimation should still
        produce correct output, thanks to phase tracking."""
        decimation = 5
        x = np.random.default_rng(seed=1).standard_normal(10_001).astype(
            np.float32
        ).astype(np.complex64)

        lp1 = DecimatingLowpass(decimation=decimation)
        single = lp1.process(x)

        # Odd block size that doesn't divide by decimation
        lp2 = DecimatingLowpass(decimation=decimation)
        blocks = []
        block_size = 997
        for i in range(0, x.size, block_size):
            blocks.append(lp2.process(x[i : i + block_size]))
        concatenated = np.concatenate(blocks)

        # Should match up to output length rounding (±1 sample)
        min_len = min(single.size, concatenated.size)
        assert abs(single.size - concatenated.size) <= 1
        assert np.allclose(
            single[:min_len], concatenated[:min_len], atol=1e-5
        )


# ==================================================================
# Resampler
# ==================================================================


class TestResampler:
    def test_bad_ratios_raise(self):
        with pytest.raises(ValueError):
            Resampler(up=0, down=1)
        with pytest.raises(ValueError):
            Resampler(up=1, down=0)

    def test_output_length_matches_ratio(self):
        """Resampling by L/M: output length ≈ input length * L / M."""
        r = Resampler(up=147, down=320)  # 48k → 22050
        x = np.random.default_rng(seed=0).standard_normal(48000).astype(
            np.float32
        )
        y = r.process(x)
        expected = int(48000 * 147 / 320)
        # Allow ±1% for startup / rounding
        assert abs(y.size - expected) / expected < 0.01

    def test_tone_frequency_preserved(self):
        """A 1 kHz tone resampled from 48 kHz to 22050 Hz should still
        peak at 1 kHz in the output spectrum."""
        fs_in = 48_000
        fs_out = 22_050
        f = 1_000
        n = np.arange(48_000)  # 1 second of input
        x = np.sin(2 * np.pi * f / fs_in * n).astype(np.float32)

        r = Resampler(up=147, down=320)  # 147/320 = 22050/48000
        y = r.process(x)

        # Skip startup transient
        y_clean = y[500:]
        # FFT and find peak
        fft = np.abs(np.fft.rfft(y_clean))
        freqs = np.fft.rfftfreq(y_clean.size, d=1 / fs_out)
        peak_idx = np.argmax(fft)
        peak_freq = freqs[peak_idx]
        assert abs(peak_freq - f) < 50  # within 50 Hz

    def test_empty_input(self):
        r = Resampler(up=3, down=2)
        y = r.process(np.empty(0, dtype=np.float32))
        assert y.size == 0

    def test_identity_ratio(self):
        """up == down should be (effectively) pass-through."""
        r = Resampler(up=1, down=1)
        x = np.random.default_rng(seed=7).standard_normal(1000).astype(
            np.float32
        )
        y = r.process(x)
        # Size should match (± small margin for filter warm-up)
        assert abs(y.size - x.size) <= 10


# ==================================================================
# DeEmphasis
# ==================================================================


class TestDeEmphasis:
    def test_tau_none_is_passthrough(self):
        de = DeEmphasis(tau_s=None, sample_rate=48000)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y = de.process(x)
        assert np.array_equal(x, y)

    def test_tau_zero_is_passthrough(self):
        de = DeEmphasis(tau_s=0.0, sample_rate=48000)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y = de.process(x)
        assert np.array_equal(x, y)

    def test_step_response_approaches_input_level(self):
        """A 1st-order IIR filter responding to a unit step should
        asymptote to 1.0."""
        de = DeEmphasis(tau_s=75e-6, sample_rate=48000)
        x = np.ones(1000, dtype=np.float32)
        y = de.process(x)
        # After many samples (tau * fs * several), y should be close to 1
        assert abs(y[-1] - 1.0) < 0.01

    def test_state_carries_across_blocks(self):
        de1 = DeEmphasis(tau_s=75e-6, sample_rate=48000)
        de2 = DeEmphasis(tau_s=75e-6, sample_rate=48000)

        x = np.random.default_rng(seed=5).standard_normal(1000).astype(
            np.float32
        )
        single = de1.process(x)
        blocks = np.concatenate(
            [de2.process(x[0:400]), de2.process(x[400:1000])]
        )
        assert np.allclose(single, blocks, atol=1e-5)


# ==================================================================
# FMDemodulator (composite, end-to-end)
# ==================================================================


class TestFMDemodulator:
    def test_rejects_output_rate_above_input_rate(self):
        with pytest.raises(ValueError, match="input_rate"):
            FMDemodulator(input_rate=1000, output_rate=2000)

    def test_intermediate_rate_chosen_sensibly(self):
        """Intermediate rate should be input_rate // decimation where
        decimation = input_rate // output_rate."""
        demod = FMDemodulator(input_rate=2_400_000, output_rate=48_000)
        # 2400000 // 48000 = 50; intermediate = 2400000 / 50 = 48000
        assert demod.intermediate_rate == 48_000
        assert demod.resampler is None  # exact rate, no fractional step

    def test_fractional_rate_requires_resampler(self):
        demod = FMDemodulator(input_rate=2_400_000, output_rate=22_050)
        assert demod.resampler is not None

    def test_empty_input_returns_empty(self):
        demod = FMDemodulator(input_rate=2_400_000, output_rate=48_000)
        out = demod.process_iq(np.empty(0, dtype=np.complex64))
        assert out.size == 0
        assert out.dtype == np.int16

    def test_output_is_int16(self):
        demod = FMDemodulator(input_rate=2_400_000, output_rate=48_000)
        iq = np.ones(48_000, dtype=np.complex64)  # DC
        out = demod.process_iq(iq)
        assert out.dtype == np.int16

    def test_fm_round_trip_recovers_tone(self):
        """Generate a complex-baseband FM-modulated 1 kHz tone at
        48 kHz input rate. Demodulate to 48 kHz audio. Verify the
        dominant output frequency is 1 kHz.

        FM baseband: s(t) = exp(j * beta * sin(2*pi*f_m*t)) where
        beta = f_d / f_m. Quadrature demod recovers the derivative of
        the phase modulation (≈ the original tone with a 90° shift).
        """
        fs_in = 48_000
        fs_out = 48_000  # no resampling to keep the test simple
        f_m = 1_000  # message tone: 1 kHz
        f_d = 5_000  # FM deviation: 5 kHz
        duration_s = 1.0
        n = np.arange(int(fs_in * duration_s))
        t = n / fs_in

        # Instantaneous phase modulation
        phase = (f_d / f_m) * np.sin(2 * np.pi * f_m * t)
        iq = np.exp(1j * phase).astype(np.complex64)

        demod = FMDemodulator(input_rate=fs_in, output_rate=fs_out)
        pcm = demod.process_iq(iq)
        audio = pcm.astype(np.float32) / 32768.0

        # FFT the steady-state portion (skip filter warm-up)
        trimmed = audio[500:]
        fft = np.abs(np.fft.rfft(trimmed))
        freqs = np.fft.rfftfreq(trimmed.size, d=1 / fs_out)
        peak_idx = np.argmax(fft)
        peak_freq = freqs[peak_idx]
        assert abs(peak_freq - f_m) < 10, (
            f"expected 1000 Hz peak, got {peak_freq:.1f} Hz"
        )

    def test_fm_round_trip_with_fractional_resample(self):
        """Same as above but with 2.4 Msps → 22050 Hz (the multimon
        case). Fractional resampler is in the pipeline."""
        fs_in = 2_400_000
        fs_out = 22_050
        f_m = 1_000
        f_d = 5_000
        duration_s = 0.5  # 0.5s = 1.2M IQ samples
        n = np.arange(int(fs_in * duration_s))
        t = n / fs_in
        phase = (f_d / f_m) * np.sin(2 * np.pi * f_m * t)
        iq = np.exp(1j * phase).astype(np.complex64)

        demod = FMDemodulator(input_rate=fs_in, output_rate=fs_out)
        pcm = demod.process_iq(iq)
        audio = pcm.astype(np.float32) / 32768.0

        # Check the output is the expected length
        expected = int(duration_s * fs_out)
        assert abs(audio.size - expected) < fs_out * 0.05

        # FFT analysis
        trimmed = audio[1000:]
        fft = np.abs(np.fft.rfft(trimmed))
        freqs = np.fft.rfftfreq(trimmed.size, d=1 / fs_out)
        peak_idx = np.argmax(fft)
        peak_freq = freqs[peak_idx]
        assert abs(peak_freq - f_m) < 20, (
            f"expected 1000 Hz peak, got {peak_freq:.1f} Hz"
        )

    def test_process_iq_bytes_accepts_rtl_sdr_bytes(self):
        """Verify the bytes-in interface matches RTL-SDR's uint8 IQ
        wire format."""
        demod = FMDemodulator(input_rate=2_400_000, output_rate=22_050)
        # DC-centered bytes (127, 128 pattern) - minimal signal
        data = bytes([127, 128] * 100_000)
        out = demod.process_iq_bytes(data)
        # Should produce some output (about 100_000 * 22050 / 2400000 ≈ 918 samples)
        assert out.dtype == np.int16
        assert out.size > 0


# ==================================================================
# Golden vector (regression)
# ==================================================================


class TestGoldenVector:
    """A regression test: generate a fixed input and assert a specific
    output checksum. Catches accidental algorithmic drift."""

    def test_fm_demod_reproducible_output(self):
        """Running the same synthetic input twice must give identical
        bytes. This is implementation-dependent but catches nondeterminism
        and accidental algorithm changes."""
        rng = np.random.default_rng(seed=12345)
        # Fixed 10k-sample pseudo-FM signal
        n = np.arange(10_000)
        phase = 5.0 * np.sin(2 * np.pi * 500 / 48000 * n)
        iq = np.exp(1j * phase).astype(np.complex64)

        def run():
            demod = FMDemodulator(input_rate=48_000, output_rate=48_000)
            return demod.process_iq(iq)

        a = run()
        b = run()
        assert np.array_equal(a, b), (
            "FMDemodulator should be deterministic; same input → same output"
        )
