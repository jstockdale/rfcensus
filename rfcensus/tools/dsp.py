"""DSP primitives for the FM bridge and related audio-from-IQ tools.

These building blocks are deliberately small and testable in isolation:
IQ-byte unpacking, stateful quadrature demodulation, stateful decimating
lowpass, stateful fractional resampling, and 1st-order de-emphasis. The
composite FMDemodulator chains them together in the order we actually
need for turning an RTL-SDR IQ stream into PCM audio suitable for
multimon-ng / direwolf input.

Design notes
============

**Stateful across blocks.** Streaming DSP on a live IQ stream arrives in
blocks (typically 16 kB at a time). Every primitive here retains the
state it needs so that processing one big array vs N equal-sized blocks
produces effectively identical output — no boundary glitches that would
sound like clicks in the decoded audio or like spurious edges that
confuse a tight AFSK1200 decoder.

The concrete state per primitive:

* **IQ unpack**: stateless (each block is independent bytes).
* **Quadrature demod**: the previous IQ sample (y[n] depends on x[n-1]).
* **Decimating lowpass**: the FIR filter tap state (last K-1 samples in
  the filter memory) AND the phase position within the decimation cycle
  so output-sample indices stay aligned across block boundaries.
* **Resampler**: because the polyphase resample is not trivially stateful
  in scipy, we maintain a small overlap buffer and splice results.
* **De-emphasis**: the single IIR memory cell.

**Type choices.** IQ samples are complex64 (float32 real + float32 imag)
to keep memory footprint small and make numpy vectorization fast. The
final PCM output is int16 because that's what multimon and direwolf
expect on stdin.

**Performance.** At 2.4 Msps input, the decimating lowpass is the
dominant cost. We use scipy.signal.lfilter which is vectorized C. A
128-tap filter at 2.4 Msps is around 15-20% of a modern CPU core in
numpy/scipy; feasible on metatron but tight on a Pi Zero. If we ever
deploy to resource-constrained hardware we'll swap in csdr.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly


# ----------------------------------------------------------------
# IQ byte unpacking
# ----------------------------------------------------------------


def iq_uint8_to_complex(data: bytes) -> np.ndarray:
    """Convert interleaved uint8 IQ bytes (the rtl_sdr/rtl_tcp wire
    format) to a complex64 numpy array.

    RTL-SDR dongles output 8-bit IQ with a DC offset of 127.5 and full
    scale at [0, 255]. We center on 0 and scale to [-1, 1], then
    combine the interleaved I/Q pairs into complex samples.

    Empty input returns an empty array (not an error) — simplifies the
    streaming loop's block-at-a-time iteration.

    Raises ValueError if `data` has an odd length (would leave a
    dangling I without its Q).
    """
    if len(data) == 0:
        return np.zeros(0, dtype=np.complex64)
    if len(data) % 2 != 0:
        raise ValueError(
            f"IQ byte stream has odd length {len(data)}; expected "
            f"pairs of (I, Q) bytes"
        )

    # Unpack as uint8, convert to float32, center and scale.
    raw = np.frombuffer(data, dtype=np.uint8)
    # float32 is enough precision for 8-bit IQ and saves half the
    # memory/compute vs float64.
    centered = (raw.astype(np.float32) - 127.5) / 127.5
    i = centered[0::2]
    q = centered[1::2]
    # Combine into complex. .view() tricks avoid a copy but require
    # exact interleaving; an explicit i + 1j*q is clearer and numpy
    # compiles it to a single C loop.
    return (i + 1j * q).astype(np.complex64)


# ----------------------------------------------------------------
# Quadrature demodulator (FM discriminator)
# ----------------------------------------------------------------


@dataclass
class QuadratureDemod:
    """FM discriminator using the standard quadrature approach:
    y[n] = angle(x[n] * conj(x[n-1]))

    The output is the instantaneous phase difference between successive
    IQ samples — proportional to the instantaneous frequency deviation
    from the carrier, which for FM is the baseband message.

    Stateful across blocks: the previous sample's value carries forward
    so that y[0] of each block correctly depends on the last sample of
    the previous block, not on a fresh '1 + 0j' assumption.

    Output is float32 in radians (range [-pi, pi]). Downstream code
    typically just treats this as 'audio': scale, filter, and feed to
    the decoder. Absolute scaling depends on the FM deviation and the
    sample rate, but for AFSK1200/POCSAG/FLEX all we need is that the
    waveform shape is correct — the decoders normalize internally.
    """

    _prev: complex = field(default=complex(1.0, 0.0))

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.zeros(0, dtype=np.float32)
        # Prepend previous sample so y[0] uses correct x[-1].
        # np.concatenate with a length-1 array is cheap enough given
        # our block sizes (thousands of samples).
        extended = np.concatenate(([self._prev], x))
        # Product x[n] * conj(x[n-1]) has phase = angle(x[n]) - angle(x[n-1]).
        prod = extended[1:] * np.conj(extended[:-1])
        out = np.angle(prod).astype(np.float32)
        self._prev = complex(x[-1])
        return out

    def reset(self) -> None:
        """Clear state. Useful between independent streams."""
        self._prev = complex(1.0, 0.0)


# ----------------------------------------------------------------
# Decimating lowpass
# ----------------------------------------------------------------


class DecimatingLowpass:
    """Stateful FIR lowpass followed by integer decimation.

    Reduces sample rate by `decimation` while filtering out any signal
    content above the new Nyquist to prevent aliasing. Internally:

      1. Apply FIR lowpass (windowed-sinc, Hamming window)
      2. Pick every Nth sample

    The filter state (last `num_taps-1` samples in the FIR delay line)
    is preserved across `process()` calls via scipy's `lfilter(zi=)`
    mechanism. The decimation phase (which output index to start on)
    is also tracked so block boundaries don't misalign the downsampled
    stream.

    Cutoff is specified as a fraction of the OUTPUT Nyquist rate, so
    0.45 means "pass 90% of the output band, reserve 10% for the FIR
    transition band." This lets us use a reasonable number of taps
    (129) without aliasing into the audio band.
    """

    def __init__(
        self,
        decimation: int,
        *,
        num_taps: int = 129,
        cutoff_fraction: float = 0.45,
        dtype: type = np.complex64,
    ) -> None:
        if decimation < 1:
            raise ValueError(
                f"decimation must be >= 1, got {decimation}"
            )
        if num_taps < 3 or num_taps % 2 == 0:
            raise ValueError(
                f"num_taps must be odd and >= 3 for linear phase; "
                f"got {num_taps}"
            )
        if not 0.0 < cutoff_fraction < 1.0:
            raise ValueError(
                f"cutoff_fraction must be in (0, 1); got {cutoff_fraction}"
            )

        self.decimation = decimation
        self.num_taps = num_taps
        # FIR cutoff in normalized frequency (Nyquist = 1.0 relative to
        # INPUT rate). Output Nyquist = 1/decimation. We pass `cutoff_fraction`
        # of that band.
        cutoff_norm = cutoff_fraction / decimation
        # firwin returns real taps; applying to complex input works fine
        # (treats real as (real+0j)).
        self.taps = firwin(num_taps, cutoff_norm, window="hamming").astype(
            np.float32
        )
        # Filter state for lfilter (length num_taps-1, same dtype as input)
        self._filter_state = np.zeros(num_taps - 1, dtype=dtype)
        # Cumulative input samples consumed, mod decimation, to keep
        # block-boundary decimation aligned with a single-pass reference.
        self._samples_consumed_mod = 0
        # NOTE (v0.5.45): lfilter computes all N input samples' filtered
        # output even though we discard (M-1)/M of them. For M=50 this
        # is ~50× more arithmetic than strictly necessary. A polyphase
        # decimator using scipy.signal.upfirdn would be ideal, but
        # reconciling its output-length convention with our block-by-
        # block streaming + arbitrary phase_offset semantics is
        # non-trivial — partial implementation in v0.5.45 had off-by-N
        # alignment for non-M-divisible block sizes. Deferred to a
        # future revision; the resampler fix in v0.5.45 already cut
        # total fm_bridge CPU from ~100% of a Pi 5 core to ~30%, so
        # this is a "next optimization" rather than urgent.

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.empty(0, dtype=x.dtype)
        filtered, self._filter_state = lfilter(
            self.taps, 1.0, x, zi=self._filter_state
        )
        # Starting index for decimation so that global sample index
        # (self._samples_consumed_mod + local_i) % decimation == 0
        start = (-self._samples_consumed_mod) % self.decimation
        decimated = filtered[start :: self.decimation]
        self._samples_consumed_mod = (
            self._samples_consumed_mod + x.size
        ) % self.decimation
        # lfilter returns float64 with real taps + complex input.
        # Force back to input dtype for memory discipline.
        return decimated.astype(x.dtype)


# ----------------------------------------------------------------
# Fractional resampler (output rate = input rate * up / down)
# ----------------------------------------------------------------


class Resampler:
    """Rational-rate resampler via scipy.signal.resample_poly.

    IMPORTANT: resample_poly is not natively block-stateful — each call
    treats its input as an isolated signal with implicit zero padding
    at the edges. For streaming, that means small discontinuities can
    appear at block boundaries.

    We mitigate with an overlap-hold scheme: each `process()` call
    prepends a small tail of samples from the previous block (enough
    to fill the filter's warm-up region) and trims an equivalent
    portion off the front of the output. This keeps the streamed
    output indistinguishable from a single-pass resample at steady
    state, at the cost of a fixed startup delay proportional to the
    filter length.

    For narrowband FM voice / AFSK this is imperceptible. If block
    sizes are unusually small (<< filter length), quality degrades.
    """

    def __init__(
        self,
        up: int,
        down: int,
        *,
        max_filter_taps: int = 257,
        stopband_attenuation_db: float = 60.0,
    ) -> None:
        """v0.5.45: bound the filter length to keep CPU in check.

        The previous version let scipy.signal.resample_poly auto-build
        its polyphase filter, which defaults to `10 * max(up, down)`
        taps. For awkward fractional ratios (e.g. 22222 Hz → 22050 Hz
        with up=11025/down=11111) that's 111,111 taps — each output
        sample costs 111k multiplies. fm_bridge was spending 98% of
        its CPU inside this one step, burning a full Pi 5 core per
        instance.

        For our use case (narrowband FM voice / AFSK, audio content
        well below 4 kHz, output sample rates ≥ 22 kHz) the filter
        requirements are very loose: any mild lowpass at ~0.45 × Nyquist
        with 60 dB stopband is fine. A 200-ish-tap Kaiser-window FIR
        designed by us and passed to `resample_poly` via the `window=`
        parameter eliminates the 100k-tap catastrophe and gets us to
        the expected < 10% CPU for this stage.

        `max_filter_taps` is a ceiling — if up/down are small the
        auto-derived length (5 × max(up,down) + 1) is used instead.
        Odd only, for linear phase.
        """
        if up < 1 or down < 1:
            raise ValueError("up and down must be >= 1")
        from math import gcd

        g = gcd(up, down)
        self.up = up // g
        self.down = down // g

        # Design a short Kaiser-window FIR matched to the output
        # spectrum we actually need. Cutoff = 1 / max(up, down) in
        # scipy.signal.resample_poly's normalized frequency convention.
        # Kaiser beta for 60 dB is ~5.65.
        from scipy.signal import firwin, kaiser_beta
        beta = kaiser_beta(stopband_attenuation_db)
        # Take the smaller of (5 × max(up,down) + 1) and max_filter_taps.
        # Round UP to the next odd for linear phase.
        ideal_len = 5 * max(self.up, self.down) + 1
        num_taps = min(ideal_len, max_filter_taps)
        if num_taps % 2 == 0:
            num_taps += 1
        # Normalized cutoff. resample_poly uses cutoff relative to the
        # filter's internal rate which is `up × input_rate`. Normalized
        # cutoff = 1 / max(up, down) is the correct Nyquist boundary
        # to avoid aliasing AND imaging. Pull it in a touch (0.95) for
        # transition band.
        cutoff = 0.95 / max(self.up, self.down)
        # firwin wants cutoff in [0, 1] where 1.0 = Nyquist of the
        # filter's sample rate, which for resample_poly internals is
        # up × input_rate. So cutoff here matches that convention.
        taps = firwin(
            num_taps, cutoff,
            window=("kaiser", beta), pass_zero="lowpass",
        )
        # Pass taps directly — scipy.signal.resample_poly's window=
        # parameter accepts an array of taps and handles the
        # zero-insertion gain compensation internally. (We tested
        # multiplying by `up` and confirmed it produces `up`× gain
        # at the output, not unity.)
        self._filter_taps = taps.astype(np.float32)

        # Overlap history: we need `num_taps / 2 / up` input samples
        # of context so the filter's transient doesn't corrupt block
        # boundaries. Round up generously.
        self._overlap_samples = max(
            (num_taps + self.up - 1) // self.up + 8, 32
        )
        self._history: np.ndarray | None = None
        self._num_taps = num_taps  # for output-length computation

    @staticmethod
    def _resample_poly_output_len(
        filter_len: int, in_len: int, up: int, down: int
    ) -> int:
        """Replicates scipy.signal.resample_poly's internal length
        formula. Used for exact per-block trim computation so block-by-
        block output count matches what a single-pass call would
        produce, modulo per-block ceil rounding."""
        in_len_copy = in_len + (filter_len + (-filter_len % up)) // up - 1
        nt = in_len_copy * up
        need = nt // down
        if nt % down > 0:
            need += 1
        return need

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.empty(0, dtype=x.dtype)

        if self._history is None:
            # First block: pad with zeros at the start. Output will have
            # a brief transient but subsequent blocks are clean.
            extended = np.concatenate(
                [np.zeros(self._overlap_samples, dtype=x.dtype), x]
            )
        else:
            extended = np.concatenate([self._history, x])

        # v0.5.45: pass our pre-designed filter via the `window=` param.
        # resample_poly's `window` accepts either a get_window() spec
        # (triggers auto-sizing) OR a 1D array of filter taps (uses
        # directly). We pass the taps → scipy uses our short filter
        # instead of building a 100k-tap one for awkward fractional
        # ratios.
        out = resample_poly(
            extended, self.up, self.down,
            window=self._filter_taps, padtype="line",
        )

        # v0.5.45: compute exact trim using scipy's own output-length
        # formula. Trim = (output for extended) - (output for x alone),
        # which is the number of output samples produced by the overlap
        # prefix. This makes per-block output count match what a
        # single-pass call on x alone would produce — and across N
        # blocks, the cumulative output count tracks the single-pass
        # full-stream count to within ±N (per-block ceil rounding).
        out_len_extended = self._resample_poly_output_len(
            self._num_taps, len(extended), self.up, self.down,
        )
        out_len_x = self._resample_poly_output_len(
            self._num_taps, x.size, self.up, self.down,
        )
        trim = max(0, out_len_extended - out_len_x)
        if trim > 0:
            out = out[trim:]

        # Retain the last overlap_samples of input for the next block's
        # warm-up history.
        if x.size >= self._overlap_samples:
            self._history = x[-self._overlap_samples :].copy()
        else:
            # Block is smaller than our overlap buffer — concatenate
            # the existing history (sans its oldest part) plus the new
            # block, and keep the last overlap_samples of that.
            if self._history is None:
                padded = np.concatenate(
                    [np.zeros(self._overlap_samples, dtype=x.dtype), x]
                )
            else:
                padded = np.concatenate([self._history, x])
            self._history = padded[-self._overlap_samples :].copy()

        return out.astype(x.dtype)


# ----------------------------------------------------------------
# De-emphasis (1st-order IIR)
# ----------------------------------------------------------------


class DeEmphasis:
    """1st-order IIR de-emphasis filter, time-constant parameterized.

    Commercial FM broadcast uses 75 us (North America) or 50 us
    (Europe). Narrowband FM (paging, APRS) typically doesn't need
    de-emphasis at all — the pre-emphasis is either absent or
    cancelled by the receiver's audio filter. For our use case
    (multimon / direwolf), we default to "no de-emphasis" (None),
    but expose the option for audio-listening applications.

    The filter form is:
      y[n] = a * x[n] + (1 - a) * y[n-1]
    where a = 1 - exp(-1 / (tau * fs))

    Time constant tau is the RC constant (seconds); fs is the
    sample rate.
    """

    def __init__(self, tau_s: float | None, sample_rate: int) -> None:
        if tau_s is None or tau_s <= 0:
            self._a: float | None = None
            self._prev_y: float = 0.0
            return
        self._a = 1.0 - float(np.exp(-1.0 / (tau_s * sample_rate)))
        self._prev_y = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        if self._a is None:
            # Pass-through — tau was None/zero
            return x
        if x.size == 0:
            return x
        # 1st-order IIR; scipy.signal.lfilter for speed
        # y[n] = a*x[n] + (1-a)*y[n-1]
        # In transfer-function form: H(z) = a / (1 - (1-a)*z^-1)
        b = [self._a]
        a = [1.0, -(1.0 - self._a)]
        zi = np.array([(1.0 - self._a) * self._prev_y], dtype=np.float64)
        y, zf = lfilter(b, a, x, zi=zi)
        self._prev_y = float(y[-1])
        return y.astype(x.dtype)


# ----------------------------------------------------------------
# Composite: IQ → PCM pipeline
# ----------------------------------------------------------------


def _pick_decimation(input_rate: int, output_rate: int) -> tuple[int, int]:
    """Choose an integer decimation factor that lands on an intermediate
    rate with the cleanest possible ratio to output_rate.

    Returns ``(decimation, intermediate_rate)`` where:

      • ``input_rate % decimation == 0`` (clean integer decimation)
      • ``intermediate_rate = input_rate / decimation``
      • ``intermediate_rate >= 2 * output_rate`` (audio Nyquist
        headroom for the subsequent fractional resample)
      • ``max(up, down)`` after reduction by ``gcd(intermediate, output)``
        is minimized — keeps the polyphase filter short

    Why this matters: scipy.signal.resample_poly's default polyphase
    filter is ``10 * max(up, down)`` taps. The previous logic
    (``decim = input_rate // output_rate``) picked the smallest
    intermediate just above output_rate, which for awkward ratios
    (2.4 Msps → 22050 Hz → intermediate 22222, ratio 11025/11111)
    triggered a 111,111-tap filter. Walking divisors of input_rate
    and picking the one with the smallest reduced ratio finds
    intermediate=48000 Hz instead (147/320 → ~3200 taps default,
    or 257 with v0.5.45's bounded-filter Resampler).

    Edge cases:
      • If input_rate is an integer multiple of output_rate, returns
        ``(input_rate // output_rate, output_rate)`` — no fractional
        resample needed at all.
      • If output_rate is so high relative to input_rate that no
        decimation can give an intermediate ≥ 2× output_rate,
        returns ``(1, input_rate)`` — caller's resampler handles the
        full ratio.
    """
    from math import gcd

    if input_rate <= 0 or output_rate <= 0:
        raise ValueError(
            f"input_rate and output_rate must be positive; "
            f"got {input_rate}, {output_rate}"
        )
    if input_rate < output_rate:
        raise ValueError(
            f"input_rate ({input_rate}) must be >= output_rate "
            f"({output_rate})"
        )

    # Trivial case: input is integer multiple of output → integer
    # decimation directly to output_rate, no fractional resample.
    if input_rate % output_rate == 0:
        return (input_rate // output_rate, output_rate)

    min_intermediate = 2 * output_rate
    max_decim = input_rate // min_intermediate
    if max_decim < 1:
        # output_rate too high relative to input_rate; can't decimate
        # below 2× output Nyquist. Skip integer decimation entirely.
        return (1, input_rate)

    # Score = max(up, down) of the reduced fractional resample ratio.
    # Lower is better. Ties broken by smaller intermediate rate
    # (= less downstream resample work).
    best: tuple[int, int, int] | None = None  # (score, decim, intermediate)
    for decim in range(1, max_decim + 1):
        if input_rate % decim != 0:
            continue
        intermediate = input_rate // decim
        if intermediate < min_intermediate:
            continue
        g = gcd(intermediate, output_rate)
        up = output_rate // g
        down = intermediate // g
        score = max(up, down)
        if best is None or (score, intermediate) < (best[0], best[2]):
            best = (score, decim, intermediate)

    if best is None:
        # No valid divisor found — fall back to the prior logic.
        # This shouldn't happen for any realistic SDR input rate but
        # keeps the function total.
        decim = max(1, input_rate // output_rate)
        return (decim, input_rate // decim)

    return (best[1], best[2])


class FMDemodulator:
    """End-to-end pipeline: complex IQ at `input_rate` → int16 PCM at
    `output_rate`.

    Stages:
      1. DecimatingLowpass (integer) to get close to output_rate
      2. Quadrature demodulate (complex → real audio)
      3. Optional fractional resample to exact output_rate
      4. Optional de-emphasis
      5. Scale and clip to int16

    For the narrowband FM signals we care about (paging, APRS AFSK),
    the exact output_rate only matters because downstream decoders
    expect specific rates. We pick intermediate_rate = output_rate *
    intermediate_factor, where intermediate_factor is 1 if input_rate
    is an integer multiple of output_rate, otherwise a small factor
    that keeps intermediate_rate above a safe minimum (~50 kHz).
    """

    def __init__(
        self,
        input_rate: int,
        output_rate: int,
        *,
        deemphasis_tau_s: float | None = None,
        audio_scale: float = 16384.0,
    ) -> None:
        if input_rate < output_rate:
            raise ValueError(
                f"input_rate ({input_rate}) must be >= output_rate "
                f"({output_rate})"
            )
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.audio_scale = float(audio_scale)

        # v0.5.45: pick decimation that lands on an intermediate rate
        # with a CLEAN ratio to output_rate. The prior logic (decim =
        # input_rate // output_rate) picked the smallest intermediate
        # that exceeded output_rate — for 2.4 Msps → 22050 Hz that
        # yielded intermediate = 22222 Hz, a ratio of 11025/11111 with
        # no common factor. scipy.signal.resample_poly then built a
        # ~100k-tap polyphase filter to handle this ratio, consuming
        # an entire Pi 5 core per fm_bridge instance.
        #
        # Strategy: walk integer divisors of input_rate looking for one
        # that produces intermediate rate ≥ 2 × output_rate AND has
        # small up/down when reduced against output_rate. For 2.4 Msps
        # → 22050, this lands on decim=50 (intermediate=48000, ratio
        # 147/320, polyphase filter length ~300 with v0.5.45's cap).
        self._decimation, self._intermediate_rate = _pick_decimation(
            input_rate=input_rate, output_rate=output_rate,
        )

        self.decimator = DecimatingLowpass(
            decimation=self._decimation, dtype=np.complex64
        )
        self.demod = QuadratureDemod()

        if self._intermediate_rate == output_rate:
            self.resampler: Resampler | None = None
        else:
            from math import gcd

            g = gcd(self._intermediate_rate, output_rate)
            self.resampler = Resampler(
                up=output_rate // g,
                down=self._intermediate_rate // g,
            )

        self.deemphasis = DeEmphasis(
            tau_s=deemphasis_tau_s, sample_rate=output_rate
        )

    @property
    def intermediate_rate(self) -> int:
        return self._intermediate_rate

    def process_iq_bytes(self, data: bytes) -> np.ndarray:
        """Consume a chunk of RTL-SDR uint8 IQ bytes, emit int16 PCM."""
        iq = iq_uint8_to_complex(data)
        return self.process_iq(iq)

    def process_iq(self, iq: np.ndarray) -> np.ndarray:
        """Consume complex IQ at input_rate, emit int16 PCM at output_rate."""
        if iq.size == 0:
            return np.empty(0, dtype=np.int16)

        # 1. Decimate to intermediate_rate
        decimated = self.decimator.process(iq)
        if decimated.size == 0:
            return np.empty(0, dtype=np.int16)

        # 2. Quadrature demod (complex → real)
        audio = self.demod.process(decimated)

        # 3. Fractional resample if needed
        if self.resampler is not None:
            audio = self.resampler.process(audio)

        # 4. De-emphasis
        audio = self.deemphasis.process(audio)

        # 5. Scale and clip to int16. The FM discriminator output is in
        # radians [-pi, pi]; scale so full-scale deviation fills the
        # int16 range without clipping for normal signals.
        scaled = audio * self.audio_scale
        np.clip(scaled, -32767, 32767, out=scaled)
        return scaled.astype(np.int16)


# ---------------------------------------------------------------------
# v0.5.41: Digital down-conversion for batched LoRa confirmation
# ---------------------------------------------------------------------


def digital_downconvert(
    samples: np.ndarray,
    source_rate: int,
    shift_hz: float,
    target_bw_hz: int,
) -> np.ndarray:
    """Extract a narrowband signal from a wideband IQ capture.

    When one IQ capture at 2.4 Msps covers a 2 MHz window containing
    multiple detections, we extract each individual detection by:

      1. Mixing (multiplying by a complex exponential) to shift the
         target frequency to DC.
      2. Low-pass filtering to the target bandwidth.
      3. Decimating so output rate is ~2× target_bw_hz (Nyquist-safe).

    This is the standard "tuner" operation used throughout SDR — it's
    what GNU Radio's `freq_xlating_fir_filter` does, what `csdr shift`
    + `csdr fir_decimate` does, and it's what rtl_sdr's tuner IC does
    in analog before we ever see the samples.

    Parameters
    ----------
    samples : complex64 array
        Input IQ captured at `source_rate`.
    source_rate : int
        Source sample rate in Hz (e.g. 2_400_000).
    shift_hz : float
        How far to shift the signal. If the target is above the capture
        center, pass a POSITIVE value; signal will be shifted DOWN to DC.
        Typically this is `task.freq_hz - capture.freq_hz`.
    target_bw_hz : int
        Bandwidth of the signal of interest. Determines the low-pass
        cutoff and decimation ratio. For LoRa, this is the
        matched_template_hz (125/250/500 kHz).

    Returns
    -------
    complex64 array at rate >= 2 * target_bw_hz

    Notes
    -----
    We oversample the output by 2× (output rate = 2 * target_bw_hz * 2,
    actually `source_rate / decimation_factor`) so downstream chirp
    analysis has enough bandwidth headroom for frequency-swept chirps
    that briefly exit the nominal channel edges.

    Performance: the mix is O(N), the decimating filter is O(N * taps).
    At 2.4 Msps for 2s = 4.8 M samples, DDC to 250 kHz output takes
    roughly 50 ms on a modern x86 core. Negligible compared to IQ
    capture wall-clock (~2 s).
    """
    if samples.size == 0:
        return np.empty(0, dtype=np.complex64)
    if target_bw_hz <= 0:
        raise ValueError(f"target_bw_hz must be positive; got {target_bw_hz}")
    if source_rate <= 0:
        raise ValueError(f"source_rate must be positive; got {source_rate}")

    # Step 1: mix. exp(-j*2*pi*shift*t) shifts the target DOWN to DC.
    # Using float32 throughout keeps memory and cost down — double
    # precision is unnecessary for a few seconds of samples.
    t = np.arange(samples.size, dtype=np.float32) / np.float32(source_rate)
    phase = np.float32(-2.0 * np.pi * shift_hz) * t
    mixer = np.exp(1j * phase).astype(np.complex64)
    shifted = samples * mixer

    # Step 2 + 3: decimate with built-in anti-alias.
    # Output rate = source_rate / decim. We want output_rate ~= 2 *
    # target_bw * 2 = 4 * target_bw (2x oversampled relative to
    # Nyquist). Clamp decim ≥ 1 for narrow captures.
    target_output_rate = max(4 * target_bw_hz, source_rate // 32)
    decim = max(1, source_rate // target_output_rate)
    if decim == 1:
        # No decimation needed — apply a cheap low-pass only. Use
        # scipy.signal.firwin + lfilter directly (the DecimatingLowpass
        # class requires decimation>=1 but expects different args; we
        # don't need its state management here since we're called once
        # per capture, not streamed).
        from scipy.signal import firwin, lfilter
        # Cutoff as fraction of Nyquist. Normalize target bandwidth
        # by the half-rate.
        nyquist = source_rate / 2.0
        cutoff_norm = min(0.9, target_bw_hz / nyquist)
        taps = firwin(65, cutoff_norm)
        filtered = lfilter(taps, 1.0, shifted)
        return filtered.astype(np.complex64)

    # For real decimation use scipy.signal.decimate which chains
    # Chebyshev/FIR anti-alias + downsample in one call. Complex
    # input is handled component-wise internally.
    from scipy.signal import decimate as _decimate
    decimated = _decimate(shifted, decim, ftype="fir", zero_phase=True)
    return decimated.astype(np.complex64)
