# Native LoRa Decoder — `rfcensus/decoders/_native/lora`

Clean-room C99 LoRa physical layer decoder for rfcensus. Designed for
headless server-side use on Pi 5 / x86, no GNU Radio runtime.

## License

BSD-3-Clause. Vendors `kiss_fft.{c,h}` (also BSD-3) from the kissfft
project. No GPL dependencies. Algorithm references:
  - Semtech SX1276/77/78/79 datasheet (packet format)
  - Semtech AN1200.18 / AN1200.22 (modulation basics, hidden registers)
  - Tapparel et al. "An Open-Source LoRa Physical Layer Prototype on
    GNU Radio", arXiv:2002.08208 — describes the algorithm only; code
    not used.

## Status — v0.6.16 Phase B (sync milestone reached, header decode pending)

### ✅ Validated

| Component                       | Status | How verified                  |
|---------------------------------|--------|-------------------------------|
| Chirp orthogonality (SF7-SF12)  | ok     | synthetic FFT(up×down) → δ[0] |
| Symbol round-trip (all SF7 vals)| ok     | encode→demod, all 128 values  |
| CFO correction (both directions)| ok     | synthetic + real captures     |
| Digital frequency mix end-to-end| ok     | synth bin 99 at -100 kHz IF   |
| Hamming(8,4) decode + correct   | ok     | all 16 nibbles + bit flip     |
| Hamming(7,4) decode             | ok     | known-vector test             |
| CRC-16 (LoRa flavor)            | ok     | published vectors             |
| Gray code round-trip            | ok     | all 12-bit values             |
| Whitening LFSR sequence         | ok     | first byte = 0xFF per spec    |
| Build clean (no warnings)       | ok     | gcc -Wall -Wextra clean       |
| Detection on real Meshtastic    | ok     | 8 preambles in 30s @ 1Msps    |
| **Sync word matching (SF9)**    | **ok** | **8/8 sync words match**      |
| Header decode                   | TBD    | need fractional STO + SFO     |

### Critical empirical findings

**The Meshtastic preamble is 16 chirps (vs LoRa's standard 8).** Our DETECT
only requires 8-in-a-row, so transition to SYNC_NETID often happens mid-
preamble. Fix: SYNC_NETID slides forward by N until we see a non-preamble
bin (the actual sync_word_1), capped at 16 extra slides. Matches gr-lora's
NET_ID1 case at frame_sync_impl.cc:568-587.

**There is a +N/4 sample gap between sync_word_1 and sync_word_2 in this
fixture's signals.** Probe data: at cursor+N (where sync_word_2 should
start per the spec), bin = 472 = expected_value - N/4. At cursor+N+N/4,
bin = 88 = exactly the expected value. Pattern reproduces across 9 packets
at SF9 with diff=(+0,+0). Cause unknown (suspected SX1262 chipset quirk
or Meshtastic firmware) but the fix is solid against this fixture.

### ⚠️  Known issues — must fix for end-to-end Meshtastic decode

1. **Header decode fails despite clean sync match.** With +N/4 fix applied
   to DEMOD reads, symbol magnitudes go from ~270 to ~340-370, but the
   peak (~496) sits at dx ≈ -N/2 + 32 — suggesting compounding fractional
   STO and SFO drift across the frame. Proper fix requires:
   - Fractional STO estimation (gr-lora's `k_hat` parameter from
     downchirp dechirp at frame sync)
   - SFO compensation (per-symbol bin shift proportional to elapsed time)
   - These are non-trivial DSP; deferred to v0.6.17

2. **Energy gate is hand-tuned.** `mag < N * 0.04f` works for the test
   fixture but needs validation across SNR range. CSV-logger TBD.

3. **Implicit-header mode not implemented.** Meshtastic always uses
   explicit header so this doesn't block our use case.

4. **Soft-decision Hamming not implemented.** ~1-2 dB sensitivity loss.
   Tracked for v0.6.17+.

## File layout

```
lora_demod.h        — public API (includes are stable; ABI not)
lora_internal.h     — private types shared across .c files
lora_demod.c        — frame sync state machine + lifecycle
lora_chirp.c        — reference chirp generation, dechirp+FFT
lora_codec.c        — Hamming, Gray, deinterleave, dewhiten, CRC
kiss_fft.{c,h}      — vendored BSD-3 single-file FFT
_kiss_fft_guts.h    — kissfft internals header

test_decode.c       — runs decoder on a .cu8 capture file, reports stats
test_synth.c        — synthetic correctness tests (no RF needed)

Makefile            — builds liblora_demod.{a,so} + test binaries
```

## Building

```bash
cd rfcensus/decoders/_native/lora
make                      # builds .a + .so
make test_decode          # builds the cu8 file harness
cc -O2 -I. -o test_synth test_synth.c liblora_demod.a -lm
./test_synth              # synthetic tests, exits 0 on pass
```

Targets Pi 5 (`-mcpu=native` auto-detected on aarch64) and x86_64
(`-msse4.2`). Both produce identical output (modulo float ULP noise);
SF12/N=4096 FFT is the slowest path at ~50 µs per symbol on Pi 5.

## API

See `lora_demod.h` for full docs. Quickstart:

```c
#include "lora_demod.h"

void on_packet(const lora_decoded_t *pkt, void *ud) {
    printf("len=%u crc=%s\n", pkt->payload_len, pkt->crc_ok ? "ok" : "FAIL");
    /* pkt->payload is the raw decoded LoRa payload bytes — still
     * Meshtastic-encrypted at this layer. Pass to meshtastic-lite for
     * AES-CTR decrypt + protobuf decode. */
}

lora_config_t cfg = {
    .sample_rate_hz = 250000,    /* must be ≥ bandwidth */
    .bandwidth = LORA_BW_250,
    .sf = 11,                    /* LongFast preset */
    .sync_word = 0x2B,           /* Meshtastic */
    .ldro = 0,
};
lora_demod_t *d = lora_demod_new(&cfg, on_packet, NULL);

/* Pump samples in any chunk size */
while (1) {
    uint8_t buf[16384];
    size_t n_bytes = read_iq_chunk(buf, sizeof(buf));
    lora_demod_process_cu8(d, buf, n_bytes / 2);
}
lora_demod_free(d);
```

## Roadmap

| Phase | Scope                                                       | Status |
|-------|-------------------------------------------------------------|--------|
| A     | Architecture + DSP scaffold, codec layer validated          | DONE   |
| B-1   | Chirp formula off-by-one fixed                              | DONE   |
| B-2   | Digital frequency mix + sign-error fix                      | DONE   |
| B-3   | Smarter DETECT loop + tighter mag floor                     | DONE   |
| B-4   | STO/CFO refinement via downchirp dechirp (partial)          | DONE   |
| B-5   | Slide-past-extended-preamble fix in SYNC_NETID              | DONE   |
| B-6   | Empirical sym2 +N/4 offset fix                              | DONE   |
| B-7   | Field validation: 8/8 sync words match on Meshtastic SF9    | DONE   |
| B-8   | Fractional STO + SFO compensation for header decode         | next   |
| C     | Python ctypes binding + meshtastic-lite Python port         | TBD    |
| D     | rfcensus integration: meshtastic_decoder consumer + bands   | TBD    |
| E     | Multi-channel + multi-PSK decryption support                | TBD    |
| F     | Soft-decision Hamming for ~1-2 dB sensitivity gain          | TBD    |

## Testing

```bash
make && ./test_synth                                  # synthetic
make test_decode && ./test_decode <capture> ...       # field test
```

For field tests, capture must be at a sample rate equal to the
bandwidth × integer (e.g. 1 Msps for BW=250 kHz). Non-integer ratios
work but with reduced sensitivity due to dechirp reference drift.
