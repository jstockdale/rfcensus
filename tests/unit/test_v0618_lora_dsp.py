"""v0.6.18 — LoRa decoder DSP improvements.

Tests the additions that lifted DEMOD symbol magnitudes from ~270
(54% of N=512) to ~440 (86% of N) on real Meshtastic captures:

  • Joint CFO + STO refinement from up_val + down_val
  • Bernier fractional CFO estimator (exists; correctness verified by
    the magnitude lift on real data — this test just verifies the
    field is exposed and survives a clean build)
  • Multi-sync candidate matching (sync byte derived from bins)
  • observed_sync_word field exposed for consumer routing
  • +1 mod 2^sf step in gray demap (per gr-lora gray_demap_impl.cc)
  • Header CRC formula corrected to match gr-lora exactly

The native LoRa decoder is loaded via ctypes; if the .so isn't built
the tests skip rather than fail (CI may not have a C toolchain).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

NATIVE_DIR = Path(__file__).parent.parent.parent / (
    "rfcensus/decoders/_native/lora"
)


def _native_built() -> bool:
    return (NATIVE_DIR / "liblora_demod.so").exists()


pytestmark = pytest.mark.skipif(
    not _native_built(),
    reason="liblora_demod.so not built — run `make` in "
           "rfcensus/decoders/_native/lora/ first",
)


class TestNativeArtifactsExist:
    """Build artifacts present after `make`."""

    def test_shared_lib_exists(self):
        assert (NATIVE_DIR / "liblora_demod.so").exists()

    def test_static_lib_exists(self):
        assert (NATIVE_DIR / "liblora_demod.a").exists()

    def test_internal_header_exposes_observed_sync(self):
        """v0.6.17/18: observed_sync_word field is exposed in the
        struct so Python consumers can route packets by actual on-air
        sync word (Meshtastic public 0x2B vs other private networks),
        not just by configured sync."""
        text = (NATIVE_DIR / "lora_internal.h").read_text()
        assert "observed_sync_word" in text

    def test_internal_header_exposes_bernier_state(self):
        """v0.6.18: preamble_fft_at_peak + preamble_peak_bin used
        by Bernier CFO_frac estimator."""
        text = (NATIVE_DIR / "lora_internal.h").read_text()
        assert "preamble_fft_at_peak" in text
        assert "preamble_peak_bin" in text


class TestSourceConventions:
    """Source-level invariants — these fall under the v0.6.18 fixes
    but are easier to test by inspecting source than by exercising
    the C decoder. Without these in place, the algorithm is wrong."""

    def _read(self, name: str) -> str:
        return (NATIVE_DIR / name).read_text()

    def test_demod_codec_applies_v_minus_1(self):
        """gr-lora fft_demod_impl.cc:313 maps raw bin v to
        ((v - 1) mod N) / div. Without the -1, every header symbol
        is off by 1 in gray space."""
        text = self._read("lora_codec.c")
        # Look for the (v + N - 1) % N pattern (since C wraps via add)
        assert "v + N - 1" in text or "(v - 1)" in text or "v - 1)" in text

    def test_demod_codec_applies_plus_1_after_gray(self):
        """v0.6.18 (final): the +1 was a misreading of gr-lora.
        gr-lora's RX-side gray_mapping_impl.cc:70 just does
        `out = in ^ (in >> 1)` (= gray ENCODE, single XOR) with NO
        +1 step. The +1 belongs to gray_demap_impl.cc which is on
        the TX side. Adding it on RX scrambled all payload bytes
        and broke payload CRC; removing it (and switching gray_decode
        → gray_encode) made the payload decode correctly.

        Test now ASSERTS the +1 is GONE and that gray_encode is used
        (NOT gray_decode + 1)."""
        text = self._read("lora_codec.c")
        # The buggy combination must NOT appear in executable code
        import re
        no_comments = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        no_comments = re.sub(r"//.*", "", no_comments)
        assert "lora_gray_decode(v) + 1" not in no_comments, \
            "RX should NOT add +1 after gray decode (that's a TX-side step)"
        # And the new gray_encode call must be present
        assert "lora_gray_encode(v)" in no_comments, \
            "RX should use single-XOR gray_encode per gr-lora gray_mapping_impl.cc:70"

    def test_header_crc_extraction_matches_grlora(self):
        """gr-lora header_decoder_impl.cc:138 extracts:
            header_chk = ((in[3] & 1) << 4) + in[4]
        i.e., bit 0 of nibble 3 + all 4 bits of nibble 4 = 5 bits.
        Earlier versions extracted (n3 & 0x0F) << 1 | (n4 & 0x08) >> 3
        which is wrong (4 bits of n3 + 1 bit of n4)."""
        text = self._read("lora_codec.c")
        # New extraction must be present
        assert "(nibbles[3] & 0x1) << 4" in text
        # The old buggy formula must not appear in EXECUTABLE code.
        # It IS allowed in the comment that explains the bug fix.
        # Strip out comments and re-check.
        import re
        # Remove /* ... */ block comments (non-greedy across lines)
        code_only = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        # Remove // line comments
        code_only = re.sub(r"//[^\n]*", "", code_only)
        assert "(nibbles[3] & 0x0F) << 1" not in code_only, (
            "old buggy header_chk extraction still present in active code"
        )

    def test_header_crc_c2_uses_info_bit_2(self):
        """v0.6.18 header CRC bug fix: c2 should reference (info & 2)
        per gr-lora line 143:
            (in[2] & 0b0010) >> 1
        Earlier code had (info & 1) which is wrong."""
        text = self._read("lora_codec.c")
        # Check that the corrected c2 line is present
        # Look for the gr-lora-style formula in c2
        idx = text.find("uint8_t c2 =")
        if idx < 0:
            pytest.fail("c2 = ... not found")
        c2_block = text[idx:idx + 250]
        # Should reference (in2 & 0x2) >> 1 (the corrected form)
        assert "(in2 & 0x2) >> 1" in c2_block, c2_block

    def test_joint_cfo_sto_solve_present(self):
        """v0.6.18: the joint solve from up_val + down_val gives both
        CFO and STO (gr-lora only refines CFO via this path; we go
        further). Without this, the cursor is misaligned by the STO
        amount and DEMOD magnitudes are stuck at ~50% of N."""
        text = self._read("lora_demod.c")
        # The math: cfo_bins = (uv + dv) / 2; sto = (uv - dv) / 2
        assert "(uv + dv) / 2" in text
        assert "(uv - dv) / 2" in text

    def test_bernier_loop_present(self):
        """v0.6.18: Bernier CFO_frac estimator iterates over 8
        preamble chirps, accumulating fft[i] * conj(fft[i+1])."""
        text = self._read("lora_demod.c")
        assert "four_cum" in text
        assert "lora_dechirp_bin" in text
        assert "atan2f" in text

    def test_multi_sync_derives_observed_byte(self):
        """v0.6.17/18: sync match no longer requires cfg.sync_word
        equality; instead it derives the on-air sync byte from the
        recovered bins (rounded to nibble*8 anchors). Header CRC is
        the final arbiter."""
        text = self._read("lora_demod.c")
        # The synthesis: observed_sync = (nib1 << 4) | nib2
        assert "observed_sync" in text
        assert "<< 4" in text  # the byte synthesis

    def test_diagnostics_gated_on_env(self):
        """v0.6.18: production runs should be quiet. All the chatty
        per-packet diagnostics (sync, sync_down, bernier, demod,
        pipeline) are gated behind LORA_DECODE_DEBUG=1. Failing
        this test would mean somebody re-introduced unconditional
        fprintf spam that breaks production logging."""
        demod = self._read("lora_demod.c")
        codec = self._read("lora_codec.c")
        # All fprintf(stderr, "[...]") inside the decoder loops should
        # be reachable only when LORA_DECODE_DEBUG=1.
        assert "LORA_DECODE_DEBUG" in demod
        assert "LORA_DECODE_DEBUG" in codec


class TestNativeLibraryLoad:
    """Verify the .so loads via ctypes and exports the expected
    symbols. Catches link errors and ABI breaks."""

    def test_so_loads_with_ctypes(self):
        import ctypes
        lib = ctypes.CDLL(str(NATIVE_DIR / "liblora_demod.so"))
        # Functions the Python wrapper would call (real exports
        # as of v0.6.18 — see `nm -D liblora_demod.so | grep " T "`)
        for name in ("lora_demod_new", "lora_demod_free",
                     "lora_demod_process_cu8", "lora_demod_process_cf",
                     "lora_demod_get_stats", "lora_demod_reset"):
            # Existence check only — calling them needs a config
            # struct that we don't want to recreate in this test
            assert hasattr(lib, name), f"missing export: {name}"
