"""v0.6.10 — passive LoRa-survey piggyback expansion.

Phase 1 of the LoRa coverage plan: opportunistically attach the
lora_survey sidecar to ANY band already running a shared rtl_tcp
fanout in the 902-928 MHz range. Previously only the 915_ism band
(centered at 915 MHz) had survey enabled; the 915_ism_r900 band's
fanout at 912.6 MHz was producing IQ for rtlamr but the survey
wasn't tapping it.

Real-world impact: the user's metatron deployment runs Meshtastic on
a non-default channel at 913.125 MHz, which falls inside r900's
±960 kHz window but OUTSIDE 915_ism's. v0.6.10 closes that gap.

Phase 2 (active channel-hop on a dedicated dongle) is intentionally
not covered here — that lands later as a separate strategy.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from rfcensus.config.loader import _load_builtin_bands


# ---------------------------------------------------------------------
# Config wiring: r900 band must opt in
# ---------------------------------------------------------------------


class TestR900BandLoraSurveyEnabled:
    """The 915_ism_r900 band must declare lora_survey = true so the
    strategy launches the sidecar against its shared fanout."""

    def test_r900_band_has_lora_survey_flag(self):
        bands = _load_builtin_bands("US")
        r900 = next((b for b in bands if b.id == "915_ism_r900"), None)
        assert r900 is not None, "915_ism_r900 band missing from builtin US config"
        assert r900.lora_survey is True, (
            "915_ism_r900 must declare lora_survey = true so the survey "
            "piggybacks on its 912.6 MHz fanout. Without this we lose "
            "passive coverage of LoRa traffic in the lower half of the "
            "ISM band (slots 35-40 in the LongFast plan)."
        )

    def test_915_ism_band_still_has_lora_survey(self):
        """Don't accidentally regress the original survey path."""
        bands = _load_builtin_bands("US")
        primary = next((b for b in bands if b.id == "915_ism"), None)
        assert primary is not None
        assert primary.lora_survey is True

    def test_combined_passive_coverage_window(self):
        """Sanity: the two lora_survey bands together cover most of
        the 911-916 MHz window, including the user's 913.125 MHz
        deployment."""
        bands = _load_builtin_bands("US")
        survey_bands = [b for b in bands if b.lora_survey]
        assert len(survey_bands) >= 2, (
            f"expected ≥2 bands with lora_survey enabled, got {len(survey_bands)}"
        )

        # Each band's usable LoRa-survey window is ±960 kHz (80% of
        # 2.4 MHz sample rate) around its center frequency.
        usable_ranges = []
        for b in survey_bands:
            center = (b.freq_low + b.freq_high) // 2
            usable_ranges.append((center - 960_000, center + 960_000))

        def covered(freq_hz: int) -> bool:
            return any(lo <= freq_hz <= hi for lo, hi in usable_ranges)

        # The user's deployment at 913.125 MHz must be covered.
        assert covered(913_125_000), (
            f"913.125 MHz (user's Meshtastic channel) not covered by any "
            f"lora_survey band's usable window: {usable_ranges}"
        )
        # MediumSlow default at 914.875 MHz must be covered.
        assert covered(914_875_000), (
            f"914.875 MHz (MediumSlow default) not covered: {usable_ranges}"
        )
        # The FACTORY default at 906.875 MHz is intentionally NOT
        # covered — that needs Phase 2 (active channel-hop).
        # Document the gap rather than silently accepting it:
        assert not covered(906_875_000), (
            "906.875 MHz (LongFast factory default) is unexpectedly "
            "covered by passive scanning — if a new band added this, "
            "great; otherwise this assertion is guarding the documented "
            "limitation that Phase 1 doesn't reach the band's lower edge."
        )


# ---------------------------------------------------------------------
# Strategy wiring: DecoderOnlyStrategy must launch the survey too
# ---------------------------------------------------------------------


class TestDecoderOnlyStrategyAttachesSurvey:
    """Before v0.6.10, only DecoderPrimaryStrategy launched the
    lora_survey sidecar. The r900 band uses DecoderOnlyStrategy, so
    flipping its lora_survey flag wasn't enough — the strategy class
    also had to be wired."""

    @pytest.mark.asyncio
    async def test_decoder_only_strategy_starts_survey_when_flag_set(
        self, monkeypatch
    ):
        from rfcensus.config.schema import BandConfig
        from rfcensus.engine.strategy import DecoderOnlyStrategy

        survey_called = []

        async def fake_run_lora_survey(band, ctx):
            survey_called.append(band.id)
            return None

        async def fake_run_decoder(band, decoder, ctx):
            return None

        monkeypatch.setattr(
            "rfcensus.engine.strategy._run_lora_survey",
            fake_run_lora_survey,
        )
        monkeypatch.setattr(
            "rfcensus.engine.strategy._run_decoder_on_band",
            fake_run_decoder,
        )

        # Mock _pick_decoders to return one fake decoder so we
        # don't bail on "no usable decoders".
        fake_decoder = MagicMock()
        fake_decoder.name = "rtlamr"
        monkeypatch.setattr(
            "rfcensus.engine.strategy._pick_decoders",
            lambda band, ctx, allowed_decoders=None: [fake_decoder],
        )

        # Patch out the 2-second sleep so the test runs fast.
        import asyncio as aio
        original_sleep = aio.sleep

        async def fast_sleep(secs):
            return await original_sleep(0)

        monkeypatch.setattr("rfcensus.engine.strategy.asyncio.sleep", fast_sleep)

        band = BandConfig(
            id="test_band",
            name="test",
            freq_low=911_000_000,
            freq_high=914_000_000,
            region="US",
            suggested_decoders=["rtlamr"],
            strategy="decoder_only",
            lora_survey=True,
        )
        ctx = MagicMock()

        strategy = DecoderOnlyStrategy()
        await strategy.execute(band, ctx)

        assert "test_band" in survey_called, (
            "DecoderOnlyStrategy did not launch lora_survey despite "
            "band.lora_survey = True. The v0.6.10 refactor must call "
            "_maybe_attach_lora_survey from this strategy."
        )

    @pytest.mark.asyncio
    async def test_decoder_only_strategy_skips_survey_when_flag_unset(
        self, monkeypatch
    ):
        """Bands without lora_survey = true must NOT spawn a survey
        task. (The flag is opt-in.)"""
        from rfcensus.config.schema import BandConfig
        from rfcensus.engine.strategy import DecoderOnlyStrategy

        survey_called = []

        async def fake_run_lora_survey(band, ctx):
            survey_called.append(band.id)
            return None

        async def fake_run_decoder(band, decoder, ctx):
            return None

        monkeypatch.setattr(
            "rfcensus.engine.strategy._run_lora_survey",
            fake_run_lora_survey,
        )
        monkeypatch.setattr(
            "rfcensus.engine.strategy._run_decoder_on_band",
            fake_run_decoder,
        )
        fake_decoder = MagicMock()
        fake_decoder.name = "rtlamr"
        monkeypatch.setattr(
            "rfcensus.engine.strategy._pick_decoders",
            lambda band, ctx, allowed_decoders=None: [fake_decoder],
        )

        band = BandConfig(
            id="not_survey", name="test",
            freq_low=900_000_000, freq_high=901_000_000,
            region="US", suggested_decoders=["rtlamr"],
            strategy="decoder_only",
            lora_survey=False,
        )
        ctx = MagicMock()

        strategy = DecoderOnlyStrategy()
        await strategy.execute(band, ctx)

        assert survey_called == [], (
            f"survey ran on band without lora_survey flag: {survey_called}"
        )


# ---------------------------------------------------------------------
# Real-world fixture: regression against captured Meshtastic IQ
# ---------------------------------------------------------------------


FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"
FIXTURE_PATH = FIXTURE_DIR / "meshtastic_real_913_5mhz.cu8"
FIXTURE_MANIFEST = FIXTURE_DIR / "meshtastic_real_913_5mhz.manifest.json"


@pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason="fixture meshtastic_real_913_5mhz.cu8 not present",
)
class TestMeshtasticRealCaptureRegression:
    """Regression test against a real RTL-SDR capture of Meshtastic
    SF9/250kHz traffic at 913.125 MHz, captured with a 915 whip
    antenna. Catches dechirp classifier regressions that synthetic
    tests miss (LDR, real frequency offset, multipath, etc.)."""

    @classmethod
    def setup_class(cls):
        cls.manifest = json.loads(FIXTURE_MANIFEST.read_text())
        raw = np.fromfile(str(FIXTURE_PATH), dtype=np.uint8)
        scaled = (raw.astype(np.float32) - 127.5) / 127.5
        cls.samples = (scaled[0::2] + 1j * scaled[1::2]).astype(np.complex64)

    def _window(self, label: str) -> np.ndarray:
        for w in self.manifest["windows"]:
            if w["label"] == label:
                return self.samples[w["fixture_start_sample"]:w["fixture_end_sample"]]
        raise KeyError(f"window {label} not in manifest")

    def test_real_burst_detected_as_meshtastic_sf9(self):
        """A clear-signal burst window must produce at least one hit
        labeled SF9 / 250 kHz, matching what we visually verified is
        present in the spectrogram."""
        from rfcensus.spectrum.in_window_survey import survey_iq_window

        win = self._window("burst_at_3.75s")
        hits = survey_iq_window(
            win,
            sample_rate=self.manifest["sample_rate"],
            capture_center_hz=self.manifest["center_freq_hz"],
        )
        assert len(hits) >= 1, "expected at least 1 hit on real LoRa burst"
        # Must include at least one SF9/250kHz hit
        sf9_250 = [
            h for h in hits
            if h.bandwidth_hz == 250_000
            and getattr(h.chirp_analysis, "estimated_sf", None) == 9
        ]
        assert sf9_250, (
            f"no SF9/250kHz hits among {len(hits)} detections — "
            f"expected at least one matching the visually-verified "
            f"Meshtastic medium_fast traffic in this capture"
        )

    def test_quiet_baseline_produces_no_hits(self):
        """A window known to contain no LoRa must produce no hits.
        Guards against the classifier learning to hallucinate."""
        from rfcensus.spectrum.in_window_survey import survey_iq_window

        win = self._window("quiet_baseline_16s")
        hits = survey_iq_window(
            win,
            sample_rate=self.manifest["sample_rate"],
            capture_center_hz=self.manifest["center_freq_hz"],
        )
        assert hits == [], (
            f"false positive: quiet-baseline window produced {len(hits)} hits"
        )

    def test_majority_of_burst_windows_detect(self):
        """Across all burst windows in the fixture, the classifier
        should hit on the majority. Set the threshold conservatively
        (≥75%) so single-window flakiness from edge effects doesn't
        break CI, but real classifier regressions still trigger.
        """
        from rfcensus.spectrum.in_window_survey import survey_iq_window

        burst_labels = [
            w["label"] for w in self.manifest["windows"]
            if not w["label"].startswith("quiet")
        ]
        hit_count = 0
        for label in burst_labels:
            hits = survey_iq_window(
                self._window(label),
                sample_rate=self.manifest["sample_rate"],
                capture_center_hz=self.manifest["center_freq_hz"],
            )
            if any(
                getattr(h.chirp_analysis, "estimated_sf", None) == 9
                for h in hits
            ):
                hit_count += 1

        rate = hit_count / len(burst_labels)
        assert rate >= 0.75, (
            f"detection rate on real-Meshtastic fixture dropped to "
            f"{rate:.0%} ({hit_count}/{len(burst_labels)}). The "
            f"dechirp classifier may have regressed — investigate "
            f"with the spectrogram tools in /home/claude/."
        )
