"""v0.6.12 — DecoderOnlyStrategy result aggregation handles
LoraSurveyStats without crashing.

Background: v0.6.10 added _maybe_attach_lora_survey to both
DecoderPrimaryStrategy and DecoderOnlyStrategy so the 915_ism_r900
band (decoder_only) could piggy-back its survey on the rtlamr fanout.
DecoderPrimaryStrategy's result loop already special-cased the
LoraSurveyStats return type, but DecoderOnlyStrategy's loop blindly
called `.decodes_emitted` on every result. That field does NOT exist
on LoraSurveyStats (it has `detections_emitted` instead — survey hits
flow as DetectionEvents through the bus, not as decoder output).

Real-world impact: every wave that ran a decoder_only band with
lora_survey enabled raised
  "ERROR strategy raised: 'LoraSurveyStats' object has no attribute
   'decodes_emitted'"
at wave teardown. The survey itself worked fine (we got 40 detections
in metatron's wave 1) but the strategy result couldn't be aggregated
cleanly into the session report.

This test pins the fix: a DecoderOnlyStrategy that returns both a
decoder result and a LoraSurveyStats result must aggregate without
crash, count decoder decodes, and surface survey errors.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest


@pytest.mark.asyncio
async def test_decoder_only_aggregates_lora_survey_without_crash(monkeypatch):
    """The crash repro from the metatron 0.6.11 run.

    Before fix: DecoderOnlyStrategy.execute() iterated results from
    asyncio.gather() and called result.decodes_emitted on every one,
    including the LoraSurveyStats which doesn't have that attribute,
    raising AttributeError.

    After fix: aggregation distinguishes the two return types via
    hasattr (mirroring DecoderPrimaryStrategy's existing logic).
    """
    from rfcensus.config.schema import BandConfig
    from rfcensus.engine.strategy import DecoderOnlyStrategy
    from rfcensus.engine.lora_survey_task import LoraSurveyStats

    # Mock decoder-side: returns an object that LOOKS like a decoder
    # task result (has decodes_emitted + errors).
    decoder_result = MagicMock()
    decoder_result.decodes_emitted = 5
    decoder_result.errors = []

    async def fake_run_decoder(band, decoder, ctx):
        return decoder_result

    # Mock lora_survey: returns a LoraSurveyStats with detections_
    # emitted but NOT decodes_emitted (this is the v0.6.10 reality).
    survey_stats = LoraSurveyStats()
    survey_stats.detections_emitted = 40
    survey_stats.errors = ["a survey-level error to surface"]

    async def fake_run_lora_survey(band, ctx):
        return survey_stats

    monkeypatch.setattr(
        "rfcensus.engine.strategy._run_decoder_on_band", fake_run_decoder
    )
    monkeypatch.setattr(
        "rfcensus.engine.strategy._run_lora_survey", fake_run_lora_survey
    )
    fake_decoder = MagicMock()
    fake_decoder.name = "rtlamr"
    monkeypatch.setattr(
        "rfcensus.engine.strategy._pick_decoders",
        lambda band, ctx, allowed_decoders=None: [fake_decoder],
    )
    # Patch the 2s sleep in _maybe_attach_lora_survey for fast tests.
    original_sleep = asyncio.sleep
    async def fast_sleep(secs):
        return await original_sleep(0)
    monkeypatch.setattr("rfcensus.engine.strategy.asyncio.sleep", fast_sleep)

    band = BandConfig(
        id="915_ism_r900_test",
        name="r900",
        freq_low=911_400_000,
        freq_high=913_800_000,
        region="US",
        suggested_decoders=["rtlamr"],
        strategy="decoder_only",
        lora_survey=True,
    )
    ctx = MagicMock()

    strategy = DecoderOnlyStrategy()
    # Before fix: this raised AttributeError partway through aggregation.
    # After fix: returns cleanly with both results merged.
    result = await strategy.execute(band, ctx)

    # Decoder decodes counted normally
    assert result.decodes_emitted == 5, (
        f"expected 5 decoder decodes counted; got {result.decodes_emitted}. "
        f"If 0, the fix accidentally also stopped counting real decoder "
        f"output."
    )
    # Survey error surfaced
    assert "a survey-level error to surface" in result.errors, (
        f"expected survey error to be surfaced in result.errors; "
        f"got {result.errors}"
    )
    # Survey detections do NOT count as decodes (they go through bus
    # as DetectionEvents). Critical: detections_emitted=40 must NOT
    # leak into result.decodes_emitted.
    assert result.decodes_emitted == 5, (
        "survey detections must not be added to decoder decode count"
    )


@pytest.mark.asyncio
async def test_decoder_only_handles_survey_with_no_errors(monkeypatch):
    """When the survey runs cleanly (no errors), aggregation must
    still not crash and decoder counts must be preserved."""
    from rfcensus.config.schema import BandConfig
    from rfcensus.engine.strategy import DecoderOnlyStrategy
    from rfcensus.engine.lora_survey_task import LoraSurveyStats

    decoder_result = MagicMock()
    decoder_result.decodes_emitted = 0
    decoder_result.errors = []

    survey_stats = LoraSurveyStats()
    survey_stats.detections_emitted = 12
    # No errors

    async def fake_decoder(b, d, c): return decoder_result
    async def fake_survey(b, c): return survey_stats

    monkeypatch.setattr(
        "rfcensus.engine.strategy._run_decoder_on_band", fake_decoder
    )
    monkeypatch.setattr(
        "rfcensus.engine.strategy._run_lora_survey", fake_survey
    )
    fake_dec = MagicMock(); fake_dec.name = "rtlamr"
    monkeypatch.setattr(
        "rfcensus.engine.strategy._pick_decoders",
        lambda b, c, allowed_decoders=None: [fake_dec],
    )
    orig_sleep = asyncio.sleep
    monkeypatch.setattr(
        "rfcensus.engine.strategy.asyncio.sleep",
        lambda s: orig_sleep(0),
    )

    band = BandConfig(
        id="quiet_test",
        name="quiet",
        freq_low=911_400_000, freq_high=913_800_000,
        region="US", suggested_decoders=["rtlamr"],
        strategy="decoder_only", lora_survey=True,
    )

    result = await DecoderOnlyStrategy().execute(band, MagicMock())
    assert result.decodes_emitted == 0
    assert result.errors == []
