"""v0.6.15 — report naming + per-decoder instrumentation + Diagnostics.

Three changes, three test groups:

  1. Report header is named after the command (`scan` / `inventory` /
     `hybrid`), not hardcoded "inventory report".

  2. StrategyResult now carries `decoder_runs: list[DecoderRunSummary]`
     so the report can show per-decoder counts + ended_reason. Without
     this, decodes=0 was ambiguous: silent band? missing binary?
     mid-run crash?

  3. Text report has a Diagnostics section near the top that buckets
     fixable failure modes (binary missing, crash, hardware loss) so
     the user notices them before scrolling through hundreds of
     mystery-carrier lines.
"""

from __future__ import annotations

from datetime import datetime, timezone


# ─────────────────────────────────────────────────────────────────────
# 1. Report header named after command
# ─────────────────────────────────────────────────────────────────────


class TestReportHeaderCommandName:
    def _build_minimal_result(self, session_id: int = 1):
        from rfcensus.engine.session import SessionResult
        from rfcensus.engine.scheduler import ExecutionPlan
        return SessionResult(
            session_id=session_id,
            started_at=datetime(2026, 4, 24, 20, 0, tzinfo=timezone.utc),
            ended_at=datetime(2026, 4, 24, 21, 0, tzinfo=timezone.utc),
            plan=ExecutionPlan(waves=[], max_parallel_per_wave=1, warnings=[], unassigned=[]),
            strategy_results=[],
            total_decodes=0,
        )

    def test_default_command_is_inventory(self):
        from rfcensus.reporting.formats.text import render_text_report
        r = self._build_minimal_result()
        text = render_text_report(r, [], [])
        assert "rfcensus inventory report" in text

    def test_command_name_scan_renders(self):
        from rfcensus.reporting.formats.text import render_text_report
        r = self._build_minimal_result()
        text = render_text_report(r, [], [], command_name="scan")
        assert "rfcensus scan report" in text
        assert "rfcensus inventory report" not in text

    def test_command_name_hybrid_renders(self):
        from rfcensus.reporting.formats.text import render_text_report
        r = self._build_minimal_result()
        text = render_text_report(r, [], [], command_name="hybrid")
        assert "rfcensus hybrid report" in text


# ─────────────────────────────────────────────────────────────────────
# 2. Per-decoder instrumentation: DecoderRunSummary on StrategyResult
# ─────────────────────────────────────────────────────────────────────


class TestDecoderRunSummary:
    def test_strategy_result_has_decoder_runs_field(self):
        from rfcensus.engine.strategy import StrategyResult
        sr = StrategyResult(band_id="x")
        assert sr.decoder_runs == []

    def test_decoder_run_summary_dataclass(self):
        from rfcensus.engine.strategy import DecoderRunSummary
        s = DecoderRunSummary(
            name="rtl_433", decodes_emitted=14,
            ended_reason="duration", errors=[],
        )
        assert s.name == "rtl_433"
        assert s.decodes_emitted == 14
        assert s.ended_reason == "duration"

    def test_decoder_runs_appears_in_text_report(self):
        from rfcensus.engine.session import SessionResult
        from rfcensus.engine.scheduler import ExecutionPlan
        from rfcensus.engine.strategy import StrategyResult, DecoderRunSummary
        from rfcensus.reporting.formats.text import render_text_report

        sr = StrategyResult(
            band_id="915_ism",
            decoders_run=["rtl_433", "rtlamr"],
            decodes_emitted=14,
            decoder_runs=[
                DecoderRunSummary(
                    name="rtl_433", decodes_emitted=14,
                    ended_reason="duration",
                ),
                DecoderRunSummary(
                    name="rtlamr", decodes_emitted=0,
                    ended_reason="binary_missing",
                ),
            ],
        )
        r = SessionResult(
            session_id=1,
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            plan=ExecutionPlan(waves=[], max_parallel_per_wave=1, warnings=[], unassigned=[]),
            strategy_results=[sr],
            total_decodes=14,
        )
        text = render_text_report(r, [], [])
        # Per-decoder lines visible
        assert "rtl_433" in text
        assert "rtlamr" in text
        # ended_reason visible per decoder
        assert "ended=duration" in text
        assert "ended=binary_missing" in text


# ─────────────────────────────────────────────────────────────────────
# 3. Diagnostics section
# ─────────────────────────────────────────────────────────────────────


class TestDiagnosticsSection:
    def _result_with(self, *strategy_results):
        from rfcensus.engine.session import SessionResult
        from rfcensus.engine.scheduler import ExecutionPlan
        return SessionResult(
            session_id=1,
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            plan=ExecutionPlan(waves=[], max_parallel_per_wave=1, warnings=[], unassigned=[]),
            strategy_results=list(strategy_results),
            total_decodes=sum(sr.decodes_emitted for sr in strategy_results),
        )

    def test_no_diagnostics_section_when_all_clean(self):
        from rfcensus.engine.strategy import StrategyResult, DecoderRunSummary
        from rfcensus.reporting.formats.text import render_text_report

        sr = StrategyResult(
            band_id="b", decoders_run=["rtl_433"],
            decodes_emitted=5,
            decoder_runs=[
                DecoderRunSummary(
                    name="rtl_433", decodes_emitted=5,
                    ended_reason="duration",
                ),
            ],
        )
        text = render_text_report(self._result_with(sr), [], [])
        assert "Diagnostics" not in text, (
            "When everything ran fine and decoded, the Diagnostics "
            "section should be omitted to keep the report compact."
        )

    def test_binary_missing_surfaces_in_cant_start_bucket(self):
        from rfcensus.engine.strategy import StrategyResult, DecoderRunSummary
        from rfcensus.reporting.formats.text import render_text_report

        sr = StrategyResult(
            band_id="pocsag_929", decoders_run=["multimon"],
            decodes_emitted=0,
            decoder_runs=[
                DecoderRunSummary(
                    name="multimon", decodes_emitted=0,
                    ended_reason="binary_missing",
                ),
            ],
        )
        text = render_text_report(self._result_with(sr), [], [])
        assert "Diagnostics" in text
        assert "couldn't start" in text
        assert "pocsag_929/multimon" in text
        assert "binary_missing" in text

    def test_crash_surfaces_in_crashed_bucket(self):
        from rfcensus.engine.strategy import StrategyResult, DecoderRunSummary
        from rfcensus.reporting.formats.text import render_text_report

        sr = StrategyResult(
            band_id="interlogix_security", decoders_run=["rtl_433"],
            decodes_emitted=0,
            decoder_runs=[
                DecoderRunSummary(
                    name="rtl_433", decodes_emitted=0,
                    ended_reason="error",
                    errors=["rtl_433 exit code 3"],
                ),
            ],
        )
        text = render_text_report(self._result_with(sr), [], [])
        assert "Diagnostics" in text
        assert "crashed mid-run" in text
        assert "interlogix_security/rtl_433" in text
        assert "rtl_433 exit code 3" in text

    def test_silent_bands_listed_separately(self):
        from rfcensus.engine.strategy import StrategyResult, DecoderRunSummary
        from rfcensus.reporting.formats.text import render_text_report

        # Two bands ran fine but produced nothing — the genuinely-
        # silent case. Should appear in Diagnostics under the
        # silent-bands bucket.
        sr1 = StrategyResult(
            band_id="ais", decoders_run=["rtl_ais"],
            decodes_emitted=0,
            decoder_runs=[
                DecoderRunSummary(
                    name="rtl_ais", decodes_emitted=0,
                    ended_reason="duration",
                ),
            ],
        )
        sr2 = StrategyResult(
            band_id="aprs_2m", decoders_run=["direwolf"],
            decodes_emitted=0,
            decoder_runs=[
                DecoderRunSummary(
                    name="direwolf", decodes_emitted=0,
                    ended_reason="duration",
                ),
            ],
        )
        text = render_text_report(self._result_with(sr1, sr2), [], [])
        assert "Diagnostics" in text
        assert "silent for the full wave" in text
        assert "ais" in text
        assert "aprs_2m" in text

    def test_band_with_decoder_silent_AND_decodes_does_not_count_as_silent(self):
        # If a band has multiple decoders and at least one produced
        # decodes, the band is not "silent" — even if other decoders
        # were quiet.
        from rfcensus.engine.strategy import StrategyResult, DecoderRunSummary
        from rfcensus.reporting.formats.text import render_text_report

        sr = StrategyResult(
            band_id="915_ism", decoders_run=["rtl_433", "rtlamr"],
            decodes_emitted=14,
            decoder_runs=[
                DecoderRunSummary(
                    name="rtl_433", decodes_emitted=14,
                    ended_reason="duration",
                ),
                DecoderRunSummary(
                    name="rtlamr", decodes_emitted=0,
                    ended_reason="duration",
                ),
            ],
        )
        text = render_text_report(self._result_with(sr), [], [])
        # No diagnostics expected at all (everything fine)
        assert "Diagnostics" not in text

    def test_diagnostics_section_appears_before_emitters(self):
        # Layout matters: user should see diagnostics first.
        from rfcensus.engine.strategy import StrategyResult, DecoderRunSummary
        from rfcensus.reporting.formats.text import render_text_report

        sr = StrategyResult(
            band_id="x", decoders_run=["rtl_433"],
            decodes_emitted=0,
            decoder_runs=[
                DecoderRunSummary(
                    name="rtl_433", decodes_emitted=0,
                    ended_reason="binary_missing",
                ),
            ],
        )
        text = render_text_report(self._result_with(sr), [], [])
        diag_pos = text.find("Diagnostics")
        emit_pos = text.find("Emitters detected")
        assert diag_pos > 0
        assert emit_pos > 0
        assert diag_pos < emit_pos, (
            "Diagnostics must appear above Emitters so the user "
            "notices fixable issues before scrolling past them."
        )
