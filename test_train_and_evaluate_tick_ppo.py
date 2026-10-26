import json

import numpy as np
import pandas as pd
import pytest

from crude_tick_env import CrudeTickEnv
from train_and_evaluate_tick_ppo import (
    chronological_split,
    evaluate_model,
    load_sheet1_tick_data,
    summarize_results,
)


def _trade(entry_time, path, entry=100.0, atr=2.0):
    return {
        "Entry Timestamp": entry_time,
        "Exit Timestamp": f"{entry_time[:10]} 10:05:00",
        "Symbol": "CRUDEOIL_OPTION",
        "Action": "CE",
        "Entry Price": entry,
        "Exit Price": path[-1],
        "Entry ATR": atr,
        "PnL": (path[-1] - entry) * 100,
        "Intratrade Option Prices": json.dumps(path),
    }


class _FixedPolicy:
    def predict(self, observation, deterministic=True):
        assert deterministic
        assert observation.ndim == 1
        # SL = 0.25 ATR, trailing gap = 1R, buffer = 0 ATR.
        action = np.asarray([0.0, 1.0, 0.0])
        return action, None


def test_chronological_split_keeps_latest_fifth_for_holdout():
    trades = CrudeTickEnv(
        pd.DataFrame(
            [
                _trade("2026-09-03 10:00:00", [100.0, 101.0]),
                _trade("2026-09-01 10:00:00", [100.0, 101.0]),
                _trade("2026-09-02 10:00:00", [100.0, 101.0]),
                _trade("2026-09-05 10:00:00", [100.0, 101.0]),
                _trade("2026-09-04 10:00:00", [100.0, 101.0]),
            ]
        )
    ).data

    train, holdout = chronological_split(trades)

    assert len(train) == 4
    assert len(holdout) == 1
    assert holdout.iloc[0]["_entry_timestamp"] == pd.Timestamp(
        "2026-09-05 10:00:00", tz="UTC"
    )


def test_evaluation_reports_one_result_for_each_holdout_trade():
    holdout = CrudeTickEnv(
        pd.DataFrame(
            [
                _trade("2026-09-01 10:00:00", [100.0, 99.0]),
                _trade("2026-09-02 10:00:00", [100.0, 104.0]),
            ]
        )
    ).data

    results = evaluate_model(_FixedPolicy(), holdout)
    summary = summarize_results(results)

    assert summary["trades_tested"] == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["total_simulated_pnl_points"] == pytest.approx(3.0)
    assert summary["exit_reasons"] == {"stop_loss": 1, "recorded_exit": 1}
    assert results[0]["simulated_pnl_points"] == pytest.approx(-1.0)
    assert results[1]["simulated_pnl_points"] == pytest.approx(4.0)
    assert summary["total_simulated_pnl_r"] == pytest.approx(3.0 / 2.2)
    assert summary["stop_exit_count"] == 1
    assert summary["worst_loss_r"] == pytest.approx(-1.0 / 2.2)
    assert summary["reached_1r"] == 1
    assert summary["reached_2r"] == 0
    assert summary["exited_at_or_above_2r"] == 0
    assert results[1]["max_favorable_r"] == pytest.approx(4.0 / 2.2)
    assert results[1]["path_peak_r_hypothetical"] == pytest.approx(4.0 / 2.2)


def test_main_refuses_to_overwrite_existing_report(tmp_path, monkeypatch):
    import sys

    import train_and_evaluate_tick_ppo as module

    existing = tmp_path / "model_holdout_report.json"
    existing.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["prog", "--model-path", str(tmp_path / "model")]
    )

    with pytest.raises(SystemExit):
        module.main()

    assert existing.read_text(encoding="utf-8") == "keep"


def test_sheet1_loader_fetches_only_populated_crude_rows(monkeypatch):
    headers = [
        "Symbol",
        "Entry Timestamp",
        "Exit Timestamp",
        "Entry Price",
        "Exit Price",
        "Entry ATR",
        "Intratrade Option Prices",
    ]
    values = [
        headers,
        ["CRUDEOIL_OPTION", "2026-09-01 10:00:00", "2026-09-01 10:05:00", "100", "101", "2", "[100, 101]"],
        ["NIFTY_OPTION", "2026-09-02 10:00:00", "2026-09-02 10:05:00", "100", "101", "2", "[100, 101]"],
        ["", "", "", "", "", "", ""],
    ]

    class FakeWorksheet:
        def get_all_values(self):
            return values

    class FakeSpreadsheet:
        sheet1 = FakeWorksheet()

        def open(self, _name):
            return self

    import sheets_logger

    monkeypatch.setattr(sheets_logger, "get_gspread_client", FakeSpreadsheet)

    frame = load_sheet1_tick_data()

    assert len(frame) == 1
    assert frame.iloc[0]["Symbol"] == "CRUDEOIL_OPTION"
