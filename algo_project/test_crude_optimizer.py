from dataclasses import replace

import pandas as pd

from scripts.optimize_crude_parameters import Candidate, optimize, prepare_trades, replay_trade


def _historical_frame() -> pd.DataFrame:
    rows = []
    for index in range(100):
        trending = index % 2 == 0
        rows.append({
            "Entry Timestamp": f"2026-01-{1 + index // 10:02d} {index % 10:02d}:00:00",
            "PnL": 120.0 if trending else -80.0,
            "Entry ATR": 5.0,
            "Entry Volume": 200.0 if trending else 50.0,
            "Entry OI_Change": 100.0 if trending else 10.0,
            "Entry Market Regime": "TRENDING" if trending else "SIDEWAYS",
            "Entry Price": 100.0,
        })
    return pd.DataFrame(rows)


def test_optimizer_uses_untouched_holdout_and_approves_stable_filter():
    candidate, report, approved = optimize(
        _historical_frame(), folds=4, min_trades_per_fold=4
    )

    assert approved
    assert report["holdout"]["win_rate"] == 1.0
    assert report["risk_replay_enabled"] is False
    assert candidate.allowed_regimes == ("TRENDING",)


def test_replay_trade_models_target_stop_and_trailing_in_order():
    frame = prepare_trades(_historical_frame().assign(**{
        "Intratrade Option Prices": ["[100, 106, 104]"] * 100,
    }))
    candidate = Candidate(0, 0, 0, 100, ("TRENDING", "SIDEWAYS"), 1.0, 0.0, 2.0, 5.0)

    assert replay_trade(frame.iloc[0], replace(candidate, risk_reward_ratio=1.0)) == 5.0
    stop_row = frame.iloc[0].copy()
    stop_row["price_path"] = [100.0, 94.0]
    assert replay_trade(stop_row, candidate) == -5.0
    trailing_row = frame.iloc[0].copy()
    trailing_row["price_path"] = [100.0, 112.0, 101.0]
    assert replay_trade(trailing_row, replace(candidate, risk_reward_ratio=4.0)) == 2.0


def test_optimizer_uses_legacy_rows_and_freezes_missing_atr_and_regime_gates():
    frame = _historical_frame()
    frame.loc[:, "Entry ATR"] = None
    frame.loc[:, "Entry Market Regime"] = ""

    candidate, report, _ = optimize(frame, folds=4, min_trades_per_fold=4)

    assert report["atr_rows"] == 0
    assert report["known_regime_rows"] == 0
    assert candidate.min_atr == 0.0
    assert candidate.max_atr == 1_000_000.0
    assert candidate.allowed_regimes == ("TRENDING", "SIDEWAYS", "UNKNOWN")