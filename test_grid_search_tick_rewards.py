import json

import numpy as np
import pandas as pd
import pytest

from crude_tick_env import CrudeTickEnv
from grid_search_tick_rewards import (
    chronological_three_way_split,
    rank_reward_trials,
)


def _trade(day, path):
    return {
        "Entry Timestamp": f"2026-09-{day:02d} 10:00:00",
        "Exit Timestamp": f"2026-09-{day:02d} 10:05:00",
        "Symbol": "CRUDEOIL_OPTION",
        "Action": "CE",
        "Entry Price": 100.0,
        "Exit Price": path[-1],
        "Entry ATR": 2.0,
        "PnL": (path[-1] - 100.0) * 100,
        "Intratrade Option Prices": json.dumps(path),
    }


def test_tick_environment_combines_separately_scaled_pnl_and_r_rewards():
    env = CrudeTickEnv(
        pd.DataFrame([_trade(1, [100.0, 102.2])]),
        pnl_reward_weight=1.0,
        rr_reward_weight=2.0,
        pnl_reward_scale=2.2,
    )
    env.reset()

    _, reward, terminated, _, info = env.step(np.asarray([0.0, 1.0, 0.0]))

    assert terminated
    assert info["pnl_points"] == pytest.approx(2.2)
    assert info["pnl_r"] == pytest.approx(1.0)
    assert info["pnl_reward_component"] == pytest.approx(1.0)
    assert info["rr_reward_component"] == pytest.approx(1.0)
    assert reward == pytest.approx(3.0)


def test_reward_weights_and_scale_are_validated():
    data = pd.DataFrame([_trade(1, [100.0, 101.0])])

    with pytest.raises(ValueError, match="not both zero"):
        CrudeTickEnv(data, pnl_reward_weight=0.0, rr_reward_weight=0.0)
    with pytest.raises(ValueError, match="scale must be finite and positive"):
        CrudeTickEnv(data, pnl_reward_scale=0.0)


def test_three_way_split_keeps_validation_and_final_periods_chronological():
    trades = CrudeTickEnv(
        pd.DataFrame(
            [_trade(day, [100.0, 101.0]) for day in (5, 1, 4, 2, 3, 6, 7, 8, 9, 10)]
        )
    ).data

    training, validation, holdout = chronological_three_way_split(trades)

    assert len(training) == 6
    assert len(validation) == 2
    assert len(holdout) == 2
    assert training["_entry_timestamp"].max() < validation["_entry_timestamp"].min()
    assert validation["_entry_timestamp"].max() < holdout["_entry_timestamp"].min()


def test_profitable_validation_trials_rank_before_unprofitable_trials():
    trials = [
        {
            "validation": {
                "total_simulated_pnl_points": -1.0,
                "total_simulated_pnl_r": 5.0,
                "profit_factor_r": 2.0,
                "win_rate": 0.8,
            }
        },
        {
            "validation": {
                "total_simulated_pnl_points": 1.0,
                "total_simulated_pnl_r": 0.2,
                "profit_factor_r": 1.1,
                "win_rate": 0.4,
            }
        },
        {
            "validation": {
                "total_simulated_pnl_points": 2.0,
                "total_simulated_pnl_r": 0.5,
                "profit_factor_r": 1.3,
                "win_rate": 0.5,
            }
        },
    ]

    ranked = rank_reward_trials(trials)

    assert ranked[0]["validation"]["total_simulated_pnl_points"] == 2.0
    assert ranked[1]["validation"]["total_simulated_pnl_points"] == 1.0
    assert ranked[2]["validation"]["total_simulated_pnl_points"] == -1.0
