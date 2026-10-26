import json

import numpy as np
import pandas as pd
import pytest

from crude_tick_env import CrudeTickEnv, OBSERVATION_FEATURES, prepare_tick_trades


def _trade(path, *, entry=100.0, exit_price=None, atr=2.0):
    exit_price = (path[-1] if path else entry) if exit_price is None else exit_price
    return {
        "Entry Timestamp": "2026-09-01 10:00:00",
        "Exit Timestamp": "2026-09-01 10:05:00",
        "Symbol": "CRUDEOIL_OPTION",
        "Action": "CE",
        "Entry Price": entry,
        "Exit Price": exit_price,
        "Entry ATR": atr,
        "PnL": (exit_price - entry) * 100,
        "Intratrade Option Prices": json.dumps(path),
    }


def _action(*, sl_atr=0.5, trail_gap_r=1.0, buffer_atr=0.0):
    return np.asarray(
        [
            (sl_atr - 0.25) / (4.0 - 0.25),
            (trail_gap_r - 0.5) / (1.0 - 0.5),
            buffer_atr / 2.0,
        ],
        dtype=np.float32,
    )


def test_replays_price_samples_and_stops_when_stop_is_hit_first():
    env = CrudeTickEnv(pd.DataFrame([_trade([100.0, 99.0, 103.0])]))
    observation, reset_info = env.reset()
    assert observation.shape == (len(OBSERVATION_FEATURES),)
    assert reset_info["path_timestamps_available"] is False

    _, reward, terminated, truncated, info = env.step(
        _action(sl_atr=0.5)
    )

    assert terminated
    assert not truncated
    assert info["exit_reason"] == "stop_loss"
    assert info["tick_index"] == 1
    assert info["exit_price"] == 99.0
    assert reward == pytest.approx(-1.0 / 2.2)


def test_one_r_activates_trailing_stop_and_later_pullback_locks_profit():
    env = CrudeTickEnv(pd.DataFrame([_trade([100.0, 102.2, 105.0, 102.7])]))
    env.reset()
    _, first_reward, terminated, _, first_info = env.step(
        _action(sl_atr=0.25, trail_gap_r=1.0)
    )
    assert not terminated
    assert first_info["trail_active"]
    assert first_info["trail_activation_price"] == pytest.approx(102.2)
    assert first_info["sl_price"] == pytest.approx(100.0)
    assert first_reward == 0.0

    _, second_reward, terminated, _, second_info = env.step(
        _action(sl_atr=0.25, trail_gap_r=1.0)
    )
    assert not terminated
    assert second_info["sl_price"] == pytest.approx(102.8)
    assert second_reward == 0.0

    _, final_reward, terminated, _, info = env.step(
        _action(sl_atr=0.25, trail_gap_r=1.0)
    )
    assert terminated
    assert info["exit_reason"] == "stop_loss"
    assert info["exit_price"] == pytest.approx(102.7)
    assert info["pnl_points"] == pytest.approx(2.7)
    assert first_reward + second_reward + final_reward == pytest.approx(2.7 / 2.2)


def test_initial_risk_defines_one_r_trail_activation():
    env = CrudeTickEnv(pd.DataFrame([_trade([100.0, 100.2, 100.4])]))
    _, reset_info = env.reset()
    assert reset_info["path_timestamps_available"] is False
    assert env.initial_risk_points == pytest.approx(2.2)
    assert env.trail_activation_price == pytest.approx(102.2)


def test_stop_can_tighten_but_never_widen():
    env = CrudeTickEnv(pd.DataFrame([_trade([100.0, 101.0, 100.0, 98.9])]))
    env.reset()
    _, _, terminated, _, first_info = env.step(
        _action(sl_atr=0.5)
    )
    assert not terminated
    initial_stop = first_info["sl_price"]

    _, _, terminated, _, second_info = env.step(
        _action(sl_atr=3.0)
    )
    assert not terminated
    assert second_info["sl_price"] >= initial_stop

    _, _, terminated, _, final_info = env.step(_action(sl_atr=3.0))
    assert terminated
    assert final_info["exit_reason"] == "stop_loss"
    assert final_info["sl_price"] == pytest.approx(initial_stop)


def test_recorded_exit_is_terminal_when_no_dynamic_level_was_touched():
    env = CrudeTickEnv(pd.DataFrame([_trade([100.0, 100.5, 101.0], exit_price=101.0)]))
    env.reset()
    env.step(_action(sl_atr=4.0))
    _, reward, terminated, _, info = env.step(_action(sl_atr=4.0))

    assert terminated
    assert info["exit_reason"] == "recorded_exit"
    assert info["exit_price"] == 101.0
    assert info["pnl_points"] == 1.0
    assert reward == pytest.approx(1.0 / 2.2)


def test_missing_paths_are_excluded_and_reported():
    frame = pd.DataFrame(
        [_trade([100.0, 101.0]), {**_trade([]), "Intratrade Option Prices": ""}]
    )
    with pytest.warns(UserWarning, match="1 trade row"):
        prepared = prepare_tick_trades(frame)

    assert len(prepared) == 1
    assert prepared.attrs["skipped_rows"] == {"missing price path": 1}


def test_invalid_actions_and_finished_episodes_are_rejected():
    env = CrudeTickEnv(pd.DataFrame([_trade([100.0, 101.0])]))
    env.reset()
    with pytest.raises(ValueError, match="three finite values"):
        env.step(np.asarray([1.5, 0.5, 0.5], dtype=np.float32))

    _, _, terminated, _, _ = env.step(_action(sl_atr=4.0))
    assert terminated
    with pytest.raises(RuntimeError, match="call reset"):
        env.step(_action())


def _run_to_end(path, **env_kwargs):
    env = CrudeTickEnv(pd.DataFrame([_trade(path)]), **env_kwargs)
    env.reset()
    terminated = False
    while not terminated:
        _, reward, terminated, _, info = env.step(_action(sl_atr=0.5, trail_gap_r=1.0))
    return reward, info


def test_default_reward_is_unchanged_without_target_bonus():
    reward, info = _run_to_end([100.0, 104.4])
    assert reward == pytest.approx(2.0)
    assert info["target_bonus_component"] == 0.0


def test_target_bonus_scales_with_realized_r_and_caps_at_target():
    kwargs = {"target_rr": 2.0, "target_bonus_weight": 0.5}
    reward_1r, _ = _run_to_end([100.0, 102.2], **kwargs)
    reward_2r, info_2r = _run_to_end([100.0, 104.4], **kwargs)
    reward_3r, info_3r = _run_to_end([100.0, 106.6], **kwargs)

    assert reward_1r == pytest.approx(1.0 + 0.25)
    assert reward_2r == pytest.approx(2.0 + 0.5)
    assert info_2r["target_bonus_component"] == pytest.approx(0.5)
    assert reward_3r == pytest.approx(3.0 + 0.5)
    assert info_3r["target_bonus_component"] == pytest.approx(0.5)
    assert reward_1r < reward_2r < reward_3r


def test_target_bonus_does_not_change_loss_reward():
    reward, info = _run_to_end(
        [100.0, 99.0], target_rr=2.0, target_bonus_weight=0.5
    )
    assert reward == pytest.approx(-1.0 / 2.2)
    assert info["target_bonus_component"] == 0.0


def test_touching_target_without_exiting_there_gets_no_bonus_for_the_touch():
    # Reaches 2R, then trails out at 1R: reward follows realized exit only.
    reward, info = _run_to_end(
        [100.0, 104.4, 102.2, 103.0],
        target_rr=2.0,
        target_bonus_weight=0.5,
    )
    assert info["exit_reason"] == "stop_loss"
    assert info["max_favorable_r"] == pytest.approx(2.0)
    assert info["rr_reward_component"] == pytest.approx(1.0)
    assert reward == pytest.approx(1.0 + 0.25)


def test_target_bonus_configuration_is_validated():
    data = pd.DataFrame([_trade([100.0, 101.0])])
    with pytest.raises(ValueError, match="target_rr is required"):
        CrudeTickEnv(data, target_bonus_weight=0.5)
    with pytest.raises(ValueError, match="target_rr must be"):
        CrudeTickEnv(data, target_rr=0.0)
    with pytest.raises(ValueError, match="target_bonus_weight"):
        CrudeTickEnv(data, target_rr=2.0, target_bonus_weight=-1.0)
    with pytest.raises(ValueError, match="rr_reward_weight"):
        CrudeTickEnv(
            data,
            target_rr=2.0,
            target_bonus_weight=0.5,
            rr_reward_weight=0.0,
            pnl_reward_weight=1.0,
        )


