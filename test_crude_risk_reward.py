from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from trading_crude import CrudeOptionBuyer, Position


@pytest.mark.parametrize("ratio", [1.0, 2.0, 3.0])
def test_option_target_distance_matches_configured_risk_reward_ratio(ratio):
    engine = CrudeOptionBuyer.__new__(CrudeOptionBuyer)
    engine.settings = SimpleNamespace(
        crude_atr_multiplier=1.5,
        buffer_points=1.0,
        max_stop_premium_fraction=0.5,
        crude_risk_reward_ratio=ratio,
    )

    levels = engine._risk_levels(entry_price=100.0, atr=4.0)

    assert levels is not None
    risk = 100.0 - levels["stop_loss"]
    reward = levels["target_price"] - 100.0
    assert reward == pytest.approx(ratio * risk)


def test_futures_structure_reference_cannot_close_option_before_premium_target():
    engine = CrudeOptionBuyer.__new__(CrudeOptionBuyer)
    engine.settings = SimpleNamespace(
        crude_trailing_activation_points=12.0,
        crude_trailing_breakeven_buffer_points=1.0,
        futures_tick_timeout_seconds=5.0,
    )
    engine.current_price = 7_210.0
    engine.current_timestamp = datetime.now().astimezone()
    engine._position = Position(
        side="CE",
        strike=7_200,
        entry_price=100.0,
        entry_time=datetime.now().astimezone(),
        stop_loss=90.0,
        target_price=120.0,
        trailing_stop=90.0,
        atr_value=5.0,
        futures_entry=7_200.0,
        futures_entry_oi=1_000.0,
        scenario="Long Buildup",
        futures_target=7_205.0,
    )
    engine._current_option_price = Mock(return_value=105.0)
    engine._option_ltp_time = 1.0
    engine._trail_distance = 3.0
    engine._exit_confirmation = None
    engine._close_position = Mock()
    engine._persist_state = Mock()

    engine._update_position_management()

    engine._close_position.assert_not_called()


def test_invalid_risk_reward_ratio_skips_entry_levels():
    engine = CrudeOptionBuyer.__new__(CrudeOptionBuyer)
    engine.settings = SimpleNamespace(
        crude_atr_multiplier=1.5,
        buffer_points=1.0,
        max_stop_premium_fraction=0.5,
        crude_risk_reward_ratio=0.0,
    )

    assert engine._risk_levels(entry_price=100.0, atr=4.0) is None
