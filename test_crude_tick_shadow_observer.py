from datetime import datetime
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import gspread
import numpy as np

from crude_tick_shadow_observer import (
    WORKSHEET_TITLE,
    CrudeTickShadowObserver,
    TickSnapshot,
)


IST = ZoneInfo("Asia/Kolkata")


def _snapshot(*, price=100.0, tick_key=1.0, timestamp=None):
    return TickSnapshot(
        timestamp=timestamp or datetime(2026, 10, 15, 10, 0, tzinfo=IST),
        tick_key=tick_key,
        trade_id="CRUDEOIL7200CE|entry",
        symbol="CRUDEOIL7200CE",
        option_type="CE",
        live_price=price,
        entry_price=100.0,
        atr=2.0,
        baseline_sl=90.0,
        baseline_tp=120.0,
    )


def test_shadow_action_maps_to_hypothetical_levels_and_premium_pnl():
    observer = CrudeTickShadowObserver()
    observer._model = Mock()
    observer._model.predict.return_value = (
        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        None,
    )

    row = observer._predict_snapshot(_snapshot(price=99.0), now=_snapshot().timestamp)

    assert row["PPO Suggested SL"] == 99.5
    assert row["PPO Trail Activation"] == 112.0
    assert row["Hypothetical PnL Points"] == -1.0
    assert row["Exit Reason"] == "PPO_STOP_LOSS"
    observer._model.predict.assert_called_once()


def test_shadow_trails_after_fixed_activation_instead_of_booking_at_activation():
    observer = CrudeTickShadowObserver()
    observer._model = Mock()
    observer._model.predict.return_value = (
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        None,
    )
    below = _snapshot(price=105.0, tick_key=0.5)
    below_row = observer._predict_snapshot(below, now=below.timestamp)
    assert below_row["PPO Suggested SL"] == 99.5  # still inactive below +12 points

    first = _snapshot(price=112.0)

    activation_row = observer._predict_snapshot(first, now=first.timestamp)
    assert activation_row is not None
    assert activation_row["Exit Reason"] == ""
    assert activation_row["PPO Suggested SL"] == 109.8

    second = _snapshot(price=115.0, tick_key=2.0)
    trailing_row = observer._predict_snapshot(second, now=second.timestamp)
    assert trailing_row is not None
    assert trailing_row["Exit Reason"] == ""
    assert trailing_row["PPO Suggested SL"] == 112.8

    third = _snapshot(price=112.7, tick_key=3.0)
    exit_row = observer._predict_snapshot(third, now=third.timestamp)
    assert exit_row is not None
    assert exit_row["Exit Reason"] == "PPO_TRAILING_STOP"


def test_shadow_observer_deduplicates_ticks_and_stops_after_hypothetical_exit():
    observer = CrudeTickShadowObserver()
    observer._model = Mock()
    observer._model.predict.return_value = (
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        None,
    )
    snapshot = _snapshot(price=100.0)

    assert observer._predict_snapshot(snapshot, now=snapshot.timestamp) is not None
    assert observer._predict_snapshot(snapshot, now=snapshot.timestamp) is None
    assert observer._predict_snapshot(
        _snapshot(price=99.0, tick_key=2.0), now=snapshot.timestamp
    )["Exit Reason"] == "PPO_STOP_LOSS"
    assert observer._predict_snapshot(
        _snapshot(price=101.0, tick_key=3.0), now=snapshot.timestamp
    ) is None
    assert observer._model.predict.call_count == 2


def test_out_of_hours_and_weekend_ticks_are_not_queued():
    observer = CrudeTickShadowObserver()
    closed = _snapshot(timestamp=datetime(2026, 10, 15, 8, 59, tzinfo=IST))
    weekend = _snapshot(timestamp=datetime(2026, 10, 17, 10, 0, tzinfo=IST))

    assert observer.submit_tick(
        timestamp=closed.timestamp,
        tick_key=closed.tick_key,
        trade_id=closed.trade_id,
        symbol=closed.symbol,
        option_type=closed.option_type,
        live_price=closed.live_price,
        entry_price=closed.entry_price,
        atr=closed.atr,
        baseline_sl=closed.baseline_sl,
        baseline_tp=closed.baseline_tp,
    ) is False
    assert observer.submit_tick(
        timestamp=weekend.timestamp,
        tick_key=weekend.tick_key,
        trade_id=weekend.trade_id,
        symbol=weekend.symbol,
        option_type=weekend.option_type,
        live_price=weekend.live_price,
        entry_price=weekend.entry_price,
        atr=weekend.atr,
        baseline_sl=weekend.baseline_sl,
        baseline_tp=weekend.baseline_tp,
    ) is False
    assert observer._snapshots.empty()


def test_prediction_is_rejected_outside_market_hours():
    observer = CrudeTickShadowObserver()
    observer._model = Mock()
    snapshot = _snapshot()

    assert observer._predict_snapshot(
        snapshot, now=datetime(2026, 10, 15, 23, 30, tzinfo=IST)
    ) is None
    observer._model.predict.assert_not_called()


def test_shadow_rows_use_only_the_dedicated_worksheet():
    observer = CrudeTickShadowObserver()
    worksheet = Mock()
    worksheet.get_all_values.return_value = []
    spreadsheet = Mock()
    spreadsheet.worksheet.side_effect = gspread.exceptions.WorksheetNotFound(
        WORKSHEET_TITLE
    )
    spreadsheet.add_worksheet.return_value = worksheet

    with patch("sheets_logger.get_gspread_client") as get_client:
        get_client.return_value.open.return_value = spreadsheet
        observer._append_shadow_row({"Timestamp": "2026-10-15T10:00:00+05:30"})

    get_client.return_value.open.assert_called_once_with("Crude_Algo_Trade_Logs")
    spreadsheet.add_worksheet.assert_called_once_with(
        title="PPO_Shadow_Logs", rows=1000, cols=11
    )
    assert worksheet.append_row.call_count == 2
    assert not any(call[0] == "sheet1" for call in spreadsheet.method_calls)
