import json
from datetime import datetime
from zoneinfo import ZoneInfo

from crude_entry_shadow_collector import CrudeEntryShadowCollector


IST = ZoneInfo("Asia/Kolkata")


def test_shadow_collector_saves_candidate_decision_and_side_adjusted_future_labels(tmp_path):
    output = tmp_path / "entry_opportunities.jsonl"
    collector = CrudeEntryShadowCollector(output)
    signal = {
        "timestamp": datetime(2026, 10, 19, 10, 0, tzinfo=IST),
        "scenario": "Short Buildup",
        "side": "PE",
        "strike": 7200,
        "futures_price": 7000.0,
        "oi": 1200.0,
        "oi_change": 100.0,
        "volume": 500.0,
        "atr": 8.0,
        "nymex_trend": "RED",
        "market_regime": "TRENDING",
        "candle_open": 7001.0,
        "candle_high": 7003.0,
        "candle_low": 6998.0,
        "candle_close": 7000.0,
    }

    signal_id = collector.record_signal(signal)
    collector.record_decision(
        signal_id, decision="SKIPPED", reason="outcome filter veto"
    )
    collector.record_futures_tick(
        datetime(2026, 10, 19, 10, 1, tzinfo=IST), 6999.0
    )
    collector.record_futures_tick(
        datetime(2026, 10, 19, 10, 3, tzinfo=IST), 6997.0
    )
    collector.record_futures_tick(
        datetime(2026, 10, 19, 10, 5, tzinfo=IST), 6996.0
    )

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == [
        "candidate",
        "decision",
        "futures_outcome",
    ]
    assert all(record["signal_id"] == signal_id for record in records)
    assert records[1]["decision"] == "SKIPPED"
    outcome = records[2]
    assert outcome["directional_return_points_1m"] == 1.0
    assert outcome["directional_return_points_3m"] == 3.0
    assert outcome["directional_return_points_5m"] == 4.0
    assert outcome["max_favorable_excursion_points"] == 4.0
    assert outcome["max_adverse_excursion_points"] == 0.0
    assert outcome["path_timestamps_available"] is True


def test_executed_trade_outcome_is_joinable_with_candidate_signal(tmp_path):
    output = tmp_path / "entry_opportunities.jsonl"
    collector = CrudeEntryShadowCollector(output)
    signal_id = collector.record_signal(
        {
            "timestamp": datetime(2026, 10, 19, 10, 0, tzinfo=IST),
            "side": "CE",
            "futures_price": 7000,
        }
    )
    collector.record_decision(
        signal_id,
        decision="TAKEN",
        reason="strategy signal entered",
        option_symbol="CRUDEOIL7200CE",
        option_entry_price=100.0,
    )
    collector.record_trade_outcome(
        signal_id=signal_id,
        option_symbol="CRUDEOIL7200CE",
        option_entry_price=100.0,
        option_exit_price=112.0,
        pnl_amount=1200.0,
        quantity=100,
        exit_reason="TARGET HIT",
        option_price_path=[100.0, 104.0, 112.0],
    )

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    outcome = records[-1]
    assert outcome["event"] == "executed_trade_outcome"
    assert outcome["signal_id"] == signal_id
    assert outcome["pnl_points"] == 12.0
    assert outcome["pnl_amount"] == 1200.0
    assert outcome["quantity"] == 100
    assert outcome["option_price_path"] == [100.0, 104.0, 112.0]
