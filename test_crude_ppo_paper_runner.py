from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from crude_ppo_paper_runner import (
    AngelOptionQuoteSource,
    BarAtr,
    FuturesTickRecorder,
    PaperRunner,
    RunnerLimits,
    ShadowLogWriter,
)
from crude_tick_env import CrudeTradingEnv, TradingRewardConfig, shape_trade_reward
from crude_tick_policy import EngineConfig, OBSERVATION_SIZE, TickFeatureBuilder, TradeEngine, map_action

LONG_ACT = [1.0, 0.0, 1.0, 1.0]  # long, 1.0 ATR (minimum) stop, 4R target
SHORT_ACT = [0.0, 0.0, 1.0, 1.0]
HOLD_ACT = [0.5, 0.0, 1.0, 1.0]


def test_long_take_profit_with_slippage():
    engine = TradeEngine(EngineConfig(slippage_points=0.1))
    assert engine.step(100.0, 1.0, LONG_ACT)[0]["type"] == "entry"
    assert engine.entry == pytest.approx(100.1)
    assert engine.risk == pytest.approx(1.0)
    event = engine.step(104.2, 1.0, HOLD_ACT)[0]
    assert event["reason"] == "take_profit"
    assert event["pnl_points"] == pytest.approx(104.2 - 0.1 - 100.1)


def test_short_stop_and_stop_never_loosens():
    engine = TradeEngine()
    engine.step(100.0, 1.0, SHORT_ACT)
    sl = engine.sl
    engine.step(99.9, 1.0, [0.5, 1.0, 1.0, 1.0])  # asks for a 3 ATR stop
    assert engine.sl == pytest.approx(sl)
    event = engine.step(101.1, 1.0, HOLD_ACT)[0]
    assert event["reason"] == "stop_loss" and event["pnl_r"] < 0


def test_trailing_stop_locks_profit():
    engine = TradeEngine(EngineConfig(trail_activation_points=1.5, trail_breakeven_buffer_points=0.5))
    engine.step(100.0, 1.0, [1.0, 0.0, 1.0, 0.0])  # 1.0 risk, 0.5R trail
    engine.step(101.4, 1.0, [0.5, 0.0, 1.0, 0.0])
    assert not engine.trail_active and engine.sl == pytest.approx(99.0)
    engine.step(101.5, 1.0, [0.5, 0.0, 1.0, 0.0])
    assert engine.trail_active and engine.sl == pytest.approx(101.0)
    event = engine.step(101.0, 1.0, HOLD_ACT)[0]
    assert event["reason"] == "trailing_stop" and event["pnl_r"] > 0
    assert event["trail_active"]


def test_trail_stays_inactive_below_fixed_activation_points():
    engine = TradeEngine()  # default 12 point activation
    engine.step(100.0, 1.0, [1.0, 0.0, 1.0, 0.0])
    engine.step(100.0 + 1.5, 1.0, [0.5, 0.0, 1.0, 0.0])  # past +1R, below +12 points
    assert not engine.trail_active and engine.sl == pytest.approx(99.0)


@pytest.mark.parametrize(
    ("side", "action", "prices"),
    [
        ("LONG", LONG_ACT, [100.0, 115.0, 104.0, 140.0]),
        ("SHORT", SHORT_ACT, [100.0, 85.0, 96.0, 60.0]),
    ],
)
def test_paper_mode_without_trailing_keeps_initial_stop_and_can_reach_target(
    side, action, prices
):
    engine = TradeEngine(EngineConfig(trailing_enabled=False))
    entry = engine.step(prices[0], 10.0, action)[0]
    initial_stop = entry["stop_loss"]
    assert entry["side"] == side
    assert engine.step(prices[1], 10.0, HOLD_ACT) == []
    assert engine.step(prices[2], 10.0, HOLD_ACT) == []
    assert not engine.trail_active
    assert engine.sl == pytest.approx(initial_stop)
    target_exit = engine.step(prices[3], 10.0, HOLD_ACT)[0]
    assert target_exit["reason"] == "take_profit"


def test_trail_jumps_to_breakeven_on_activation_then_ratchets_long():
    engine = TradeEngine(EngineConfig(trail_activation_points=2.0, trail_breakeven_buffer_points=1.0))
    engine.step(100.0, 1.0, [1.0, 0.0, 1.0, 1.0])  # risk 1.0, 1R trail gap
    engine.step(102.0, 1.0, [0.5, 0.0, 1.0, 1.0])
    assert engine.trail_active and engine.sl == pytest.approx(101.0)  # entry + buffer


def test_trail_ratchets_only_upward_long():
    engine = TradeEngine(EngineConfig(trail_activation_points=2.0, trail_breakeven_buffer_points=1.0))
    engine.step(100.0, 1.0, [1.0, 0.0, 1.0, 0.0])  # 0.5R gap
    levels = []
    for price in (102.0, 103.0, 102.6):
        engine.step(price, 1.0, [0.5, 0.0, 1.0, 0.0])
        levels.append(engine.sl)
    assert levels == sorted(levels) and levels[1] == pytest.approx(102.5)
    assert levels[2] == pytest.approx(102.5)


def test_trail_ratchets_only_downward_short():
    engine = TradeEngine(EngineConfig(trail_activation_points=2.0, trail_breakeven_buffer_points=1.0))
    engine.step(100.0, 1.0, [0.0, 0.0, 1.0, 0.0])
    engine.step(98.0, 1.0, [0.5, 0.0, 1.0, 0.0])
    assert engine.trail_active and engine.sl == pytest.approx(98.5)
    first = engine.sl
    engine.step(98.3, 1.0, [0.5, 0.0, 1.0, 0.0])
    assert engine.sl == pytest.approx(first)
    event = engine.step(98.6, 1.0, HOLD_ACT)[0]
    assert event["reason"] == "trailing_stop"


def test_reward_bonus_and_early_stop_penalty():
    cfg = TradingRewardConfig()
    win = {"pnl_r": 2.5, "reason": "take_profit", "hold_ticks": 50}
    loss = {"pnl_r": -1.0, "reason": "stop_loss", "hold_ticks": 3}
    assert cfg.early_stop_penalty == pytest.approx(1.0)
    assert shape_trade_reward(win, cfg) > 2.5 + cfg.target_bonus - 1e-9
    assert shape_trade_reward(loss, cfg) == pytest.approx(-1.0 - cfg.early_stop_penalty)


def test_env_trades_and_closes_at_episode_end():
    prices = np.linspace(100, 110, 40)
    env = CrudeTradingEnv([(prices, 2.0)], warmup_ticks=3)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBSERVATION_SIZE,)
    exits, done = [], False
    while not done:
        obs, _, done, _, info = env.step(np.asarray(LONG_ACT, dtype=np.float32))
        exits += [e for e in info["events"] if e["type"] == "exit"]
    assert exits and exits[-1]["pnl_r"] > 0


def test_runner_disables_local_logs_by_default_and_respects_daily_limit(tmp_path):
    log_dir = tmp_path / "ppo_shadow_logs"
    writer = ShadowLogWriter(log_dir)
    runner = PaperRunner(
        lambda obs: np.asarray(LONG_ACT),
        writer,
        limits=RunnerLimits(max_trades_per_day=1),
        fixed_atr=2.0,
    )
    start = datetime(2026, 10, 26, 10, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    prices = [100.0, 101.0, 99.0, 102.0, 103.0, 104.0]
    for i, price in enumerate(prices):
        runner.on_tick(start + timedelta(seconds=i), price)
    assert not log_dir.exists()
    assert writer.trades == 1 and writer.pnl_points < 0
    assert runner.engine.side == 0  # max_trades_per_day blocked re-entry


def test_runner_can_opt_in_to_local_logs(tmp_path):
    log_dir = tmp_path / "ppo_shadow_logs"
    writer = ShadowLogWriter(log_dir, local_logging=True)
    runner = PaperRunner(
        lambda obs: np.asarray(LONG_ACT),
        writer,
        fixed_atr=2.0,
    )
    start = datetime(2026, 10, 26, 10, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    for i, price in enumerate([100.0, 101.0, 99.0, 102.0, 103.0, 104.0]):
        runner.on_tick(start + timedelta(seconds=i), price)
    files = list(log_dir.iterdir())
    assert {f.name for f in files} == {"ppo_paper_events_2026-10-26.jsonl", "ppo_paper_summary.json"}


def test_runner_records_ticks_before_atr_warmup_and_after(tmp_path):
    class Store:
        def __init__(self):
            self.rows = []

        def append(self, rows):
            self.rows.extend(rows)

    store = Store()
    atr = BarAtr(period=2)
    recorder = FuturesTickRecorder("123", store)
    runner = PaperRunner(
        lambda obs: np.asarray(HOLD_ACT),
        ShadowLogWriter(tmp_path),
        atr_source=atr,
        tick_recorder=recorder,
    )
    start = datetime(2026, 10, 26, 10, 0)
    runner.on_tick(start, 100.0)
    atr.seed([(101.0, 99.0, 100.0), (102.0, 100.0, 101.0)])
    runner.on_tick(start + timedelta(seconds=1), 101.0)
    runner.on_tick(start + timedelta(seconds=10), 101.0)
    recorder.close()
    assert len(store.rows) == 2
    assert store.rows[0][2] == "" and store.rows[1][2] == pytest.approx(2.0)


def test_recording_failure_does_not_block_paper_trading(tmp_path, caplog):
    caplog.set_level("INFO", logger="crude_ppo_paper")

    class Recorder:
        def __init__(self):
            self.fail = True

        def record(self, timestamp, price, atr):
            if self.fail:
                raise OSError("disk unavailable")

    recorder = Recorder()
    runner = PaperRunner(
        lambda obs: np.asarray(LONG_ACT), ShadowLogWriter(tmp_path),
        fixed_atr=2.0, tick_recorder=recorder,
    )
    start = datetime(2026, 10, 26, 10, 0)
    assert runner.on_tick(start, 100.0) == []
    assert runner.on_tick(start + timedelta(seconds=1), 101.0)[0]["side"] == "LONG"
    assert caplog.text.count("PPO futures tick recording failed") == 1
    recorder.fail = False
    runner.on_tick(start + timedelta(seconds=2), 101.0)
    assert "PPO futures tick recording recovered" in caplog.text


@pytest.mark.parametrize("enabled", [True, False])
def test_background_runner_wires_optional_futures_recording(monkeypatch, tmp_path, enabled):
    import threading
    import crude_ppo_paper_runner as module

    stop = threading.Event()
    recorders = []
    rows = []

    def recorder_factory(token):
        class Store:
            def append(self, batch):
                rows.extend(batch)
        recorder = FuturesTickRecorder(token, Store())
        recorders.append(recorder)
        return recorder

    def ticks(*args):
        yield datetime(2026, 10, 26, 10, 0), 8050.0
        stop.set()

    monkeypatch.setenv("ENABLE_PPO_FUTURES_RECORDING", "1" if enabled else "0")
    monkeypatch.setattr(module, "FuturesTickRecorder", recorder_factory)
    monkeypatch.setattr(module, "PpoSheetSink", lambda: None)
    monkeypatch.setattr(module, "resolve_front_month_token", lambda: "123")
    monkeypatch.setattr(module, "seed_atr_from_history", lambda *args: False)
    monkeypatch.setattr(module, "make_live_option_quote_source", lambda: object())
    monkeypatch.setattr(module, "load_policy", lambda path: lambda obs: np.asarray(HOLD_ACT))
    monkeypatch.setattr(module, "angel_ticks", ticks)

    module.run_live_forever(stop_event=stop, retry_seconds=0)

    assert bool(recorders) is enabled
    assert bool(rows) is enabled


def test_option_quote_source_resolves_atm_leg_and_requires_fresh_quote():
    class Reader:
        def find_option_token(self, underlying, strike, right, **kwargs):
            assert (underlying, strike, right) == ("CRUDEOIL", 8150.0, "PE")
            assert kwargs == {"instrument_type": "OPTFUT", "exchange": "MCX"}
            return {"token": "pe-token", "symbol": "CRUDEOIL15NOV268150PE", "expiry": "15NOV2026"}

    class Store:
        def __init__(self):
            self.subscriptions = []
            self.quotes = {}

        def ensure_subscribed(self, token, exchange):
            self.subscriptions.append((token, exchange))
            return True

        def get(self, token, max_age):
            return self.quotes.get(token)

    store = Store()
    store.quotes["pe-token"] = {"ltp": 42.5}
    source = AngelOptionQuoteSource(store, Reader())

    contract, price, reason = source.prepare(8126.0, "PE")

    assert contract["strike"] == 8150.0
    assert contract["right"] == "PE" and contract["lot_size"] == 100
    assert price == pytest.approx(42.5) and reason is None
    assert store.subscriptions == [("pe-token", "MCX")]
    store.quotes.clear()
    assert source.quote(contract) is None


def test_ppo_sheet_adds_option_fields_to_existing_worksheet(monkeypatch):
    import sheets_logger

    class Worksheet:
        def __init__(self):
            self.headers = sheets_logger.PPO_SHEET_COLUMNS[:15]
            self.col_count = 15
            self.added_columns = 0
            self.rows = []

        def row_values(self, row):
            return self.headers

        def add_cols(self, count):
            self.added_columns += count
            self.col_count += count

        def update(self, cell, rows):
            assert cell == "A1"
            self.headers = rows[0]

        def append_row(self, row):
            self.rows.append(row)

    worksheet = Worksheet()

    class Book:
        def worksheet(self, title):
            return worksheet

    class Client:
        def open(self, name):
            return Book()

    monkeypatch.setattr(sheets_logger, "get_gspread_client", Client)

    assert sheets_logger.log_ppo_trade({"Option Side": "CE", "Strike": 8100})
    assert worksheet.added_columns == len(sheets_logger.PPO_SHEET_COLUMNS) - 15
    assert "Option Side" in worksheet.headers and "Underlying PnL R" in worksheet.headers
    assert worksheet.rows[0][worksheet.headers.index("Option Side")] == "CE"


def test_paper_runner_records_actual_option_premium_pnl(tmp_path):
    class OptionQuoteSource:
        def __init__(self):
            self.price = 50.0
            self.released = []

        def prepare(self, underlying_price, right):
            return (
                {
                    "token": "ce-token",
                    "symbol": "CRUDEOIL15NOV268100CE",
                    "right": right,
                    "strike": 8100.0,
                    "expiry": "15NOV2026",
                    "lot_size": 100,
                },
                self.price,
                None,
            )

        def quote(self, contract):
            return self.price

        def release(self, contract):
            self.released.append(contract["token"])

    source = OptionQuoteSource()
    writer = ShadowLogWriter(tmp_path)
    actions = iter([HOLD_ACT] * 20 + [[1.0, 0.0, 0.0, 1.0]] * 3)
    runner = PaperRunner(
        lambda obs: np.asarray(next(actions)),
        writer,
        fixed_atr=2.0,
        option_quote_source=source,
    )
    start = datetime(2026, 10, 26, 10, 0)
    entry_events = []
    prices = [100.0] * 15 + [100.5] * 5 + [101.0]
    for index, price in enumerate(prices):
        events = runner.on_tick(start + timedelta(seconds=index), price)
        if events:
            entry_events = events
    assert entry_events[0]["type"] == "entry"
    assert entry_events[0]["side"] == "BUY_CE"
    assert entry_events[0]["entry_price"] == pytest.approx(50.0)
    assert entry_events[0]["underlying_entry_price"] == pytest.approx(101.0)

    source.price = 55.0
    runner.on_tick(start + timedelta(seconds=21), 103.0)
    exit_events = runner.on_tick(start + timedelta(seconds=22), 104.0)

    assert exit_events[0]["type"] == "exit"
    assert exit_events[0]["side"] == "BUY_CE"
    assert exit_events[0]["option_exit_price"] == pytest.approx(55.0)
    assert exit_events[0]["pnl_points"] == pytest.approx(5.0)
    assert exit_events[0]["gross_pnl_inr"] == pytest.approx(500.0)
    assert exit_events[0]["net_pnl_inr"] == pytest.approx(450.0)
    assert exit_events[0]["underlying_pnl_r"] == pytest.approx(1.5)
    assert source.released == ["ce-token"]
    row = ShadowLogWriter._sheet_row(exit_events[0])
    assert row["Symbol"] == "CRUDEOIL15NOV268100CE"
    assert row["Action"] == "BUY_CE"
    assert row["Exit Reason"] == "underlying_take_profit"
    assert row["Underlying Exit Price"] == pytest.approx(104.0)


def test_stale_option_quote_at_session_end_is_not_counted_as_pnl(tmp_path):
    class OptionQuoteSource:
        price = 50.0

        def prepare(self, underlying_price, right):
            return (
                {
                    "token": "ce-token",
                    "symbol": "CRUDEOIL15NOV268100CE",
                    "right": right,
                    "strike": 8100.0,
                    "expiry": "15NOV2026",
                    "lot_size": 100,
                },
                self.price,
                None,
            )

        def quote(self, contract):
            return self.price

        def release(self, contract):
            pass

    source = OptionQuoteSource()
    actions = iter([HOLD_ACT] * 20 + [[1.0, 0.0, 0.0, 1.0]])
    writer = ShadowLogWriter(tmp_path, local_logging=True)
    runner = PaperRunner(
        lambda obs: np.asarray(next(actions)),
        writer,
        fixed_atr=2.0,
        option_quote_source=source,
    )
    start = datetime(2026, 10, 26, 10, 0)
    for index, price in enumerate([100.0] * 15 + [100.5] * 5 + [101.0]):
        events = runner.on_tick(start + timedelta(seconds=index), price)
    assert events[0]["type"] == "entry"

    source.price = None
    runner.on_tick(datetime(2026, 10, 26, 23, 25), 102.0)

    assert runner.engine.side == 0
    assert writer.trades == 0
    import json

    event_path = tmp_path / "ppo_paper_events_2026-10-26.jsonl"
    records = [json.loads(line) for line in event_path.read_text().splitlines()]
    unpriced = [record for record in records if record["type"] == "unpriced_exit"]
    assert len(unpriced) == 1
    assert unpriced[0]["option_pnl_unavailable"] is True
    assert unpriced[0]["option_exit_price"] is None


def test_runner_processes_ticks_without_per_tick_output(tmp_path, capsys):
    runner = PaperRunner(lambda obs: np.asarray(LONG_ACT), ShadowLogWriter(tmp_path), fixed_atr=2.0)
    start = datetime(2026, 10, 26, 10, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    assert runner.on_tick(start, 100.0) == []
    events = runner.on_tick(start + timedelta(seconds=1), 101.0)
    assert events[0]["type"] == "entry"
    assert capsys.readouterr().out == ""


def test_runner_processes_short_and_flat_intents_without_output(tmp_path, capsys):
    actions = iter([SHORT_ACT, SHORT_ACT, HOLD_ACT])
    runner = PaperRunner(lambda obs: np.asarray(next(actions)), ShadowLogWriter(tmp_path), fixed_atr=2.0)
    start = datetime(2026, 10, 26, 10, 0)
    assert runner.on_tick(start, 100.0) == []
    events = runner.on_tick(start + timedelta(seconds=1), 99.0)
    assert events[0]["side"] == "SHORT"
    assert runner.on_tick(start + timedelta(seconds=2), 99.0) == []
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("side", "action", "direction"),
    [("LONG", LONG_ACT, 1), ("SHORT", SHORT_ACT, -1)],
)
def test_runner_enters_both_sides_at_relaxed_momentum_thresholds(
    tmp_path, capsys, side, action, direction
):
    actions = iter([HOLD_ACT] * 20 + [action])
    runner = PaperRunner(
        lambda obs: np.asarray(next(actions)), ShadowLogWriter(tmp_path), fixed_atr=20.0
    )
    start = datetime(2026, 10, 26, 10, 0)
    prices = [100.0] * 15 + [100.0 + direction] * 5 + [100.0 + 2 * direction]
    for i, price in enumerate(prices[:-1]):
        assert runner.on_tick(start + timedelta(seconds=i), price) == []

    events = runner.on_tick(start + timedelta(seconds=20), prices[-1])

    assert len(events) == 1
    assert events[0]["type"] == "entry" and events[0]["side"] == side
    assert capsys.readouterr().out == ""


def test_runner_silently_respects_daily_trade_limit(tmp_path, capsys):
    runner = PaperRunner(
        lambda obs: np.asarray(LONG_ACT),
        ShadowLogWriter(tmp_path),
        limits=RunnerLimits(max_trades_per_day=0),
        fixed_atr=2.0,
    )
    start = datetime(2026, 10, 26, 10, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    assert runner.on_tick(start, 100.0) == []
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("timestamp", "price", "fixed_atr"),
    [
        (datetime(2026, 10, 26, 10, 0), 0.0, 2.0),
        (datetime(2026, 10, 24, 10, 0), 100.0, 2.0),
        (datetime(2026, 10, 26, 10, 0), 100.0, None),
        (datetime(2026, 10, 26, 23, 26), 100.0, 2.0),
    ],
)
def test_runner_silently_skips_policy_when_not_evaluated(
    tmp_path, capsys, timestamp, price, fixed_atr
):
    def policy(obs):
        pytest.fail("policy must not run")

    runner = PaperRunner(policy, ShadowLogWriter(tmp_path), fixed_atr=fixed_atr, atr_source=BarAtr())
    assert runner.on_tick(timestamp, price) == []
    assert capsys.readouterr().out == ""


def test_runner_flattens_before_close_and_ignores_closed_market(tmp_path):
    runner = PaperRunner(lambda obs: np.asarray(LONG_ACT), ShadowLogWriter(tmp_path), fixed_atr=2.0)
    ist = timezone(timedelta(hours=5, minutes=30))
    runner.on_tick(datetime(2026, 10, 26, 22, 59, tzinfo=ist), 100.0)
    runner.on_tick(datetime(2026, 10, 26, 23, 0, tzinfo=ist), 101.0)
    assert runner.engine.side == 1
    events = runner.on_tick(datetime(2026, 10, 26, 23, 26, tzinfo=ist), 100.2)
    assert events[0]["reason"] == "session_end"
    assert runner.on_tick(datetime(2026, 10, 24, 12, 0, tzinfo=ist), 100.0) == []  # Saturday


def test_bar_atr_needs_full_period():
    atr = BarAtr(bar_seconds=60, period=2)
    base = datetime(2026, 10, 26, 10, 0, tzinfo=timezone.utc)
    for minute in range(4):
        atr.update(base + timedelta(minutes=minute), 100.0 + minute)
        atr.update(base + timedelta(minutes=minute, seconds=30), 102.0 + minute)
    assert atr.value is not None and atr.value > 0


def test_bar_atr_uses_completed_bar_close_when_calculating_gap():
    atr = BarAtr(bar_seconds=60, period=2)
    base = datetime(2026, 10, 26, 10, 0, tzinfo=timezone.utc)
    atr.update(base, 100.0)
    atr.update(base + timedelta(seconds=30), 102.0)
    atr.update(base + timedelta(minutes=1), 110.0)
    atr.update(base + timedelta(minutes=1, seconds=30), 111.0)
    atr.update(base + timedelta(minutes=2), 112.0)

    assert atr.value == pytest.approx((2.0 + 9.0) / 2.0)


def test_bar_atr_seeds_from_historical_one_minute_candles():
    from types import SimpleNamespace
    from crude_ppo_paper_runner import historical_atr_bars

    now = datetime(2026, 10, 26, 12, 7, tzinfo=timezone.utc).astimezone(ZoneInfo("Asia/Kolkata"))
    start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)
    rows = [[(start + timedelta(minutes=m)).isoformat(), 100, 101 + (m % 5), 99, 100.5, 1] for m in range(120)]
    client = SimpleNamespace(get_candle_data=lambda *a, **k: {"data": rows})

    bars = historical_atr_bars(client, "T", "MCX", 300, 15, now=now)
    atr = BarAtr(300)
    atr.seed(bars)

    assert len(bars) == 15 and atr.value is not None and atr.value > 0


def test_bar_atr_seed_replaces_previous_history():
    atr = BarAtr(bar_seconds=60, period=2)
    atr.seed([(110.0, 100.0, 105.0), (115.0, 105.0, 110.0)])
    seeded_value = atr.value

    atr.seed([(101.0, 99.0, 100.0), (102.0, 100.0, 101.0)])

    assert seeded_value == pytest.approx(10.0)
    assert atr.value == pytest.approx(2.0)


def test_sheet1_rows_become_episodes(monkeypatch):
    import json, sys, types
    from crude_tick_env import CrudeTradingEnv, load_sheet1_trades

    header = ["Entry Timestamp", "Exit Timestamp", "Symbol", "Action", "Entry Price",
              "Exit Price", "Entry ATR", "PnL", "Intratrade Option Prices"]
    path = json.dumps([100, 101, 102, 101.5, 103, 104, 103.5, 105])
    rows = [header,
            ["2026-10-02 10:00:00", "2026-10-02 10:05:00", "CRUDEOIL15OCT268900CE", "CE", "100", "105", "2", "500", path],
            ["2026-10-01 10:00:00", "2026-10-01 10:05:00", "CRUDEOIL15OCT268900PE", "PE", "100", "105", "2", "500", path],
            ["", "", "NIFTY", "", "", "", "", "", ""]]
    sheet = types.SimpleNamespace(get_all_values=lambda: rows)
    client = types.SimpleNamespace(open=lambda name: types.SimpleNamespace(sheet1=sheet))
    module = types.ModuleType("sheets_logger")
    module.get_gspread_client = lambda: client
    monkeypatch.setitem(sys.modules, "sheets_logger", module)

    frame = load_sheet1_trades()
    assert len(frame) == 2
    env = CrudeTradingEnv.from_tick_trades(frame)
    assert len(env.episodes) == 2 and env.episodes[0][1] == 2.0


def test_min_target_rr_and_breakeven_jump_on_trail_activation():
    from crude_tick_policy import map_action
    assert map_action([1.0, 0.5, 0.0, 0.5])[2] == pytest.approx(1.5)
    engine = TradeEngine(EngineConfig(trail_activation_points=1.0, trail_breakeven_buffer_points=0.25))
    engine.step(100.0, 1.0, [1.0, 0.0, 1.0, 1.0])  # risk 1.0, loosest 1R trail gap
    engine.step(101.0, 1.0, [0.5, 0.0, 1.0, 1.0])  # +1 point arms the trail
    assert engine.trail_active and engine.sl == pytest.approx(100.25)


def test_min_stop_is_one_atr_and_ranges_are_configurable():
    from crude_tick_policy import map_action
    assert map_action([0.5, 0.0, 0.0, 0.0])[1] == pytest.approx(1.0)
    assert map_action([0.5, 0.0, 0.0, 0.0], sl_atr_bounds=(1.5, 2.0))[1] == pytest.approx(1.5)
    engine = TradeEngine(EngineConfig(sl_atr_bounds=(2.0, 3.0), min_entry_momentum_atr=()))
    engine.step(100.0, 1.0, LONG_ACT, momentum=(-5.0, -5.0, -5.0))
    assert engine.risk == pytest.approx(2.0)
    with pytest.raises(ValueError):
        EngineConfig(sl_atr_bounds=(3.0, 1.0))


def test_reward_bonuses_keep_net_r_primary():
    cfg = TradingRewardConfig()
    three = shape_trade_reward({"pnl_r": 3.0, "reason": "take_profit", "hold_ticks": 50}, cfg)
    assert three - 3.0 <= 1.5


def test_three_r_bonus_and_dead_chase_penalty():
    cfg = TradingRewardConfig()
    two = shape_trade_reward({"pnl_r": 2.0, "reason": "take_profit", "hold_ticks": 50}, cfg)
    three = shape_trade_reward({"pnl_r": 3.0, "reason": "take_profit", "hold_ticks": 50}, cfg)
    assert three - two > 1.0 + cfg.target_3r_bonus - 1e-9
    prices = np.concatenate([np.linspace(100.0, 101.0, 10), np.full(50, 101.0)])
    env = CrudeTradingEnv([(prices, 2.0)], warmup_ticks=3)
    env.reset(seed=0)
    wide = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    rewards = [env.step(wide)[1] for _ in range(45)]
    assert env._engine.side == 1
    assert rewards[-1] == pytest.approx(-cfg.dead_chase_penalty)


def test_entry_requires_short_and_medium_term_momentum_only():
    engine = TradeEngine()
    assert engine.step(100.0, 1.0, LONG_ACT, momentum=(0.5, 0.09, 0.5)) == []
    entry = engine.step(100.0, 1.0, LONG_ACT, momentum=(0.5, 0.5, -5.0))
    assert entry[0]["side"] == "LONG"
    engine.reset()
    assert engine.step(100.0, 1.0, SHORT_ACT, momentum=(0.5, 0.5, 0.5)) == []
    assert engine.step(100.0, 1.0, SHORT_ACT, momentum=(-0.5, -0.5, -0.5))[0]["side"] == "SHORT"


@pytest.mark.parametrize(
    ("action", "sign", "side"),
    [(LONG_ACT, 1, "LONG"), (SHORT_ACT, -1, "SHORT")],
)
def test_momentum_gate_requires_both_signed_thresholds(action, sign, side):
    engine = TradeEngine()
    assert engine.step(100.0, 1.0, action, momentum=(sign * 0.049, sign * 0.1, 0)) == []
    assert engine.step(100.0, 1.0, action, momentum=(sign * 0.05, sign * 0.099, 0)) == []
    assert engine.step(100.0, 1.0, action, momentum=(-sign * 0.2, -sign * 0.2, 0)) == []
    assert engine.step(100.0, 1.0, action, momentum=(sign * 0.05, sign * 0.1, 0))[0]["side"] == side


@pytest.mark.parametrize("direction", [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0])
def test_action_direction_is_symmetric_around_hold(direction):
    action = [direction, 0.5, 0.5, 0.5]
    mirror = [1.0 - direction, 0.5, 0.5, 0.5]
    mapped = map_action(action)
    mirrored = map_action(mirror)
    assert mapped[0] == -mirrored[0]
    assert mapped[1:] == pytest.approx(mirrored[1:])


def test_tick_observation_directional_features_are_mirrored():
    observations = []
    for sign in (1, -1):
        features = TickFeatureBuilder()
        for offset in (0, 1, 3, 2, 4, 5):
            features.update(100.0 + sign * offset)
        observations.append(features.observation(10.0, TradeEngine()))

    up, down = observations
    assert up[:3] == pytest.approx(-down[:3])
    assert up[3:5] == pytest.approx(down[3:5])
    assert up[5] == pytest.approx(-down[6])
    assert up[6] == pytest.approx(-down[5])
    assert up[7:9] == pytest.approx(-down[7:9])
    assert up[9] == pytest.approx(10.0 / 105.0)
    assert down[9] == pytest.approx(10.0 / 95.0)
    assert up[10:] == pytest.approx(down[10:])


def test_mirrored_trading_episodes_have_equal_rewards():
    rewards_and_exits = []
    for sign, action in [(1, LONG_ACT), (-1, SHORT_ACT)]:
        prices = 100.0 + sign * np.arange(12, dtype=float)
        env = CrudeTradingEnv(
            [(prices, 5.0)],
            engine_config=EngineConfig(slippage_points=0.1, cost_points=0.2),
            warmup_ticks=3,
        )
        env.reset(seed=0)
        rewards, exits = [], []
        done = False
        while not done:
            _, reward, done, _, info = env.step(np.asarray(action, dtype=np.float32))
            rewards.append(reward)
            exits.extend(event for event in info["events"] if event["type"] == "exit")
        rewards_and_exits.append((rewards, exits))

    long_result, short_result = rewards_and_exits
    assert long_result[1] and short_result[1]
    assert long_result[1][0]["side"] == "LONG"
    assert short_result[1][0]["side"] == "SHORT"
    assert long_result[0] == pytest.approx(short_result[0])
    assert long_result[1][0]["pnl_r"] == pytest.approx(short_result[1][0]["pnl_r"])


def test_exit_rows_go_to_ppo_sheet_with_fixed_charges(tmp_path):
    from crude_ppo_paper_runner import PpoSheetSink

    rows = []
    sink = PpoSheetSink(rows.append)
    runner = PaperRunner(
        lambda obs: np.array(LONG_ACT), ShadowLogWriter(tmp_path, sheet_sink=sink), fixed_atr=1.0
    )
    start = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
    runner._emit(start, runner.engine.step(100.0, 1.0, LONG_ACT))
    runner._emit(start, runner.engine.force_close(101.0, "manual"))
    sink.close()
    assert len(rows) == 1
    row = rows[0]
    assert row["Gross PnL"] == pytest.approx(100.0)
    assert row["Charges"] == 50.0 and row["Net PnL"] == pytest.approx(50.0)
    assert row["Exit Reason"] == "manual"
