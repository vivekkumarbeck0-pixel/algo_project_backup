from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from crude_ppo_paper_runner import BarAtr, PaperRunner, RunnerLimits, ShadowLogWriter
from crude_tick_env import CrudeTradingEnv, TradingRewardConfig, shape_trade_reward
from crude_tick_policy import EngineConfig, OBSERVATION_SIZE, TradeEngine

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


def test_runner_logs_only_to_ppo_shadow_dir_and_respects_daily_limit(tmp_path):
    writer = ShadowLogWriter(tmp_path / "ppo_shadow_logs")
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
    files = list((tmp_path / "ppo_shadow_logs").iterdir())
    assert {f.name for f in files} == {"ppo_paper_events_2026-10-26.jsonl", "ppo_paper_summary.json"}
    assert writer.trades == 1 and writer.pnl_points < 0
    assert runner.engine.side == 0  # max_trades_per_day blocked re-entry


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
    assert engine.step(100.0, 1.0, LONG_ACT, momentum=(0.5, 0.1, 0.5)) == []
    entry = engine.step(100.0, 1.0, LONG_ACT, momentum=(0.5, 0.5, -5.0))
    assert entry[0]["side"] == "LONG"
    engine.reset()
    assert engine.step(100.0, 1.0, SHORT_ACT, momentum=(0.5, 0.5, 0.5)) == []
    assert engine.step(100.0, 1.0, SHORT_ACT, momentum=(-0.5, -0.5, -0.5))[0]["side"] == "SHORT"


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
