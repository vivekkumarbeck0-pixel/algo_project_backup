import pandas as pd
import pytest

from crude_ppo_env import (
    ACTION_ENTER_CE,
    ACTION_ENTER_PE,
    ACTION_EXIT,
    ACTION_HOLD,
    CrudePPOEnv,
    ENTRY_BONUS,
    IDLE_REWARD,
    OBSERVATION_FEATURES,
    UNSUPPORTED_ACTION_REWARD,
    missing_feature_groups,
    prepare_trade_data,
)
from train_and_evaluate_ppo import (
    calculate_metrics,
    chronological_split,
    load_sheet1_trade_data,
    run_pipeline,
)


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Entry Timestamp": "2026-01-01T09:00:00",
                "Entry Open": 70.0,
                "Entry High": 72.0,
                "Entry Low": 69.0,
                "Entry Close": 71.0,
                "Entry ATR": 2.0,
                "Entry Volume": 100,
                "Entry OI_Change": 20,
                "NYMEX Trend": "bullish",
                "Support": 68.0,
                "Resistance": 75.0,
                "Option Side": "CE",
                "Exit Reason": "Optimal Profit",
                "PnL": 500.0,
            },
            {
                "Entry Timestamp": "2026-01-02T09:00:00",
                "Entry Open": 71.0,
                "Entry High": 72.0,
                "Entry Low": 68.0,
                "Entry Close": 69.0,
                "Entry ATR": 2.5,
                "Entry Volume": 120,
                "Entry OI_Change": -10,
                "NYMEX Trend": "bearish",
                "Support": 67.0,
                "Resistance": 73.0,
                "Option Side": "PE",
                "Exit Reason": "Panic Exit",
                "PnL": -50.0,
            },
            {
                "Entry Timestamp": "2026-01-03T09:00:00",
                "Entry Open": 69.0,
                "Entry High": 71.0,
                "Entry Low": 67.0,
                "Entry Close": 68.0,
                "Entry ATR": 1.5,
                "Entry Volume": 90,
                "Entry OI_Change": -5,
                "NYMEX Trend": "sideways",
                "Support": 66.0,
                "Resistance": 72.0,
                "Option Side": "CE",
                "Exit Reason": "Valid Stop Loss",
                "PnL": -100.0,
            },
            {
                "Entry Timestamp": "2026-01-04T09:00:00",
                "Entry Open": 68.0,
                "Entry High": 70.0,
                "Entry Low": 67.0,
                "Entry Close": 69.0,
                "Entry ATR": 1.0,
                "Entry Volume": 80,
                "Entry OI_Change": 8,
                "NYMEX Trend": "bullish",
                "Support": 66.0,
                "Resistance": 71.0,
                "Option Side": "PE",
                "Exit Reason": "Invalid Trade SL",
                "PnL": -200.0,
            },
            {
                "Entry Timestamp": "2026-01-05T09:00:00",
                "Entry Open": 69.0,
                "Entry High": 73.0,
                "Entry Low": 68.0,
                "Entry Close": 72.0,
                "Entry ATR": 2.0,
                "Entry Volume": 150,
                "Entry OI_Change": 25,
                "NYMEX Trend": "up",
                "Support": 67.0,
                "Resistance": 75.0,
                "Option Side": "CE",
                "Exit Reason": "Target Hit",
                "PnL": 250.0,
            },
        ]
    )


@pytest.mark.parametrize(
    ("outcome", "pnl", "expected"),
    [
        ("Optimal Profit", 1, 5.0),
        ("Target Hit", 1, 4.5),
        ("Profitable Exit", 1, 4.0),
        ("Panic Exit", -1, -1.0),
        ("Valid Stop Loss", -1, -4.0),
        ("Invalid Trade SL", -1, -5.0),
    ],
)
def test_reward_mapping(outcome, pnl, expected):
    frame = pd.DataFrame(
        [{"Option Side": "CE", "Outcome": outcome, "PnL": pnl}]
    )
    prepared = prepare_trade_data(frame)
    assert prepared.loc[0, "_ppo_reward"] == expected


def test_environment_enters_then_exits_with_logged_reward_and_pnl():
    env = CrudePPOEnv(_trades())
    observation, _ = env.reset(options={"row_index": 0})

    assert observation.shape == (len(OBSERVATION_FEATURES),)
    assert env.observation_space.contains(observation)
    _, reward, terminated, truncated, info = env.step(ACTION_ENTER_CE)
    assert reward == ENTRY_BONUS
    assert info["reward"] == ENTRY_BONUS
    assert info["position_opened"]
    assert not terminated
    assert not truncated

    _, reward, terminated, truncated, info = env.step(ACTION_EXIT)
    assert reward == 5.0
    assert terminated
    assert not truncated
    assert info["entered_trade"]
    assert info["pnl"] == 500.0


def test_atr_normalized_pivot_distances_are_not_scaled_twice():
    frame = _trades().iloc[[0]].copy()
    frame["Support Distance ATR"] = 1.25
    frame["Resistance Distance ATR"] = 2.5
    env = CrudePPOEnv(frame)
    observation, _ = env.reset()

    assert observation[8] == pytest.approx(1.25)
    assert observation[9] == pytest.approx(2.5)


def test_option_entry_price_is_not_mistaken_for_underlying_ohlc_close():
    frame = pd.DataFrame(
        [
            {
                "Entry Price": 4.2,
                "Entry ATR": 2.0,
                "Support": 68.0,
                "Resistance": 75.0,
                "Option Side": "CE",
                "PnL": 10.0,
            }
        ]
    )
    env = CrudePPOEnv(frame)
    observation, _ = env.reset()

    assert observation[3] == 0.0
    assert observation[8] == 0.0
    assert observation[9] == 0.0
    assert "entry close" in missing_feature_groups(frame)
    assert "support distance or support level" in missing_feature_groups(frame)


def test_opposite_side_is_flagged_without_inventing_counterfactual_loss():
    env = CrudePPOEnv(_trades())
    env.reset(options={"row_index": 0})
    _, reward, terminated, _, info = env.step(ACTION_ENTER_PE)

    assert reward == UNSUPPORTED_ACTION_REWARD
    assert terminated
    assert info["unsupported_counterfactual"]
    assert info["pnl"] == 0.0
    assert info["reward"] == reward


@pytest.mark.parametrize("action", [ACTION_HOLD, ACTION_EXIT])
def test_flat_hold_or_exit_skips_trade(action):
    env = CrudePPOEnv(_trades())
    env.reset(options={"row_index": 0})
    _, reward, terminated, _, info = env.step(action)

    assert reward == IDLE_REWARD
    assert terminated
    assert info["skipped_trade"]
    assert info["reward"] == reward


def test_entering_twice_is_invalid_trade_sl():
    env = CrudePPOEnv(_trades())
    env.reset(options={"row_index": 0})
    env.step(ACTION_ENTER_CE)
    _, reward, terminated, _, info = env.step(ACTION_ENTER_CE)

    assert reward == -2.5
    assert terminated
    assert info["invalid_reentry"]


def test_split_is_chronological_and_keeps_newest_rows_for_validation():
    training, validation = chronological_split(prepare_trade_data(_trades()))

    assert len(training) == 4
    assert len(validation) == 1
    assert training["_ppo_timestamp"].max() < validation["_ppo_timestamp"].min()


def test_action_column_is_accepted_as_option_side():
    frame = pd.DataFrame(
        [
            {
                "Action": "CE",
                "PnL": 100.0,
                "Entry Timestamp": "15-09-2026 09:50",
                "Entry Candle Open": 70.0,
                "Entry Candle High": 72.0,
                "Entry Candle Low": 69.0,
                "Entry Candle Close": 71.0,
                "Exit Scenario": "TARGET HIT",
            }
        ]
    )

    prepared = prepare_trade_data(frame)

    assert prepared.loc[0, "_ppo_side"] == "CE"
    assert prepared.loc[0, "_ppo_pnl"] == 100.0
    assert prepared.loc[0, "_ppo_timestamp"] == pd.Timestamp("2026-09-15 09:50")
    assert prepared.loc[0, "_ppo_outcome"] == "TARGET HIT"
    env = CrudePPOEnv(prepared)
    observation, _ = env.reset()
    assert observation[:4].tolist() == pytest.approx([70.0, 72.0, 69.0, 71.0])


def test_metrics_use_realized_pnl_and_handle_empty_trades():
    metrics = calculate_metrics([{"pnl": 10.0}, {"pnl": -5.0}])
    assert metrics["trades"] == 2
    assert metrics["win_rate"] == 0.5
    assert metrics["total_pnl"] == 5.0
    assert metrics["profit_factor"] == 2.0
    assert calculate_metrics([])["trades"] == 0


def test_sheet1_loader_reads_all_crude_rows_and_ignores_other_symbols(monkeypatch):
    headers = [
        "Entry Timestamp", "Symbol", "Action", "PnL",
        "Entry Candle Open", "Entry Candle High", "Entry Candle Low", "Entry Candle Close",
        "Entry ATR", "Entry Volume", "Entry OI_Change", "Entry NYMEX_Trend",
    ]
    crude_rows = [
        ["01-09-2026 10:00", "CRUDEOIL17SEP268250CE", "CE", "100", "100", "102", "99", "101", "1", "50", "10", "GREEN"],
        ["02-09-2026 10:00", "CRUDEOIL17SEP268250PE", "PE", "-50", "101", "102", "98", "99", "1", "40", "-5", "RED"],
    ]

    class Worksheet:
        def get_all_values(self):
            return [headers, *crude_rows, ["03-09-2026 10:00", "NIFTYCE", "CE", "200"]]

    class Spreadsheet:
        sheet1 = Worksheet()

    class Client:
        def open(self, _title):
            return Spreadsheet()

    monkeypatch.setattr("sheets_logger.get_gspread_client", lambda: Client())

    frame = load_sheet1_trade_data()

    assert len(frame) == 2
    assert frame["_ppo_side"].tolist() == ["CE", "PE"]
    assert frame["_ppo_pnl"].tolist() == [100, -50]
    assert frame.attrs["has_timestamp"] is True


def test_sheet1_loader_drops_invalid_timestamps_and_sorts_mixed_formats(monkeypatch):
    headers = [
        "Entry Timestamp", "Symbol", "Action", "PnL",
        "Entry Candle Open", "Entry Candle High", "Entry Candle Low", "Entry Candle Close",
        "Entry ATR", "Entry Volume", "Entry OI_Change", "Entry NYMEX_Trend",
    ]
    rows = [
        ["14-10-2026 09:30", "CRUDEOILCE", "CE", "100", "100", "102", "99", "101", "1", "50", "10", "GREEN"],
        ["2026-10-13T08:45:00", "CRUDEOILPE", "PE", "-50", "101", "102", "98", "99", "1", "40", "-5", "RED"],
        ["2026/10/12 07:05", "CRUDEOILCE", "CE", "75", "100", "102", "99", "101", "1", "50", "10", "GREEN"],
        ["", "CRUDEOILPE", "PE", "-25", "101", "102", "98", "99", "1", "40", "-5", "RED"],
        ["not a timestamp", "CRUDEOILCE", "CE", "25", "100", "102", "99", "101", "1", "50", "10", "GREEN"],
        ["09:30", "CRUDEOILPE", "PE", "-10", "101", "102", "98", "99", "1", "40", "-5", "RED"],
    ]

    class Worksheet:
        def get_all_values(self):
            return [headers, *rows]

    class Spreadsheet:
        sheet1 = Worksheet()

    class Client:
        def open(self, _title):
            return Spreadsheet()

    monkeypatch.setattr("sheets_logger.get_gspread_client", lambda: Client())

    frame = load_sheet1_trade_data()

    assert len(frame) == 3
    assert frame["_ppo_timestamp"].tolist() == [
        pd.Timestamp("2026-10-12 07:05"),
        pd.Timestamp("2026-10-13 08:45"),
        pd.Timestamp("2026-10-14 09:30"),
    ]


def test_training_pipeline_saves_model_and_evaluates_only_holdout(tmp_path):
    data_path = tmp_path / "closed_trades.csv"
    model_path = tmp_path / "models" / "ppo_trial"
    _trades().to_csv(data_path, index=False)

    report = run_pipeline(
        data_path=data_path,
        model_base=model_path,
        total_timesteps=128,
        seed=7,
        mode="train",
    )

    assert report["training_rows"] == 4
    assert report["holdout_rows"] == 1
    assert report["holdout"]["holdout_rows"] == 1
    assert (tmp_path / "models" / "ppo_trial.zip").is_file()
    assert (tmp_path / "models" / "ppo_trial_vecnormalize.pkl").is_file()
    assert (tmp_path / "models" / "ppo_trial_holdout_report.json").is_file()

    evaluation = run_pipeline(
        data_path=data_path,
        model_base=model_path,
        total_timesteps=128,
        seed=7,
        mode="evaluate",
    )
    assert evaluation["training_rows"] == 4
    assert evaluation["holdout"]["holdout_rows"] == 1
