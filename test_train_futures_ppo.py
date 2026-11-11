from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from crude_tick_env import CrudeTradingEnv
from train_futures_ppo import build_episodes, evaluate, load_csv_rows, run, split_days


def samples(days=20, samples_per_day=125):
    start = datetime(2026, 9, 1, 10, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    return [
        {
            "Timestamp": (start + timedelta(days=day, seconds=10 * tick)).isoformat(),
            "Price": str(8000 + tick * (0.2 if day % 2 == 0 else -0.2)),
            "ATR": str(10 + tick / 100),
            "Token": "token",
            "Session ID": f"session-{day}",
        }
        for day in range(days)
        for tick in range(samples_per_day)
    ]


def test_loader_deduplicates_and_splits_by_whole_dates():
    rows = samples()
    rows.insert(1, rows[0].copy())
    episodes, counts = build_episodes(rows)
    train, holdout, train_days, holdout_days = split_days(episodes, 0.2)
    assert counts["duplicates"] == 1
    assert len(episodes) == 20
    assert len(train) == 16 and len(holdout) == 4
    assert max(train_days) < min(holdout_days)
    assert train[0][1][0] == 10 and train[0][1][-1] == pytest.approx(11.24)


def test_segments_split_on_gaps_restart_token_and_missing_atr():
    rows = samples(days=1, samples_per_day=500)
    for i, row in enumerate(rows):
        if i >= 250:
            row["Session ID"] = "restart"
        if i == 125:
            row["ATR"] = ""
    episodes, counts = build_episodes(rows)
    assert [len(prices) for _, prices, _ in episodes] == [125, 124, 250]
    assert counts["atr_not_ready"] == 1


@pytest.mark.parametrize("reason", ["gap", "token"])
def test_segments_split_on_long_gap_and_token_change(reason):
    rows = samples(days=1, samples_per_day=250)
    if reason == "gap":
        for row in rows[125:]:
            row["Timestamp"] = (datetime.fromisoformat(row["Timestamp"]) + timedelta(seconds=40)).isoformat()
    else:
        for row in rows[125:]:
            row["Token"] = "new-contract"
    episodes, _ = build_episodes(rows)
    assert [len(prices) for _, prices, _ in episodes] == [125, 125]


def test_conflicting_samples_and_insufficient_days_are_rejected():
    rows = samples(days=1)
    conflict = rows[0].copy()
    conflict["Price"] = "9000"
    with pytest.raises(ValueError, match="Conflicting"):
        build_episodes(rows + [conflict])
    episodes, _ = build_episodes(rows)
    with pytest.raises(ValueError, match="20 distinct"):
        split_days(episodes, 0.2)


def test_per_sample_atr_drives_observation_and_entry():
    prices = np.linspace(100, 103, 8)
    atr = np.arange(1, 9, dtype=float)
    env = CrudeTradingEnv([(prices, atr)], warmup_ticks=2)
    obs, _ = env.reset(seed=0)
    assert obs[9] == pytest.approx(atr[0] / prices[0])
    env.step(np.array([0.5, 0, 0, 0], dtype=np.float32))
    obs, _, _, _, info = env.step(np.array([1, 0, 0, 0], dtype=np.float32))
    assert obs[9] == pytest.approx(atr[2] / prices[2])
    assert info["events"][0]["atr"] == pytest.approx(atr[2])


def test_evaluate_reports_sides_and_intents():
    episodes, _ = build_episodes(samples(days=2))
    env = CrudeTradingEnv([(p, a) for _, p, a in episodes], warmup_ticks=100)

    class Model:
        def predict(self, obs, deterministic):
            return np.array([1.0 if obs[0] >= 0 else 0.0, 0, 0, 0]), None

    result = evaluate(Model(), env)
    assert result["intents"]["LONG"] > 0 and result["intents"]["SHORT"] > 0
    assert result["sides"]["LONG"]["trades"] > 0
    assert result["sides"]["SHORT"]["trades"] > 0
    assert result["max_drawdown_points"] >= 0


def test_pipeline_saves_separate_candidate_and_refuses_overwrite(monkeypatch, tmp_path):
    import sys

    class Model:
        def __init__(self, policy, env, verbose, seed, n_steps, batch_size):
            self.env = env
            self.num_timesteps = 0

        def learn(self, total_timesteps):
            assert total_timesteps == 100
            self.num_timesteps = total_timesteps

        def predict(self, obs, deterministic):
            return np.array([1.0 if obs[0] >= 0 else 0.0, 0, 0, 0]), None

        def save(self, path):
            from pathlib import Path
            Path(path + ".zip").write_text("test model", encoding="utf-8")

    monkeypatch.setitem(sys.modules, "stable_baselines3", SimpleNamespace(PPO=Model))
    path = tmp_path / "candidate"
    report = run(samples(), path, 100, 42, 0.2)
    assert len(report["train_days"]) == 16 and len(report["holdout_days"]) == 4
    assert path.with_suffix(".zip").exists()
    assert (tmp_path / "candidate_holdout_report.json").exists()
    with pytest.raises(FileExistsError):
        run(samples(), path, 100, 42, 0.2)


def test_csv_requires_expected_headers(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_text("Timestamp,Price\n", encoding="utf-8")
    with pytest.raises(ValueError, match="headers"):
        load_csv_rows(path)


def test_real_ppo_training_round_trip(tmp_path):
    ppo = pytest.importorskip("stable_baselines3")
    path = tmp_path / "candidate"
    report = run(samples(), path, 32, 7, 0.2)
    assert report["actual_timesteps"] >= 32
    model = ppo.PPO.load(str(path.with_suffix(".zip")))
    episodes, _ = build_episodes(samples(days=1))
    env = CrudeTradingEnv([(episodes[0][1], episodes[0][2])], warmup_ticks=100)
    observation, _ = env.reset()
    action, _ = model.predict(observation, deterministic=True)
    assert env.action_space.contains(action)
