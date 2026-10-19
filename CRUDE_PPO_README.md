# Isolated crude-options PPO experiment

This experiment is separate from `trading_crude.py`: it does not import, edit,
or connect to the live order-execution bot. Model files and reports are written
under `models/` by default.

## Data contract and limits

Use one completed trade per row in a CSV or JSONL file. Rows need an option side
(`Option Side`, `Option Type`, or `side`, with CE/PE values) and realized PnL
(`PnL` or `pnl`). The state recognizes entry OHLC, ATR, volume, OI change, NYMEX
trend, and support/resistance levels or distances. Feature columns that are
absent are zero-filled with a warning; that can make a model less informative.

Only the recorded CE/PE side has an observed outcome. The existing
`CrudePPOEnv` does not invent the PnL for the opposite side. It remains a
trade-level entry/exit experiment and cannot test stop/target timing.
`crude_tick_env.py` is a separate risk-management environment: it can replay
rows with `Intratrade Option Prices`, adjusting ATR-based SL, TP, and SL-buffer
parameters on every recorded price sample. Missing price paths and missing or
invalid entry ATR are excluded with a warning. The saved price arrays are
ordered, de-duplicated option-price changes (up to 5,000 samples) without
per-sample timestamps; replay therefore tests observed price order, not
timestamped exchange ticks or exact intratick execution. Costs, spread,
slippage beyond the observed exit quote, and counterfactual changes to future
trade entries are not modeled. Explicit outcome labels map as
follows after 0.25 scaling: optimal profit `+2.5`, target `+2.25`, other
positive trade `+2`, panic exit `-0.5`, valid stop-loss `-2`, and invalid
stop-loss `-2.5`. When no recognized exit label exists, reward falls back to
realized PnL sign (`+2`, `-2`, or `0`). Skipping a trade or selecting an
unsupported counterfactual side receives `-0.05`; invalid re-entry receives
`-2.5`.
The original trade-level backtest PnL and win rate use realized PnL only for
trades where the agent selected the side actually recorded.

### Dynamic observed-path environment

The current `data/crude_closed_trades.csv` contains 647 trades. Of these,
411 have a non-empty `Intratrade Option Prices` path and positive `Entry ATR`;
all 411 paths begin at the recorded entry price and 410 end at the recorded
exit price. These rows are sufficient for ordered sample-path replay with the
new environment, but not exact timestamp-by-timestamp replay because
intermediate samples have no timestamps. The other 236 rows cannot be replayed
through dynamic stops and targets.

`CrudeTickEnv` treats each trade as a long option-premium position. Each
continuous action has three values in `[0, 1]`: desired SL ATR multiplier,
trailing-stop gap (0.5R to 1R), and SL buffer in ATR units. The initial 1R is
the entry-to-initial-stop distance. Reaching +1R activates the trailing stop;
it does not book a fixed take-profit. The stop follows the best observed price
by the policy-selected gap and can only tighten, allowing winners to run while
locking at least breakeven once 1R is reached. The simulator advances one
stored price sample per action and exits when the stop is hit or at the
recorded exit price. Price gaps between stored samples can still make
realized losses exceed the intended stop level.

Non-terminal samples receive zero reward; when the trade closes, the terminal
reward is realized PnL divided by initial risk. The episode reward is therefore
the simulated realized R multiple (before unmodeled costs and gap effects),
not a bonus for a temporary unrealized move. Previous tick-PPO models used a different
observation/action meaning and reward scale; retrain from scratch before using
the new trailing policy. The default model path is
`models/crude_tick_ppo_run_trailing_1r`, separate from the prior models.
Example:

```python
from stable_baselines3 import PPO

from crude_tick_env import CrudeTickEnv

env = CrudeTickEnv("data/crude_closed_trades.csv")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```

The simulation reports option-premium PnL points as well as R multiples; it
does not reuse the historical PnL as the simulated result. Run its focused tests with
`python -m pytest test_crude_tick_env.py`.

### Train on the first 80% and test on the newest 20%

To train the tick-risk policy on older valid price paths and evaluate it once on
the newest chronological 20%, run this from the `algo_project` directory:

```powershell
python train_and_evaluate_tick_ppo.py --data .\data\crude_closed_trades.csv --model-path .\models\crude_tick_ppo_run_trailing_1r --total-timesteps 100000
```

The command saves the model as
`models/crude_tick_ppo_run_trailing_1r.zip`, prints the holdout
trade count, win/loss count, simulated PnL in option premium points and R
multiples, and exit reasons, and writes per-trade results to
`models/crude_tick_ppo_run_trailing_1r_holdout_trades.csv` and the summary to
`models/crude_tick_ppo_run_trailing_1r_holdout_report.json`. The holdout comprises the newest
20% of valid trade paths sorted by entry timestamp. It is evaluated
deterministically once per trade path. Use `--mode evaluate` with the same data
and model path to evaluate the saved model again without retraining.

To automatically compare PnL/R reward settings, run the reward grid from the
`algo_project` directory:

```powershell
python grid_search_tick_rewards.py --sheet1 --model-path .\models\crude_tick_ppo_reward_grid_best --timesteps-per-trial 30000
```

The grid trains nine combinations. PnL points are normalized by median initial
risk in the training period, and the R reward is simulated PnL divided by each
trade's initial risk. It selects on a chronological validation period,
preferring positive option-point PnL, then higher total PnL, net R, R profit
factor, and win rate.
The newest 20% stays an untouched final holdout and is evaluated only after
selection. The chosen model, ranked trials, final report, and holdout trades
are saved under `models/`. All candidates replay the same already-executed
trades; this tunes exits, not whether to enter or skip a signal. The
read-only shadow observer loads `models/crude_tick_ppo_reward_grid_best.zip`
after that selected model has been trained.

Reported PnL is option premium points, not rupees; the report does not apply
lot quantity, brokerage, spread, or slippage. The holdout results are
historical simulation only and are not a guarantee of future performance or
authorization for live trading.

The repository's current paper-trade JSONL may not yet contain realized PnL and
entry OHLC/OI-change for completed trades. Export a complete closed-trade
dataset first; the loader will reject a dataset without realized PnL and CE/PE
labels rather than pretending open positions are outcomes.

### Collect entry/avoidance shadow data

The live Crude process writes eligible rule-generated entry candidates to
`data/crude_entry_opportunities.jsonl`, separate from the trade sheet and
without placing or changing orders. It records signal-time features, whether
the signal was taken or skipped (including outcome-filter vetoes and entry
attempts that did not open a position), and timestamped futures prices for
five minutes after the candidate. It adds side-adjusted futures returns at
1/3/5 minutes plus favorable/adverse excursions. Executed trades are linked
using `Entry Signal ID` in Sheet1 and a separate outcome event containing
actual option entry/exit, PnL points, PnL amount, and captured option-price path.

The skipped-signal outcome is explicitly a futures-direction proxy, not
counterfactual option PnL. This is useful to build an entry-filter dataset, but
it is not enough to claim a profitable learned entry policy. PPO tick rewards
still manage exits for already-open trades; entry avoidance needs a separate
trained model and a later unseen-period evaluation. Only candidates emitted by
the strategy setup are collected; rule failures before a signal is created are
not included.

## Install the isolated dependencies

From the `algo_project` directory:

```powershell
python -m pip install -r requirements-crude-ppo.txt
```

Railway builds from the repository root using the root `requirements.txt`,
which includes both this app's `requirements.txt` and the PPO requirements.
The root `Procfile` starts the combined `main.py` orchestrator from this
directory.

## Train and evaluate

The newest 20% of records, ordered by timestamp (or input order if no timestamp
exists), is held out from PPO training and evaluated once after training:

```powershell
python train_and_evaluate_ppo.py --data .\data\crude_closed_trades.csv
```

To fetch all populated rows from `Crude_Algo_Trade_Logs` Sheet 1 directly
(without first exporting a CSV), run:

```powershell
python train_and_evaluate_ppo.py --sheet1 --model-path .\models\crude_ppo_sheet1_150k --total-timesteps 150000
```

Tick-risk PPO can also read Sheet 1 directly. It filters to Crude rows and uses
only rows with a valid intratrade option-price path and Entry ATR; no CSV export
is needed:

```powershell
python train_and_evaluate_tick_ppo.py --sheet1 --model-path .\models\crude_tick_ppo_run_trailing_1r --total-timesteps 100000
```

The newest 20% of valid paths remains a holdout for evaluation, not training.
Rows without the logged `Intratrade Option Prices` path or a positive `Entry
ATR` are excluded. The sheet is read-only.

This uses the same `service_account.json` / `SERVICE_ACCOUNT_JSON` credentials
as the existing Google Sheets logger. The service-account email must have
permission to read the spreadsheet. The report identifies the source and
includes only rows whose `Symbol` contains `CRUDE`; valid rows must include
`Action`, `PnL`, and `Entry Timestamp`. The sheet is read-only: training does
not write model results or changes back to Google Sheets.
The example uses a separate model path so it does not overwrite an existing PPO
model.

### Search reward parameters

Run the Sheet1 reward grid search with:

```powershell
python grid_search_rewards.py
```

The script trains 27 PPO configurations for 30,000 steps each, using the same
chronological 80/20 split and holdout evaluation as the training pipeline. It
prints configurations ranked by holdout total PnL, along with win rate,
executed trades, and profit factor. The highest-PnL configuration is marked
with `*`. Intermediate normalization files are temporary; the script does not
save or overwrite trained models.

For JSONL:

```powershell
python train_and_evaluate_ppo.py --data .\logs\crude_closed_trades.jsonl
```

Train/evaluate output can be changed without touching live-bot files:

```powershell
python train_and_evaluate_ppo.py --data .\data\crude_closed_trades.csv --model-path .\models\crude_ppo_trial --total-timesteps 50000
```

Re-evaluate the saved model on the same newest 20% split:

```powershell
python train_and_evaluate_ppo.py --data .\data\crude_closed_trades.csv --model-path .\models\crude_ppo_trial --mode evaluate
```

The script saves a PPO ZIP, observation-normalization statistics, and a
`*_holdout_report.json` containing realized PnL, win rate, profit factor, and
drawdown. A positive historical holdout result is not a guarantee of future
profitability and should not be treated as authorization for live trading.

## Focused tests

```powershell
python -m pytest test_crude_ppo_env.py
```
