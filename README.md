# Ripple for RLBot v5

Ripple v1.0 and v1.1 share one runtime, with CPU inference through ONNX Runtime.
Each version keeps its configuration, models and cosmetics in its own folder.

## Run

Install [uv](https://docs.astral.sh/uv/) and Python 3.12 or newer, then add a
source config to the RLBot v5 GUI. Runtime inference uses the included ONNX
models; PyTorch and the original `.pt` exports are not required.

| Version | Source config | RL policy | Imitation model |
| --- | --- | --- | --- |
| v1.1 | [Config](bots/v1.1/ripple-v1.1.bot.toml) | 1M parameters | 1M parameters |
| v1.0 | [Config](bots/v1.0/ripple-v1.0.bot.toml) | 6M parameters | 6M parameters |

For a standalone executable, follow [the packaging guide](docs/DEVELOPMENT.md#packaging).
Bob produces GUI-ready configs. For a local Windows build, copy a source
config alongside the original, naming it `ripple-v1.1-executable.bot.toml`
(or v1.0), and replace its run command with:

```toml
run_command = '..\..\dist\bot.exe --variant v1.1'
```

Use `v1.0` for the other version. These local configs are ignored by Git.
Source and executable configs share an agent ID; use one launcher per version
in a match. Both default to bandit selection. The executable needs no uv or
Python installation.

The imitation-model pairings follow the old port's version configs. They were
not chosen through a comparison of playing strength.

To check the models and bundled assets without starting a match:

```console
uv run python src/bot.py --variant v1.1 --self-test
uv run python src/bot.py --variant v1.0 --self-test
```

## Imitation-only and selection modes

Add `--bcm-only` to either version's run command to use its imitation model
instead of its RL policy. This selects the **1M BCM for v1.1** and the **6M BCM
for v1.0**, without a separate BCM config.

```console
uv run python src/bot.py --variant v1.1 --bcm-only --self-test
```

Both RL versions default to `--mode bandit`. Like the old port, the bandit
adapts between `kl`, `inverse-kl`, `stochastic` and `deterministic` using goal
results and matchup history. KL modes evaluate both the RL and imitation
models; both networks warm up before play.

History is kept for the current session. To retain it across runs, append
`--bandit-result-file PATH` with a writable SQLite file, for example
`--bandit-result-file C:/Users/Rolv/Ripple/bandit.sqlite3`. Versions keep
separate statistics even when sharing a file. Existing old-port databases
are left intact; their results are not imported.

Use `--mode stochastic`, `deterministic`, `kl` or `inverse-kl` to fix the
selection mode. BCM-only defaults to stochastic and also supports deterministic
selection. Kickoffs sample stochastically unless
`--disable-stochastic-kickoffs` is set.

For a comparison against the old port, use the same model and fixed mode on
both sides: old `--action-mode STOCHASTIC` and new `--mode stochastic`, for
example. Alternate team colors across several games.

## Runtime status

Ripple logs startup details and a status summary **once per minute** in the
RLBot console. Summaries include the phase, score, controlled cars, current
selection mode, decision-time percentiles, decisions exceeding 33.3 ms and
missed decision slots. Bandit summaries also include episode results.
Missed slots can come from packet gaps or other delays as well as inference.
Individual bandit updates are debug-only (`RLBOT_LOG_LEVEL=debug`).

## Development and packaging

Shared code lives in `src/`; model conversion, tests, performance measurements
and packaging are described in [the development guide](docs/DEVELOPMENT.md).
Bob builds one executable for both version configs. The Windows executable has
been smoke-tested. A user-run 2v2 comparison against the old port ended 51–51
after 73 minutes, showing no apparent regression in that setup. Bob's
Docker build produces Windows and Linux binaries; the packaging guide
includes the final archive command.

## License

Copyright (c) 2026 Rolv. Ripple-owned code, models, documentation, artwork and
loadouts use [CC BY-NC-SA 4.0](LICENSE). Attribute Ripple to Rolv and identify
modifications when sharing adapted material. Third-party dependencies and
resources retain their own rights and licenses; see [NOTICE](NOTICE) for scope.
