# Development

## Layout

```text
bots/
  v1.0/   config, loadout, logo, RL policy and 6M BCM
  v1.1/   config, loadout, logo, RL policy and 1M BCM
src/      shared packet adapter, observations, actions and inference
tools/    ONNX conversion, validation and benchmarks
tests/    model parity and RLBot packet checks
```

Git contains the float32 `.onnx` models used for inference. Original `.pt`
exports are optional local development inputs, ignored by Git and excluded
from executable builds. `src/variants.py` defines the model pairings and agent IDs. BCM-only
inference uses the selected version's BCM as the policy and loads no RL model.
It preserves the old port's initial throttle/boost queue to encourage movement.

The original model names are:

| Version | RL policy | BCM |
| --- | --- | --- |
| v1.1 | `ripple-v1.1-81300-fast.pt` | `bcm-1M-low_decay3-1654.pt` |
| v1.0 | `ripple-v1.0-36B-fast.pt` | `bcm-6M-normalized_dot-331.pt` |

These pairings were copied from the old port, without a strength comparison.
The former standalone BCM config used the 6M model; its equivalent is now
`--variant v1.0 --bcm-only`.

For conversion, parity checks or the Torch backend, obtain the original
exports from Ripple-RLGymPack and copy them as follows. Keep these originals
in your own archive; they are not duplicated in this repository.

| Original file | Local destination |
| --- | --- |
| `v1.1/ripple-v1.1-81300-fast.pt` | `bots/v1.1/ripple-v1.1.pt` |
| `v1.1/bcm-1M-low_decay3-1654.pt` | `bots/v1.1/ripple-v1.1-bcm.pt` |
| `v1.0/ripple-v1.0-36B-fast.pt` | `bots/v1.0/ripple-v1.0.pt` |
| `v1.0/bcm-6M-normalized_dot-331.pt` | `bots/v1.0/ripple-v1.0-bcm.pt` |

The ONNX metadata records each original's SHA-256 for verification.

## Checks and conversion

```console
uv run python -m unittest discover -s tests -v
```

The default checks require neither Torch nor `.pt` files. Original-model
comparisons are explicitly skipped until both are available. After restoring
the files above, run the complete development checks:

```console
uv run --group export python -m unittest discover -s tests -v
uv run --group export python tools/export_models.py
uv run --group export python tools/validate_models.py --output parity-results.json
uv run --group export python tools/benchmark.py --iterations 300 --output benchmark-results.json
```

Add `--bcm-only` to the benchmark command to measure imitation-only inference.
The optional PyTorch backend is available for source comparisons:

```console
uv run --extra torch python src/bot.py --variant v1.0 --backend torch --self-test
```

`--seed` makes the NumPy action and bandit samplers reproducible between backends;
it does not reproduce the old port's PyTorch random stream. Dependency versions
are pinned in `pyproject.toml` and resolved in `uv.lock`. The default installation
needs no PyTorch or Git dependency.

Conversion loads all learned weights strictly into the training architecture,
uses explicit attention reshapes and precomputes action embeddings. No weights
are quantized. Each ONNX file records its source SHA-256. Validation compares
logits, probabilities and greedy decisions against the original TorchScript
exports. Tests also compare seeded stochastic and KL decisions between backends.

Floating-point differences are small but not zero. The 1M BCM's unusually large
logits amplify rounding after softmax: its absolute probability tolerance is
0.005, versus 0.0001 for the other models. All tested greedy choices must agree.
`tools/validate_models.py` produces the optional local `parity-results.json`;
generated reports are ignored by Git.

## Training fidelity and performance

The adapter follows `configuration/agent2.py`, `observation_builder2.py` and
`action_parser2.py` in the training source. The unnumbered modules are obsolete.
Training-observation tests use the optional `external/Ripple` reference; runtime
and packaging do not need `external/`.

The port preserves normalization, mirroring, five padded player slots, the
276-action table, timer logic and training's boolean handbrake-history behavior.
The network already unmirrors action logits. Decisions occur every four physics
ticks, with a six-decision (200 ms) action queue. Ball observations include the
current state and predictions at one-second intervals through five seconds.

RL kickoff queues start neutral; mid-play resets fill an unknown queue with the
first selection. Countdown does not advance it. After missed packets, the
scheduler preserves frame alignment and repeats the latest plan in missed slots.
Transitions hidden in missing packets cannot be reconstructed exactly.

The old port discarded optimized models, built unused opponent observations
and lost scheduling remainders after late packets. These issues are corrected.
Ball predictions use training's reuse tolerances rather than the old port's
stricter 0.01 tolerance.

The bandit preserves the old port's candidate order, hierarchical beta priors
(other sizes, same sizes, exact matchup), Thompson sampling and perfect-success
shortcut. It selects at episode boundaries rather than on each decision.
Score changes are credited once, including scores delayed until replay or
countdown; pauses and roster/frame resets do not count as losses. The bandit
uses a separate seeded RNG and omits the old 10,000 diagnostic samples.

SQLite stores aggregate per-matchup counters instead of an unbounded episode
log. `--bandit-result-file PATH` enables persistence; otherwise history lasts
for the process. Profiles isolate versions and custom model paths. The new
`ripple_mode_counts` table does not modify or import the old port's results.
Counter queries and writes occur at episode boundaries, outside normal
decisions. Fixed stochastic/deterministic selection creates no bandit database
and loads no BCM. Adaptive selection warms both networks before play.

Local median forward-pass times on a Ryzen 9 5950X, one thread, 300 samples:

| Version | ONNX, one car | TorchScript, one car | ONNX full decision, three cars |
| --- | --- | --- | --- |
| v1.1 | 0.34 ms | 1.20 ms | 1.47 ms |
| v1.0 | 1.55 ms | 3.48 ms | 3.47 ms |

Full decisions use stochastic selection and cached ball predictions. KL modes
cost another network evaluation. `benchmark-results.json` includes percentiles
and prediction-rebuild cases. Synthetic benchmarks exclude Rocket League,
server and socket costs. Runtime logs report decision percentiles and missed
slots once per wall-clock minute, including during pauses. Summaries also
report phase, score, current mode, slow decisions and bandit results. Startup
logs once; per-episode bandit diagnostics use DEBUG. A user-run 2v2
head-to-head against the old port ended 51–51 after 73 minutes. Other formats
and the botpack strength ceiling still need live validation.

## Packaging

```console
uv run --group build python -m PyInstaller --noconfirm bot.spec
bob build bob.toml
uv run python tools/package_release.py bob_build/Ripple
```

Bob requires Docker. One executable bundles ONNX Runtime, all four models and
the RocketSim meshes; it selects the version through `RLBOT_AGENT_ID`. Original
`.pt` files and PyTorch are excluded. Both version folders use distinct config,
logo and loadout filenames because Bob flattens those files into one output
folder. Shared source code and inference are compiled only once.

Models use the same version prefix as the other assets:
`ripple-v1.1.onnx` and `ripple-v1.1-bcm.onnx`, with matching names for v1.0.

The last command creates `dist/Ripple.zip` from Bob's output, adds visible
LICENSE and NOTICE files, normalizes Linux launch commands generated on
Windows, and preserves executable permissions for the Linux binary in the
archive. It does not change Bob's original output. Import the configs from
the extracted archive into the RLBot GUI.

Bob installs from `requirements.txt`, which pins the runtime dependencies
and PyInstaller. It does not install Torch: the build consumes the committed
ONNX files directly. `pyproject.toml` keeps Torch/ONNX conversion tools in
optional development dependencies. Keep the runtime pins in the two files
aligned when updating dependencies.

To enable BCM-only inference, append `--bcm-only` to the chosen config's run
command. For a Bob-built pack, edit the resulting launch command after building,
because Bob replaces source run commands. Launch options apply to the entire
hivemind instance for that agent ID.

The spec avoids obsolete C++ runtimes found on Windows PATH and embeds the
CC BY-NC-SA 4.0 LICENSE and the attribution/scope NOTICE. The Docker-based Bob
build was checked using a snapshot without Torch or `.pt` inputs. Validate
both binaries with `--self-test` after changing dependencies or packaging.

Before submission, confirm the provisional repository URL in both configs,
check license acceptance with the maintainers, and test both versions against
the botpack's Nexto strength limit in 1v1, 2v2 and 3v3.
