# Running experiments and producing comparison plots

This covers everything needed to go from a clean checkout to a mechanism-comparison
plot: environment setup, running a single episode, and running the curated
harness-mechanism comparison (`run_mechanism_comparison.py`) or the ALE-specific
baseline-vs-harness comparison (`run_ale_comparison.py`).

## 1. Prerequisites

**Python packages** (no `requirements.txt` in this repo yet — install directly):

```bash
pip install requests pyyaml matplotlib
```

**TextQuests game data** — a git submodule at `textquests/`:

```bash
cd "long-horizon"
git submodule update --init
```

If you'd rather point at a checkout elsewhere, set `TEXTQUESTS_ROOT` instead of
using the submodule (`env.py`'s `resolve_root()` checks, in order: `TEXTQUESTS_ROOT`
env var → `config.yaml`'s `textquests_root` key → the local `textquests/` submodule).

**OpenRouter API key** — every run makes real LLM calls and spends API credits.
Export it in your shell (or add `api_key:` under `openrouter:` in `config.yaml`):

```bash
export OPENROUTER_API_KEY=sk-...
```

## 2. Configuration (`experiments/config.yaml`)

All run behavior is driven by this file — no code changes needed to tune a run.
Key sections:

- `openrouter.model` — which model to call (currently `anthropic/claude-sonnet-5`).
- `experiments.max_steps` — episode length cap. **Lower this (e.g. to 5–10) for a
  quick smoke test** before running a full 500-step episode.
- `experiments.games` — the games available to pick from (`zork1`, `wishbringer`).
- `experiments.drift`, `experiments.budget`, `experiments.integration` — thresholds
  and weights for the three composite channel detectors (drift, budget/liveness,
  integration). These detectors always run and log events regardless of the
  `mechanisms` block below.
- `experiments.memory` — scratchpad size limits for M2 (only takes effect if
  `mechanisms.m2_memory` is on).
- `experiments.mechanisms` — the master on/off switch for each harness mechanism:

  ```yaml
  mechanisms:
    m1_info_seeking: true      # lead: prompted correction on confirmed ALE
    m2_memory: true            # structured scratchpad
    m3_state_ext: true         # ground-truth location/inventory probe
    m4_adaptive: true          # risk-vector -> mechanism-bundle dispatch
    m5_action_templating: true # normalize_action guardrail before env.step
    m6_planning: true          # LLM replan on integration gap or sustained drift
  ```

  Note: **M4 has no isolated behavioral effect** — it's the dispatch policy that
  decides *when* M1/M6 fire. If `m4_adaptive: false`, M1 still fires directly on a
  confirmed ALE, but M6 never fires at all (the fallback bundle hardcodes
  `trigger_replan_from_drift: false`). Any combo that includes M1 or M6 should
  keep M4 on.

You can edit `config.yaml` directly for one-off runs, or pass per-run overrides
programmatically (see `run_mechanism_comparison.py`'s `mechanism_overrides` pattern
below) without touching the file at all.

## 3. Running a single episode

```bash
cd experiments
python runner.py wishbringer
```

This builds a `Runner` from `config.yaml`, plays one episode with whatever
mechanisms are enabled there, and writes a transcript to `logs/<run_id>.json`.
Console output shows a line per progress milestone and a final summary line
with the log path, e.g.:

```
Loading game wishbringer.
[472776c] step    2  0.0% → 5.0%  score 1  action: LISTEN TO CRISP
[472776c] done — 6 steps, progress 5.0%  →  experiments/logs/472776ce....json
```

To play a different game, pass its name (must be one `find_game_path` can
resolve under `textquests/data/`, e.g. `zork1`).

## 4. Plotting a single run

```bash
cd experiments
python plot.py logs/<run_id>.json --out progress.png
```

- Omit `--out` to open an interactive matplotlib window instead of saving.
- X-axis: cumulative tokens used (this includes all judge/replan overhead, not
  just the main generation call — see `steps[].tokens` in the transcript schema
  documented at the top of `logger.py`).
- Y-axis: game progress (%).
- Dots mark each progress milestone, annotated with the action that caused it.

You can pass multiple paths to overlay several runs on one chart:

```bash
python plot.py logs/run1.json logs/run2.json --out compare.png
```

Each line is labeled using the transcript's `summary.label` field (falls back
to `<game> run <id prefix>` if no label was set).

## 5. Curated mechanism comparison (`run_mechanism_comparison.py`)

This is the main deliverable: run a fixed set of mechanism combinations on the
same game/seed and plot cumulative tokens vs. progress for all of them on one
chart.

```bash
cd experiments
python run_mechanism_comparison.py wishbringer --seed 0 --out compare.png
```

This runs all 8 curated combinations in sequence, each as its own episode using
`config.yaml`'s settings with only the `mechanisms` block overridden in memory
(the file itself is never modified):

| Combo name | What's on |
|---|---|
| `vanilla` | nothing — pure baseline |
| `m1_info_seeking` | M1 + M4 (M4 forced on so M1 can dispatch) |
| `m2_memory` | M2 only |
| `m3_state_ext` | M3 only |
| `m5_action_templating` | M5 only |
| `m6_planning` | M6 + M4 (M4 forced on so M6 can dispatch) |
| `m2_m3_m6` | M2 + M3 + M6 + M4 (the natural memory+state+planning bundle) |
| `all_on` | every mechanism |

Each combo's transcript is saved with a distinct `summary.label` so `plot.py`
renders one labeled line per combo (that's what the script shells out to at
the end).

**Useful flags:**

```bash
# only run a subset of combos
python run_mechanism_comparison.py wishbringer --combos vanilla all_on --out compare.png

# different seed
python run_mechanism_comparison.py wishbringer --seed 3 --out compare.png

# omit --out to display interactively instead of saving
python run_mechanism_comparison.py wishbringer
```

**Before running the full sweep**, it's worth doing a cheap smoke test first —
temporarily set `experiments.max_steps: 6` (or similar) in `config.yaml`, run a
2-combo subset (`--combos vanilla all_on`), confirm the plot looks sane, then
restore `max_steps` to its real value (e.g. 500) before the real run. The full
8-combo sweep at 500 steps each makes a lot of LLM calls and will take a while
and spend real API credits.

## 6. ALE-specific baseline-vs-harness comparison (`run_ale_comparison.py`)

A narrower, older comparison: plays one baseline episode unaided; the moment an
Accepted-Local-Error is confirmed, it forks a second episode from that exact
step with the M1 info-seeking harness turned on, so you can see the two
branches diverge right at the failure point.

```bash
cd experiments
python run_ale_comparison.py wishbringer --seed 0 --out compare_ale.png
```

If no ALE is ever confirmed during the baseline episode, only the baseline
branch is plotted (the script prints a message and skips the harness branch).

## 7. Inspecting a transcript directly

Transcripts are plain JSON (`logs/<run_id>.json`); the schema is documented in
full at the top of `logger.py`. Useful fields per step: `action`, `reasoning`,
`goal`, `subgoal`, `belief`, `confidence`, `tokens` (total, judge-inclusive),
`judge_tokens` (breakdown), and `mechanisms_used` (e.g.
`["m6_integration_gap", "m6_planning"]`). Top-level `drift_events`,
`ale_events`, `budget_events`, and `integration_events` arrays (omitted when
empty) record every time a channel detector actually flagged something.

```bash
python -c "
import json
data = json.load(open('logs/<run_id>.json'))
print(data['summary'])
for s in data['steps']:
    print(s['step'], s['action'], s['tokens'], s['mechanisms_used'])
"
```

## 8. Running the test suite

```bash
cd experiments
python -m pytest tests/test_harness.py
```

These are all deterministic/offline (fake LLM client doubles for the
judge/planning mechanisms) — no API key or network access required.
