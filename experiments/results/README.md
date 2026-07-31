# 5. TextQuests Harness Mechanisms (M1–M6): Implementation and Zork1 Results

## 5.1 Design and background

**Failure channels.** The harness targets four failure hypotheses, one detector each
(`channels/`), summarized briefly here and covered in implementation detail in 5.2:

- **Accepted Local Error (ALE)** — the agent acts on a false belief (e.g. thinks it's
  holding an item it dropped). Detected by repetition (same action, no progress) plus a
  read-only ground-truth probe that checks the agent's self-reported belief against actual
  game state. M1's primary target.
- **Control-path drift** — the agent's reasoning stops being goal-directed (stagnation,
  behavioral loops, or incoherent reasoning), scored as one composite per step. M2's
  primary target.
- **Budget / liveness** — the agent burns steps/tokens without progress (loops, stalls,
  runaway cost). Pure arithmetic on step/token/stall ratios, no LLM call. M4's primary
  target.
- **Integration / synthesis** — the agent holds known pieces (items, discoveries) it never
  combines toward its goal. Cheap pre-filter, LLM judge only when the pre-filter trips. M6's
  primary target.

M5 (action templating) doesn't have its own channel — it's a mechanical guardrail on the
interface between the model's raw output and `env.step()`, not something with a failure
signal to detect.

**TextQuests mechanics used by the harness.** TextQuestsEnv checkpoints the full game state
after every step (`save_checkpoint`/`restore_checkpoint` in `textquests_env.py`) — a
built-in feature of TextQuestsEnv itself, not something added for this project — which is
what lets an episode be rewound and replayed from an earlier decision point. The model's
structured per-step reasoning/goal/subgoal/belief/action, by contrast, is *not* saved by
TextQuestsEnv; that's captured by our own harness (`agent.py`'s structured output +
`logger.py`'s `RunLogger`) and persisted to the run transcript. Game progress itself comes
from TextQuestsEnv's own checkpoint-style scoring: each step it substring-matches the
observation text against a per-game list of milestone strings (`game_progress.json`'s
`checkpoints`) and takes the max percentage reached so far — this runs alongside the game's
own raw score, not instead of it, so transcripts log both (`final_progress`, `final_score`)
side by side every run.

**Branching-simulator** — implemented as `Runner.run_comparison()` in `runner.py`, and
narrower in scope than originally floated: it plays one baseline episode with M1
(info-seeking) disabled; the first time an ALE is confirmed, it uses TextQuestsEnv's
checkpoint to rewind to the step right before that ALE and replays a second ("harness")
episode from there with M1 intervening on every subsequent confirmed ALE. M2–M6 run
identically on both branches, so M1 is the only isolated variable. It is currently
**ALE-triggered and M1-only** — it does not yet support forking at an arbitrary chosen step
or comparing arbitrary mechanism combinations against vanilla; generalizing it is still
open (5.9).

## 5.2 Implementation — M1–M6 harness on TextQuests (new as of 30 Jul)

Built out the full parallel mechanism harness on the TextQuests/Jericho side:

- **`channels/`** — one detector per failure hypothesis, each independently loggable
  regardless of which mechanisms are switched on:
  - `ale.py` — accepted-local-error detector (structured belief vs. ground-truth probe).
  - `drift.py` — composite stagnation + behavioral-loop + reasoning-coherence score
    (`w_stagnation=0.5, w_loop=0.3, w_coherence=0.2`, threshold 0.65); coherence now comes
    from a real per-step LLM judge, not a vague/specific regex.
  - `budget.py` (new) — `BudgetLivenessDetector`: pure arithmetic on `step_ratio`,
    `token_ratio`, `stall_ratio` (tokens burned since last progress increase), composited
    into `budget_risk` at weights 0.3/0.3/0.4, flagged at threshold 0.7. **Detection only —
    it logs risk, it does not stop the run.** The one place it changes behavior is gating
    the integration judge off above `budget_risk ≥ 0.9` (see `mechanisms/adaptive.py`).
  - `integration.py` (new) — `IntegrationDetector`: a free attribute pre-filter (≥2
    discoveries/inventory items unreferenced in the last 10 actions) gates a single LLM
    call (`LLMIntegrationJudge`) that asks whether the agent is holding known pieces it
    hasn't combined — the primary signal for M6.
  - `env_checker.py` — read-only `look`/`inventory` probes via save/restore, used by both
    ALE and M3.
- **`mechanisms/`** — the six harness interventions, independently toggleable via
  `config.yaml`'s `mechanisms:` block:
  - **M1 info_seeking** — prompted correction on a confirmed ALE.
  - **M2 memory** — structured goal/subgoal/discoveries/visited-locations scratchpad
    injected into the prompt (replaces the earlier `_SUBGOAL_RE`/`_DISCOVERY_RE` regex
    scrape of raw reasoning text with fields the model now self-reports directly).
  - **M3 state_ext** — ground-truth location/inventory probe (same save/restore pattern as
    `env_checker.py`), fed into the scratchpad so the agent sees actual state, not just its
    own claim.
  - **M4 adaptive** — `AdaptiveComputeController.decide()`: a many-to-many risk-vector →
    mechanism-bundle dispatcher. Has no isolated behavioral effect of its own; without it,
    M6 never fires and M1 only fires directly off a confirmed ALE — it is the reason M6 and
    the compute-aware gating exist at all, but produces nothing standalone. **Note: M4 does
    not auto-stop stalled runs — it only logs a budget alert and gates one specific LLM
    call; the run loop only exits early on `game_over`.**
  - **M5 action_templating** — `normalize_action()` guardrail (strips label prefixes like
    `ACTION:`/`>`) applied right before every `env.step()` call, on both the normal and the
    M1-intervention path.
  - **M6 planning** — LLM replan of the current subgoal, triggered primarily by a confirmed
    integration gap, secondarily by sustained high drift.
- Also shipped: `run_mechanism_comparison.py` (runs a curated combo sweep — vanilla, each
  of M1/M2/M3/M5/M6 in isolation, the natural M2+M3+M6 bundle, and all-on — on one
  game/seed and plots them together), `tests/test_harness.py` (18 deterministic tests,
  fake-LLM doubles for every judge/replan call, no API key needed), and the experiment logs
  analyzed below.
- Repo: `github.com/Utsabab/long-horizon`

## 5.3 Experimental setup

All results below use model `anthropic/claude-sonnet-5` and the logged `steps[].tokens`
field (judge-inclusive — every ALE/drift/integration/replan call is counted, not just the
main action-generation call). Five sets of runs are covered, the first four on Zork1
(5.4–5.6), the last two on Wishbringer (5.7–5.8):

1. **Pairwise pilots (100 steps, Zork1)** — the first isolated-mechanism checks, run before
   the full 8-combo sweep existed: vanilla vs. M1 alone, and separately vanilla vs. M2 alone.
2. **8-combo sweep, 100 steps (Zork1)** — vanilla, each of M1/M2/M3/M5/M6 alone, the
   M2+M3+M6 bundle, and all-on, run together via `run_mechanism_comparison.py`.
3. **8-combo sweep, 500 steps (Zork1)** — the same 8 combinations at a 5x step budget.
4. **8-combo sweep, 100 and 500 steps (Wishbringer)** — the same 8 combinations, same
   `run_mechanism_comparison.py` driver, run on a second game.

| Combo (8-combo sweep) | Mechanisms on |
|---|---|
| `vanilla` | none |
| `m1_info_seeking` | M1 + M4 |
| `m2_memory` | M2 only |
| `m3_state_ext` | M3 only |
| `m5_action_templating` | M5 only |
| `m6_planning` | M6 + M4 |
| `m2_m3_m6` | M2 + M3 + M4 + M6 |
| `all_on` | all six |

## 5.4 Results — pairwise pilots (100 steps)

Source logs: `vanilla_m1_ex100(vanilla).json` / `vanilla_m1_ex100(M1).json`, and
`vanilla_m2_ex100(vanilla).json` / `vanilla_m2_ex100(M2).json`. Figures:
`results/van_m1_ex100.png`, `results/van_m2_ex100.png`, `results/van_m2.png`.

| Run | Steps | Final progress | Final score | Total tokens | Judge tokens | Drift flags | ALE flags |
|---|---|---|---|---|---|---|---|
| vanilla (M1 pilot's baseline) | 100 | 25% | 50 | 1,052k | 54k | 0 | 1 |
| m1_info_seeking | 100 | 25% | 44 | 1,457k | 67k | 0 | 0 |
| vanilla (M2 pilot's baseline) | 100 | 40% | 50 | 1,142k | 54k | 2 | 0 |
| m2_memory | 100 | 40% | 50 | 1,013k | 46k | 0 | 0 |

In these pilots, neither M1 nor M2 alone clearly beats its own paired vanilla: M1 clears its
one logged ALE flag but ends with a lower final score (44 vs. 50) at 39% more tokens; M2
matches its paired vanilla's progress and score at slightly fewer tokens, but the vanilla
baseline here already reaches 40% on its own, so the comparison doesn't isolate an M2 effect.
Note the two "vanilla" baselines above disagree with each other (25% vs. 40% final progress)
despite being nominally the same condition — a reminder that these pilots were not run at a
fixed shared seed, unlike the 8-combo sweep in 5.5/5.6, so pilot numbers should be read as
rough single-run checks rather than a controlled comparison.

## 5.5 Results — 8-combo sweep, 100 steps

Figure: `results/vanall_100_zork1.png` (log-token-x, one line per combo, peak-milestone
starred), `seed=0` for all 8 runs.

| Combo | Steps | Final progress | Final score | Ended in death? | Total tokens | Judge tokens | Drift flags | ALE flags | Integration checks |
|---|---|---|---|---|---|---|---|---|---|
| vanilla | 100 | 25% | 44 | no | 769k | 21k | 57 | 0 | 0 |
| m1_info_seeking | 100 | **40%** | 50 | no | 985k | 45k | 0 | 0 | 0 |
| m2_memory | 100 | 25% | 50 | no | 994k | 33k | 33 | 8 | 0 |
| m3_state_ext | 21 | 15% | 25 | **yes** | 55k | 8k | 0 | 0 | 0 |
| m5_action_templating | 21 | 15% | 25 | **yes** | 51k | 7k | 0 | 0 | 0 |
| m6_planning | 100 | 25% | 40 | no | 1,108k | 46k | 27 | 0 | 0 |
| m2_m3_m6 | 40 | 25% | 50 | no | 178k | 37k | 0 | 0 | 18 |
| all_on | 100 | 25% | 50 | no | 1,187k | 133k | 17 | 0 | 39 |

Notable: **m1_info_seeking is the only combo that clears the 25% plateau at 100 steps**
(40% final progress), and does so with zero drift flags, while every other combo (including
all_on) logs drift or ALE events along the way. `m2_m3_m6` reaches the same 50-score
plateau as `all_on` and `m2_memory` at roughly **1/7th the token cost** (178k vs. 1,187k /
994k) — but it gets there in 40 steps, not the full 100, so this reflects the agent's own
trajectory terminating early (looping/giving up), not a mechanism-driven efficiency win.

Both `m3_state_ext` and `m5_action_templating` die at step 21, in both cases from the same
cause: the agent repeats `KILL TROLL WITH SWORD` at steps 18–20 with progress stuck at 15%,
then dies at step 21. This is a genuine in-game failure (Zork's troll fight is notoriously
unreliable without retry variation), not a harness bug — neither M3 nor M5 targets "vary
your action if it keeps failing," which is closer to what M1's ALE-triggered correction is
built for.

## 5.6 Results — 8-combo sweep, 500 steps

Figure: `results/vanall_500_zork1.png`, `seed=0` for all 8 runs.

| Combo | Steps | Final progress | Final score | Ended in death? | Total tokens | Judge tokens | Drift flags | ALE flags | Integration checks |
|---|---|---|---|---|---|---|---|---|---|
| vanilla | 500 | **40%** | 99 | no | 20.6M | 183k | 167 | 11 | 0 |
| m1_info_seeking | 500 | 35% | 79 | no | 26.3M | 205k | 189 | 11 | 0 |
| m2_memory | 235 | 25% | 40 | **yes** | 3.5M | 77k | 196 | 145 | 0 |
| m3_state_ext | 500 | 35% | 68 | no | 36.1M | 377k | 59 | 6 | 0 |
| m5_action_templating | 268 | **40%** | 50 | no | 6.9M | 98k | 112 | 6 | 0 |
| m6_planning | 500 | **40%** | 79 | no | 34.9M | 340k | 71 | 11 | 0 |
| m2_m3_m6 | 153 | **40%** | 45 | **yes** | 2.8M | 273k | 61 | 0 | 59 |
| all_on | 19 | 15% | 25 | **yes** | 68k | 20k | 0 | 0 | 8 |

Headline result of this sweep: **`all_on` dies at step 19** — far short of even the
100-step sweep's own `all_on` run — while `m5_action_templating` alone reaches the same 40%
progress ceiling in 268 steps at 6.9M tokens. Stacking all six mechanisms together produced
the *worst* outcome in this sweep, not the best. `m2_m3_m6` also dies (step 153) but still
reaches 40% progress before doing so. No combo in the 500-step sweep exceeds vanilla's 99
final score, though several (m5, m6, m2_m3_m6) match its 40% progress ceiling for a
fraction of the token spend, and progress plateaus at 35–40% across nearly every combo
while token spend keeps climbing.

## 5.7 Results — 8-combo sweep, Wishbringer, 100 steps

Seed 0 for all 8 runs (`vanilla_comboall_wishbringer_100(*).json`). Wishbringer's opening
obstacle is a poodle blocking the road outside Miss Voss's cottage; the checkpoint scoring
in `game_progress.json` puts the first milestone (picking up the delivery envelope) at 5%.

| Combo | Steps | Final progress | Final score | Terminal outcome | Total tokens | Judge tokens | Drift flags | ALE flags | Integration checks |
|---|---|---|---|---|---|---|---|---|---|
| vanilla | 100 | 5% | 6 | no (step cap) | 884k | 9k | 76 | 0 | 0 |
| m1_info_seeking | 100 | 5% | 6 | no (step cap) | 926k | 11k | 71 | 0 | 0 |
| m2_memory | 100 | 5% | 6 | no (step cap) | 1,043k | 27k | 34 | 7 | 0 |
| m3_state_ext | 100 | 5% | 6 | no (step cap) | 887k | 10k | 74 | 0 | 0 |
| m5_action_templating | 100 | 5% | 6 | no (step cap) | 1,094k | 15k | 71 | 7 | 0 |
| m6_planning | 100 | 5% | 6 | no (step cap) | 1,153k | 35k | 65 | 0 | 0 |
| m2_m3_m6 | 100 | 5% | 6 | no (step cap) | 1,218k | 174k | 78 | 0 | 93 |
| all_on | 100 | 5% | 6 | no (step cap) | 936k | 114k | 77 | 1 | 92 |

**Every combo lands on the identical 5% progress / score 6 plateau at 100 steps** — none
of the eight get past the poodle within 100 steps, regardless of tokens spent (936k for
`all_on` vs. 1,218k for `m2_m3_m6`, no better outcome for the extra ~280k tokens). `m2_m3_m6`
and `all_on` rack up the heaviest integration-judge activity of the whole sweep (93 and 92
checks respectively) with zero payoff in progress — the agent is repeatedly flagged as
holding unused pieces (it's carrying the envelope, has heard about the poodle, etc.) without
ever combining them into "give it something to eat."

## 5.8 Results — 8-combo sweep, Wishbringer, 500 steps

Seed 0 for all 8 runs (`vanilla_comboall_wishbringer_500(*).json`).

| Combo | Steps | Final progress | Final score | Terminal outcome | Total tokens | Judge tokens | Drift flags | ALE flags | Integration checks |
|---|---|---|---|---|---|---|---|---|---|
| vanilla | 334 | 5% | 6 | fired (in-game timeout) | 13,975k | 105k | 242 | 0 | 0 |
| m1_info_seeking | 340 | 5% | 6 | fired (in-game timeout) | 14,431k | 110k | 240 | 1 | 0 |
| m2_memory | 364 | **12%** | **13** | fired (in-game timeout) | 15,887k | 126k | 203 | 9 | 0 |
| m3_state_ext | 197 | 5% | 1 | stopped early, no game_over | 3,759k | 27k | 143 | 0 | 0 |
| m5_action_templating | 282 | 5% | 6 | fired (in-game timeout) | 6,800k | 26k | 241 | 2 | 0 |
| m6_planning | 320 | 5% | 6 | stopped early, no game_over | 15,018k | 213k | 193 | 0 | 0 |
| m2_m3_m6 | 300 | 5% | 6 | fired (in-game timeout) | 8,872k | 584k | 260 | 7 | 237 |
| all_on | 295 | 5% | 6 | fired at a prior attempt / stopped early | 8,992k | 676k | 225 | 0 | 266 |

Two headline results, both new failure patterns not visible in the 100-step Wishbringer
sweep or in either Zork1 sweep:

**All eight combos self-issue a `RESTART` command mid-episode, between steps 127 and 210,
believing they have already been fired** — before that ever actually happens in-game.
Concretely: the model's own reasoning concludes something like *"I failed to deliver the
envelope in time and got fired, ending the game. I need to restart"* and it plays `RESTART`
as a normal action, which the interpreter honors (Wishbringer supports it as a real verb),
resetting score/progress back to their step-0 baseline. None of the four channels catch
this: it isn't an ALE (no belief/ground-truth contradiction — the agent isn't wrong about
its inventory or location), it's not what the drift detector's incoherence judge is built
to catch (the reasoning is internally consistent, just factually wrong about the game
clock), and neither budget nor integration are aimed at "hallucinated terminal state."
Every combo hits this at roughly the same wall-clock point (steps 127–210), independent of
which mechanisms are on — it looks like a property of the underlying model's narrative
expectations for a "time pressure" framing, not something any current harness mechanism
addresses.

**Only `m2_memory` escapes the poodle obstacle at all**, via `GIVE BONE TO POODLE` at step
91 (progress 5%→12%, score 10→13) — the single best result across both Wishbringer sweeps.
Every other combo, `all_on` included, remains stuck at the poodle for the entire run. Note
`m3_state_ext`'s final score of 1 is *lower* than its own 100-step run's score of 6: its
self-restart at step 144 reset score to 0, and the run's harness-level step/token counters
kept accumulating post-restart without recognizing the reset, so the "197 steps, score 1"
result reflects a truncated post-restart continuation, not a clean single trajectory.
`m3_state_ext`, `m6_planning`, and `all_on` all stop noticeably short of the 500-step cap
with `game_over: false` — consistent with being manually halted once stuck in the same
restart→re-stall pattern a second time, rather than a designed termination.

## 5.9 Open questions / caveats

- **Single seed** — every sweep table above (5.5, 5.6, 5.7, 5.8) is `seed=0` only; the
  pairwise pilots (5.4) weren't even run at a matched seed between their vanilla and
  mechanism runs. None of the effects above should be read as more than single-run signal
  yet.
- **`all_on`'s step-19 death (5.6) is one data point** — before treating "stacking all
  mechanisms is actively harmful" as a claim, this needs at least 2–3 more seeds to rule out
  an unlucky combat RNG draw versus a genuine interaction effect between mechanisms (e.g.
  M6 replanning competing with M2's scratchpad for prompt budget, or M4's judge-skipping
  under budget risk cutting off M6 at a bad moment).
- **The Wishbringer 500-step sweep (5.8) has its own single-run caveat: the universal
  self-restart hallucination (steps 127–210, all 8 combos) is itself unreplicated** —
  worth checking against another seed before treating "the model spontaneously restarts
  under simulated time pressure" as a stable property rather than a one-seed artifact of
  this particular play-through's pacing.
- **Two games now covered, not yet cross-compared in depth** — 5.5/5.6 (Zork1) and 5.7/5.8
  (Wishbringer) use the same 8 combos and driver, but the games fail very differently:
  Zork1's plateau is combat/parser-driven (the troll fight), Wishbringer's is a single
  early puzzle gate (the poodle) compounded by a model-side hallucinated restart that Zork1
  never exhibited in any combo. Whether that's a per-game quirk or something about
  Wishbringer's "beat the clock" framing specifically inducing it is untested — the two
  games differ in more than one axis (genre, deadline mechanic, puzzle vs. combat), so any
  channel-effect comparison across them should stay tentative for now. The older
  `wbprogress.png`/`wishbringer1.png`/`zorkprogress.png`/`zork1.png` images in `results/`
  still predate the current 6-mechanism harness and aren't comparable to either game's
  tables above.
- **Branching-simulator is ALE/M1-only (5.1)** — `run_comparison()` only forks on a
  confirmed ALE and only isolates M1; none of the results in 5.4–5.8 were produced through
  it (they're independent full episodes, one per combo, not matched forks). Generalizing it
  to fork at an arbitrary step and isolate any mechanism/combo — which would let 5.5/5.6's
  comparisons run on identical game states instead of merely the same seed — is not yet
  built or scheduled.

## 5.10 Hypotheses from the 8-combo sweeps, vs. the original research agenda

The original agenda (*Harness Mechanisms and Failure Channels in Long-Horizon Game
Agents*) frames four failure channels — world-state tracking, goal/subgoal forgetting,
integration (combining known information into a decision), and domain knowledge — and
states four expected results (§5): (a) state-tracking mechanisms should reduce their
targeted channel specifically, not lift scores uniformly; (b) that channel-specific effect
should persist on stronger models; (c) planning should improve the integration category at
every level of state knowledge; (d) harness mechanisms should qualitatively change *how* an
agent fails even when they don't change *whether* it succeeds. Its own CookingWorld pilot
(Llama 3.2 3B / Gemini Flash) found planning +0.07, state-externalization ~+0.01 (crosses
zero), memory -0.15 (early buggy harness), and — separately — that giving world state
directly improved recall but barely moved score, because CookingWorld's strict
verb-to-appliance mapping made many failures a domain-knowledge problem rather than a
state-tracking one. It also self-flags an unresolved compute confound: score differences
may just reflect some harnesses spending more tokens, not doing anything qualitatively
better (§3.1).

Below, "the paper" refers to that agenda; "here"/"our data" refers to the Zork1 sweeps in
5.5–5.6.

**Where our data supports the paper's hypotheses:**

- **(d) is well supported.** At 100 steps, `vanilla`, `m2_memory`, `m6_planning`, and
  `all_on` all land in the same 25% progress / 40–50 score band, but with unrelated failure
  signatures — vanilla logs 57 drift flags and 0 ALE; `m2_memory` logs 33 drift + 8 ALE;
  `m6_planning` logs 27 drift + 0 ALE; `all_on` logs 17 drift + 39 integration checks.
  Nearly identical outcomes, different failure channels active underneath — exactly the
  paper's point that scores alone hide what's actually going wrong.
- **The paper's domain-knowledge/state-tracking dissociation reappears here.** The paper's
  CookingWorld finding — recall improves but score doesn't, because the real bottleneck is
  domain knowledge (verb-appliance mapping) — has a structural match in `m3_state_ext` at
  500 steps: it spends the *most* tokens of any combo (36.1M) with clean state-tracking
  (ale=6, second-lowest of all combos) but scores *worse* than vanilla (68 vs. 99). Zork's
  troll fight is the sharpest version: `m3_state_ext` and `m5_action_templating` both die at
  step 21 in the 100-step sweep repeating `KILL TROLL WITH SWORD`, with perfect knowledge of
  their own state (sword in hand, troll present) — the failure is not knowing *which*
  action variant the parser/combat system will accept, a domain-knowledge gap neither
  mechanism targets.
- **The compute confound the paper flags as a risk to its own conclusion shows up directly
  in our numbers.** `m6_planning` spends 1.1M (100-step) to 34.9M tokens (500-step) — more
  than nearly every other combo — for outcomes at or below vanilla's. This is exactly the
  scenario the paper's §5 says would overturn a positive planning conclusion, and here it
  looks like it does: no extra-token combo beats vanilla's 500-step score of 99.

**Where our data contradicts or complicates the paper's hypotheses:**

- **Stacking mechanisms is not more-is-better, contradicting the implicit assumption that
  harness effects compose.** The paper's framing (isolate which mechanism moves which
  channel) implies combining the winners should help. Our `all_on` result is a direct
  counterexample: it is the single worst outcome in the 500-step sweep (dies at step 19, 15%
  progress) even though `m5_action_templating` alone reaches 40% progress in 268 steps.
  Something about the full stack is actively harmful, not merely neutral.
- **M6 can't actually be isolated the way an ablation study assumes — this is a measurement
  gap, not just a result.** `m6_planning` run alone (no M2/M3) logs **zero** integration
  checks at both 100 and 500 steps, because `IntegrationDetector`'s pre-filter requires ≥2
  unused discoveries/inventory items, which only M2 and M3 populate. Only `m2_m3_m6` and
  `all_on` ever show integration activity (18–59 checks). So "M6 alone" never exercises M6's
  own target channel in this setup — it's structurally confounded with M2/M3's presence,
  which complicates any claim (c) about planning's effect at varying levels of state
  knowledge, since the isolated-M6 condition can't produce that variation on its own.
- **The step-survival pattern the paper reports for planning reverses under stacking.** The
  paper's Figure 3 found planning-enabled failing runs survive longer before failing
  (median 59 vs. 20 baseline steps). `m6_planning` alone is consistent with this (runs the
  full 500 steps, never dies). But `all_on` — which includes planning — dies fastest of any
  combo (step 19). Planning's step-extending effect doesn't hold up once other mechanisms
  are layered on top of it.
- **M1 shows a channel-specific win, but on the wrong channel.** At 100 steps,
  `m1_info_seeking` is the only combo to break the 25% plateau (40%), with zero drift flags
  — a clean, channel-specific-looking signature as the paper's (a) predicts. But M1's
  primary target is ALE, and the visible effect here is on drift, not ALE (which is already
  near-zero across most combos). Whether M1 is actually working through its intended
  ALE-correction path, or through some side effect on drift, is unresolved and worth
  testing directly.

**Gap the paper's taxonomy has that our harness doesn't cover:** the paper's four channels
include "lack of domain knowledge," with a dedicated game-rules mechanism as the fix. This
harness has no analogous detector — the `m3`/`m5` troll-fight deaths (repeating a failing
combat action without variation, despite accurate state) look like exactly this failure
category slipping through undetected by all four implemented channels, mirroring the
paper's own experience of a domain-knowledge failure initially looking like a
state-tracking one until specifically investigated.

**New hypotheses worth testing next:**

- **H1 — anti-synergy between M6 and M2/M4 under budget pressure.** `all_on`'s early death
  may come from M4's `skip_integration_judge_above_budget_risk` gate cutting off M6 exactly
  when M2's growing scratchpad has pushed `budget_risk` up — i.e. M2 and M6 competing for
  the same compute headroom M4 is protecting. Testable by running `m2_m6` (no M3/M5) against
  `all_on`.
- **H2 — M3/M5 need an explicit "vary the action on repeated failure" fallback, not more
  state.** Since both die identically at the troll fight with perfect state info, the fix
  isn't more state-tracking or valid-action normalization — it's closer to M1's
  correction loop, applied to action repetition specifically rather than belief
  contradiction.
- **H3 — the integration channel needs a minimum-memory precondition, not an
  independent on/off condition.** Before running further isolated M6 tests, the comparison
  set should be "M6 with minimal memory" vs. "M6 without," rather than "M6 alone" vs.
  vanilla, since the latter can't exercise M6's target channel at all.
