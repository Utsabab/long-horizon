"""Run a curated set of harness-mechanism combinations and plot the result.

For one game/seed, runs several independent episodes under different
mechanisms: overrides (vanilla, each mechanism in isolation, one natural
bundle, and all-on), then hands every resulting transcript to plot.py to
render cumulative tokens vs. game progress as one chart with a labeled line
per combination.

M4 (adaptive compute) has no isolated behavioral effect: it's the dispatch
policy that decides *when* M1/M6 fire, so it has nothing to gate on its own.
It's therefore left off in combinations that don't include M1 or M6, and
forced on whenever M1 or M6 is — without it, the runner's fallback bundle
only ever triggers M1 on a confirmed ALE and never triggers M6 at all, which
would make an "M6-only" run behaviorally identical to vanilla.

Usage
-----
    cd experiments
    python run_mechanism_comparison.py wishbringer
    python run_mechanism_comparison.py wishbringer --seed 0 --out compare.png
"""

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from runner import _build_runner, _load_config

# name -> mechanisms overrides (all six keys explicit, so each combo is unambiguous
# regardless of config.yaml's own defaults).
COMBINATIONS = {
    'vanilla': dict(m1_info_seeking=False, m2_memory=False, m3_state_ext=False,
                     m4_adaptive=False, m5_action_templating=False, m6_planning=False),
    'm1_info_seeking': dict(m1_info_seeking=True, m2_memory=False, m3_state_ext=False,
                             m4_adaptive=True, m5_action_templating=False, m6_planning=False),
    'm2_memory': dict(m1_info_seeking=False, m2_memory=True, m3_state_ext=False,
                       m4_adaptive=False, m5_action_templating=False, m6_planning=False),
    'm3_state_ext': dict(m1_info_seeking=False, m2_memory=False, m3_state_ext=True,
                          m4_adaptive=False, m5_action_templating=False, m6_planning=False),
    'm5_action_templating': dict(m1_info_seeking=False, m2_memory=False, m3_state_ext=False,
                                  m4_adaptive=False, m5_action_templating=True, m6_planning=False),
    'm6_planning': dict(m1_info_seeking=False, m2_memory=False, m3_state_ext=False,
                         m4_adaptive=True, m5_action_templating=False, m6_planning=True),
    'm2_m3_m6': dict(m1_info_seeking=False, m2_memory=True, m3_state_ext=True,
                      m4_adaptive=True, m5_action_templating=False, m6_planning=True),
    'all_on': dict(m1_info_seeking=True, m2_memory=True, m3_state_ext=True,
                    m4_adaptive=True, m5_action_templating=True, m6_planning=True),
}


def main():
    parser = argparse.ArgumentParser(description='Compare harness-mechanism combinations on one game.')
    parser.add_argument('game', help="Game name, e.g. 'wishbringer'")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--combos', nargs='+', choices=list(COMBINATIONS), default=list(COMBINATIONS),
                         help='Subset of combinations to run (default: all curated combinations)')
    parser.add_argument('--out', default=None, help='Save comparison plot to this file instead of displaying')
    args = parser.parse_args()

    cfg = _load_config()
    paths = []

    for name in args.combos:
        overrides = COMBINATIONS[name]
        print(f'--- running {name} ({args.game}, seed={args.seed}) ---')
        runner = _build_runner(cfg, game_name=args.game, mechanism_overrides=overrides)
        run_id = runner.run_once(seed=args.seed, label=name)
        paths.append(runner.out_dir / f'{run_id}.json')

    plot_cmd = [sys.executable, str(_HERE / 'plot.py'), *[str(p) for p in paths]]
    if args.out:
        plot_cmd += ['--out', args.out]
    subprocess.run(plot_cmd, check=True)


if __name__ == '__main__':
    main()
