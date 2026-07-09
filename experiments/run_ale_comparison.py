"""Run a baseline-vs-harness ALE comparison and plot the result.

Runs Runner.run_comparison(): a baseline episode plays unaided; if an
Accepted-Local-Error is confirmed, a second episode forks from that exact
decision point with the info-seeking harness active. Both transcripts are
then handed to plot.py to render cumulative tokens vs. game progress for both
branches on one chart.

Usage
-----
    cd experiments
    python run_ale_comparison.py wishbringer
    python run_ale_comparison.py wishbringer --seed 0 --out compare.png
"""

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from runner import _build_runner, _load_config


def main():
    parser = argparse.ArgumentParser(description='Run baseline vs. info-seeking-harness ALE comparison.')
    parser.add_argument('game', help="Game name, e.g. 'wishbringer'")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=None, help='Save comparison plot to this file instead of displaying')
    args = parser.parse_args()

    cfg = _load_config()
    runner = _build_runner(cfg, game_name=args.game)

    baseline_id, harness_id, fork_step = runner.run_comparison(seed=args.seed)

    if harness_id is None:
        sys.exit(f'No ALE detected for {args.game} (seed={args.seed}); nothing to compare.')

    baseline_path = runner.out_dir / f'{baseline_id}.json'
    harness_path = runner.out_dir / f'{harness_id}.json'
    print(f'Forked at step {fork_step}. baseline={baseline_path}  harness={harness_path}')

    plot_cmd = [sys.executable, str(_HERE / 'plot.py'), str(baseline_path), str(harness_path)]
    if args.out:
        plot_cmd += ['--out', args.out]
    subprocess.run(plot_cmd, check=True)


if __name__ == '__main__':
    main()
