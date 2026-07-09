"""Plot cumulative token usage vs. game progress for one or more runs.

Usage
-----
    cd experiments

    # single run
    python plot.py logs/abc123.json

    # compare multiple runs
    python plot.py logs/abc123.json logs/def456.json

    # all runs in the logs folder
    python plot.py logs/*.json

    # save to file instead of showing interactively
    python plot.py logs/abc123.json --out progress.png
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    sys.exit('matplotlib is required:  pip install matplotlib')


def load_run(path):
    with open(path) as f:
        data = json.load(f)
    steps = data.get('steps', [])
    milestones = data.get('milestones', [])
    summary = data.get('summary', {})
    branch_label = summary.get('label')
    if branch_label:
        label = f"{data.get('game', '?')} - {branch_label}"
    else:
        label = f"{data.get('game', '?')}  run {data.get('run_id', '')[:8]}"
    return steps, milestones, summary, label


def compute_series(steps):
    """Return (cumulative_tokens, progress) lists, one point per step."""
    cum_tokens = []
    progress = []
    total = 0
    for s in steps:
        total += s.get('tokens', 0)
        cum_tokens.append(total)
        progress.append(s.get('progress', 0))
    return cum_tokens, progress


def plot_runs(paths, out_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))

    for path in paths:
        steps, milestones, summary, label = load_run(path)
        if not steps:
            print(f'No steps in {path}, skipping.')
            continue

        x, y = compute_series(steps)

        line, = ax.plot(x, y, linewidth=1.8, label=label)

        # mark each progress milestone with a dot + annotation
        for m in milestones:
            # find cumulative tokens at this step
            step_idx = m['step']
            if step_idx < len(x):
                mx, my = x[step_idx], m['progress_after']
                ax.scatter(mx, my, color=line.get_color(), zorder=5, s=60)
                ax.annotate(
                    f"{m['action']} → {my:.0f}%",
                    xy=(mx, my),
                    xytext=(8, 4),
                    textcoords='offset points',
                    fontsize=7,
                    color=line.get_color(),
                )

    ax.set_xlabel('Cumulative tokens used', fontsize=11)
    ax.set_ylabel('Game progress (%)', fontsize=11)
    ax.set_title('Token usage vs. game progress', fontsize=13)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v/1000:.0f}k' if v >= 1000 else str(int(v))))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=0.4)
    if len(paths) > 1:
        ax.legend(fontsize=8)

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f'Saved to {out_path}')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot token usage vs. game progress.')
    parser.add_argument('logs', nargs='+', help='Path(s) to run JSON files')
    parser.add_argument('--out', default=None, help='Save plot to this file instead of displaying')
    args = parser.parse_args()

    paths = [Path(p) for p in args.logs]
    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.exit(f'File(s) not found: {", ".join(str(p) for p in missing)}')

    plot_runs(paths, out_path=args.out)


if __name__ == '__main__':
    main()
