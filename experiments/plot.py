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


_COLORS = plt.get_cmap('tab10').colors
_LINESTYLES = ['-', '--', '-.', ':']


def plot_runs(paths, out_path=None, annotate=False, all_milestones=False, logx=True):
    fig, ax = plt.subplots(figsize=(12, 6))

    runs = []
    for path in paths:
        steps, milestones, summary, label = load_run(path)
        if not steps:
            print(f'No steps in {path}, skipping.')
            continue
        x, y = compute_series(steps)
        runs.append((label, x, y, milestones))

    # draw the runs that go furthest (by final progress) last / on top, so
    # the interesting lines aren't buried under saturated ones sharing the
    # same flat plateau
    runs.sort(key=lambda r: r[2][-1] if r[2] else 0)

    for i, (label, x, y, milestones) in enumerate(runs):
        color = _COLORS[i % len(_COLORS)]
        linestyle = _LINESTYLES[(i // len(_COLORS)) % len(_LINESTYLES)]

        final_progress = y[-1] if y else 0
        final_tokens = x[-1] if x else 0
        full_label = f'{label}  (final {final_progress:.0f}% @ {final_tokens/1000:.0f}k tok)'

        line, = ax.plot(x, y, linewidth=1.8, label=full_label, color=color, linestyle=linestyle)

        # by default, only mark the milestone where a run reaches its highest
        # progress — every combo shares the same early 5/15/20/25% jumps, so
        # dotting all of them just stacks identical markers on top of each other
        shown = milestones if all_milestones else ([max(milestones, key=lambda m: m['progress_after'])]
                                                     if milestones else [])
        for m in shown:
            step_idx = m['step']
            if step_idx < len(x):
                mx, my = x[step_idx], m['progress_after']
                ax.scatter(mx, my, color=color, zorder=5, s=55,
                           edgecolors='black', linewidths=0.6, marker='*' if not all_milestones else 'o')
                if annotate:
                    ax.annotate(
                        f"{m['action']} → {my:.0f}%",
                        xy=(mx, my),
                        xytext=(8, 4),
                        textcoords='offset points',
                        fontsize=7,
                        color=color,
                    )

    ax.set_xlabel('Cumulative tokens used' + (' (log scale)' if logx else ''), fontsize=11)
    ax.set_ylabel('Game progress (%)', fontsize=11)
    ax.set_title('Token usage vs. game progress', fontsize=13)
    if logx:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v/1000:.0f}k' if v >= 1000 else str(int(v))))
    else:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v/1000:.0f}k' if v >= 1000 else str(int(v))))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.set_ylim(bottom=0)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    if len(paths) > 1:
        # legend already ordered by ascending final progress (draw order); reverse
        # so the best-performing run is listed first
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], fontsize=8, loc='center left', bbox_to_anchor=(1.0, 0.5))

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
    parser.add_argument('--annotate', action='store_true',
                         help='Label the marked milestone dot(s) with action text (default off)')
    parser.add_argument('--all-milestones', action='store_true',
                         help='Mark every progress milestone instead of just each run\'s final '
                              '(highest) one — default off, since shared early jumps stack up '
                              'into an illegible cluster when runs saturate at the same progress')
    parser.add_argument('--linear-x', action='store_true',
                         help='Use a linear token axis instead of the default log scale '
                              '(log scale is usually clearer here: it expands the early '
                              'divergence and compresses the long flat saturated tail)')
    args = parser.parse_args()

    paths = [Path(p) for p in args.logs]
    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.exit(f'File(s) not found: {", ".join(str(p) for p in missing)}')

    plot_runs(paths, out_path=args.out, annotate=args.annotate,
              all_milestones=args.all_milestones, logx=not args.linear_x)


if __name__ == '__main__':
    main()
