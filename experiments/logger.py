"""Run logger.

Accumulates step records in memory during a run, then writes a single
structured JSON transcript when save() is called.

Output file: logs/<run_id>.json

Schema
------
{
  "run_id":    str,
  "game":      str,
  "summary": {
    "total_steps":    int,
    "final_progress": float,   # 0–100 %
    "final_score":    int,
    "game_over":      bool,
    "started_at":     str,     # ISO-8601
    "completed_at":   str
  },
  "milestones": [              # one entry each time progress increases
    { "step": int, "progress_before": float, "progress_after": float,
      "score": int, "action": str }
  ],
  "steps": [                   # one entry per game step
    { "step": int, "obs": str, "action": str, "reasoning": str,
      "progress": float, "score": int, "reward": int|float }
  ],
  "errors": [                  # omitted if empty
    { "step": int, "error": str }
  ],
  "ale_events": [               # omitted if empty; one entry each time check_ale() confirms
    { "step": int, "belief": [...], "contradiction": bool, "evidence": {...} }
  ]
}
"""

import json
from datetime import datetime, timezone
from pathlib import Path


class RunLogger:
    """Accumulate records for one run and flush to a structured JSON file."""

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self._reset()

    def _reset(self):
        self._steps = []
        self._milestones = []
        self._errors = []
        self._ale_events = []
        self._started_at = datetime.now(timezone.utc).isoformat()

    def add_step(self, record):
        """Record one game step. Expected keys: step, obs, action, reasoning, progress, score, reward."""
        self._steps.append(record)

    def add_milestone(self, record):
        """Record a progress increase. Expected keys: step, progress_before, progress_after, score, action."""
        self._milestones.append(record)

    def add_error(self, record):
        """Record an error. Expected keys: step, error."""
        self._errors.append(record)

    def add_ale(self, record):
        """Record a confirmed Accepted-Local-Error. Expected keys: step, belief, contradiction, evidence."""
        self._ale_events.append(record)

    def save(self, run_id, game_name, summary):
        """Write the transcript and reset for the next run.

        Args:
            run_id:    hex run identifier.
            game_name: e.g. 'zork1'.
            summary:   dict with keys total_steps, final_progress, final_score, game_over.

        Returns:
            Path to the written file.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)

        transcript = {
            'run_id': run_id,
            'game': game_name,
            'summary': {
                **summary,
                'started_at': self._started_at,
                'completed_at': datetime.now(timezone.utc).isoformat(),
            },
            'milestones': self._milestones,
            'steps': self._steps,
        }
        if self._errors:
            transcript['errors'] = self._errors
        if self._ale_events:
            transcript['ale_events'] = self._ale_events

        path = self.out_dir / f'{run_id}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, default=str)

        self._reset()
        return path
