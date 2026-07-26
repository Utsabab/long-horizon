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
      "progress": float, "score": int, "reward": int|float,
      "tokens": int,          # main generation/intervention tokens + all judge/replan tokens this step
      "judge_tokens": int,    # breakdown: portion of "tokens" spent on coherence/integration/replan judges
      "goal": str|None, "subgoal": str|None,
      "belief": {"type": str, "item": str, "location": str}|None,
      "confidence": float|None,
      "mechanisms_used": [str, ...] }   # e.g. ["m1_info_seeking"], ["m6_integration_gap", "m6_planning"]
  ],
  "drift_events": [              # omitted if empty; one entry each time check_drift() flags
    { "step": int, "off_path_rate": float, "steps_since_on_path": int,
      "window": int, "threshold": float, "recent_actions": [...] }
  ],
  "ale_events": [               # omitted if empty; one entry each time check_ale() confirms
    { "step": int, "belief": [...], "contradiction": bool, "evidence": {...} }
  ],
  "budget_events": [            # omitted if empty; one entry each time check_budget() flags
    { "step": int, "budget_risk": float, "step_ratio": float, "token_ratio": float,
      "stall_ratio": float, "cumulative_tokens": int, "tokens_since_progress": int, "threshold": float }
  ],
  "integration_events": [       # omitted if empty; one entry each time the integration judge confirms a gap
    { "step": int, "unused": [...], "gap": bool, "suggested_combination": str|None }
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
        self._drift_events = []
        self._budget_events = []
        self._integration_events = []
        self._started_at = datetime.now(timezone.utc).isoformat()

    def add_step(self, record):
        """Record one game step. Expected keys: step, obs, action, reasoning, progress,
        score, reward, tokens, judge_tokens, goal, subgoal, belief, confidence, mechanisms_used."""
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

    def add_drift_event(self, record):
        """Record a confirmed drift detection. Expected keys: step, off_path_rate, steps_since_on_path, ..."""
        self._drift_events.append(record)

    def add_budget_event(self, record):
        """Record a confirmed budget/liveness risk. Expected keys: step, budget_risk, step_ratio, ..."""
        self._budget_events.append(record)

    def add_integration_event(self, record):
        """Record a confirmed integration gap. Expected keys: step, unused, gap, suggested_combination."""
        self._integration_events.append(record)

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
        if self._drift_events:
            transcript['drift_events'] = self._drift_events
        if self._budget_events:
            transcript['budget_events'] = self._budget_events
        if self._integration_events:
            transcript['integration_events'] = self._integration_events

        path = self.out_dir / f'{run_id}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, default=str)

        self._reset()
        return path
