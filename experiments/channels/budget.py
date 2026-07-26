"""Budget / Liveness detector (research agenda §2.1 — budget/liveness channel).

Budget/liveness: the agent spends compute badly — looping, stalling, retrying
without ever running out of "moves" in the game sense but burning steps and
tokens without progress. This is M4 (Adaptive compute)'s primary target.

Pure arithmetic
---------------
Unlike ale.py (belief verification) and drift.py (an LLM coherence judge),
this detector needs no LLM call and no text parsing at all — every input is
already numeric and already flowing through the run loop (step index,
per-step token count, and whether progress increased this step).

Three components, each in [0, 1]:

  1. step_ratio    steps_used / max_steps — how much of the step budget is
                    already spent.
  2. token_ratio    cumulative_tokens / token_budget (0 if no token_budget is
                    configured) — how much of the token budget is spent.
  3. stall_ratio    tokens spent since the last progress increase, normalized
                    against stall_token_threshold — high when the agent keeps
                    burning tokens without the game rewarding it.

Composite
---------
  budget_risk = w_step * step_ratio + w_token * token_ratio + w_stall * stall_ratio

  is_at_risk when budget_risk >= threshold.
"""


class BudgetLivenessDetector:
    """Detect wasted-compute / stalled-liveness risk from run bookkeeping alone."""

    def __init__(self, max_steps, token_budget=None, stall_token_threshold=4000,
                 w_step=0.3, w_token=0.3, w_stall=0.4, threshold=0.7):
        """
        Args:
            max_steps: the run's configured step budget.
            token_budget: optional total token budget for the run; when None,
                          the token_ratio component is always 0.
            stall_token_threshold: tokens spent since the last progress
                          increase at which stall_ratio saturates to 1.0.
            w_step, w_token, w_stall: composite weights (should sum to 1.0).
            threshold: budget_risk at or above which is_at_risk is True.
        """
        self.max_steps = max_steps
        self.token_budget = token_budget
        self.stall_token_threshold = stall_token_threshold
        self.w_step = w_step
        self.w_token = w_token
        self.w_stall = w_stall
        self.threshold = threshold

        self._last_step = -1
        self._cumulative_tokens = 0
        self._tokens_since_progress = 0

    def add(self, step, tokens, progress_increased):
        """Record one step's bookkeeping.

        Args:
            step: the step index just completed.
            tokens: tokens spent generating this step's action.
            progress_increased: True if game_progress increased this step.
        """
        self._last_step = step
        self._cumulative_tokens += tokens or 0
        if progress_increased:
            self._tokens_since_progress = 0
        else:
            self._tokens_since_progress += tokens or 0

    def check_budget(self):
        """Return (is_at_risk: bool, details: dict)."""
        step_ratio = min(1.0, (self._last_step + 1) / self.max_steps) if self.max_steps else 0.0
        token_ratio = (min(1.0, self._cumulative_tokens / self.token_budget)
                       if self.token_budget else 0.0)
        stall_ratio = (min(1.0, self._tokens_since_progress / self.stall_token_threshold)
                       if self.stall_token_threshold else 0.0)

        risk = (self.w_step * step_ratio
                + self.w_token * token_ratio
                + self.w_stall * stall_ratio)
        is_at_risk = risk >= self.threshold

        details = {
            'budget_risk': round(risk, 3),
            'step_ratio': round(step_ratio, 3),
            'token_ratio': round(token_ratio, 3),
            'stall_ratio': round(stall_ratio, 3),
            'cumulative_tokens': self._cumulative_tokens,
            'tokens_since_progress': self._tokens_since_progress,
            'threshold': self.threshold,
        }
        if not is_at_risk:
            details['reason'] = 'below_threshold'
        return is_at_risk, details
