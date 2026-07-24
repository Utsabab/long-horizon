"""Control-Path Drift detector (research agenda §2.1 — drift channel).

Drift: the agent loses its goal or subgoal and acts without purpose —
looping, re-exploring already-solved areas, or issuing semantically empty
commands.

Three-signal composite
----------------------
Rather than comparing against a single walkthrough oracle (brittle, path-specific),
we measure drift as a behavioural pattern across three independent signals:

  1. Progress stagnation
       Fraction of the rolling window in which game_progress did not increase.
       A stuck agent keeps acting but the game does not reward it.

  2. Behavioral loop rate
       1 - (unique actions / total actions) in the window.
       A looping agent issues the same commands repeatedly, getting nowhere.

  3. Reasoning incoherence
       Fraction of reasoning language that is vague/exploratory vs.
       goal-directed/specific.  Measured with lightweight regex — no extra
       LLM call required.

Composite
---------
  drift_score = w_stag * stagnation + w_loop * loop_rate + w_coh * incoherence

  is_drifting when drift_score >= threshold AND window is full.

All three signals are already available in the run loop (progress from info,
action from the model output, reasoning from the parsed response).
"""

import re
from collections import deque

# ---------------------------------------------------------------------------
# Reasoning-coherence helpers
# ---------------------------------------------------------------------------

# Vague / aimless language that suggests the agent has lost direction
_VAGUE_RE = re.compile(
    r'\b(?:explore|wander|random(?:ly)?|maybe|perhaps|not sure|unclear|'
    r'try (?:different|various|random)|look around|see what happens|'
    r'something|anything|somewhere|just try|no idea|might as well|'
    r'not sure what|unsure|I guess|let me just)\b',
    re.I,
)

# Goal-directed / intentional language
_SPECIFIC_RE = re.compile(
    r'\b(?:need to|must|should|will|going to|plan to|intend to|trying to)\s+\w+|'
    r'\b(?:take|get|open|close|drop|read|examine|enter|climb|push|pull|use|unlock|'
    r'attack|kill|put|insert|turn|light|extinguish)\b|'
    r'\b(?:because|in order to|so that|to (?:reach|find|open|get|use|unlock|access|escape))\b',
    re.I,
)


def _reasoning_incoherence(reasoning: str) -> float:
    """Return incoherence in [0, 1]: 0 = goal-directed, 1 = vague/aimless."""
    if not reasoning or not reasoning.strip():
        return 0.5
    vague    = len(_VAGUE_RE.findall(reasoning))
    specific = len(_SPECIFIC_RE.findall(reasoning))
    total = vague + specific
    if total == 0:
        return 0.3   # reasoning present but no clear signal — mild
    return vague / total


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------

class DriftDetector:
    """Composite drift detector using three behavioral signals.

    Parameters
    ----------
    window : int
        Rolling window of recent steps to examine. Default 20.
    threshold : float
        Composite drift_score at or above which drift is flagged. Default 0.65.
    w_stagnation : float
        Weight for the progress-stagnation component. Default 0.5.
    w_loop : float
        Weight for the behavioral-loop component. Default 0.3.
    w_coherence : float
        Weight for the reasoning-incoherence component. Default 0.2.

    Notes
    -----
    Weights should sum to 1.0.  They can be tuned in config.yaml under
    ``experiments.drift.w_stagnation``, ``w_loop``, ``w_coherence``.
    """

    def __init__(self, window: int = 20, threshold: float = 0.65,
                 w_stagnation: float = 0.5,
                 w_loop: float = 0.3,
                 w_coherence: float = 0.2):
        self.window      = window
        self.threshold   = threshold
        self.w_stagnation = w_stagnation
        self.w_loop       = w_loop
        self.w_coherence  = w_coherence

        self._actions:     deque[str]   = deque(maxlen=window)
        self._progress:    deque[float] = deque(maxlen=window)
        self._incoherence: deque[float] = deque(maxlen=window)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def add(self, action: str, progress: float, reasoning: str) -> None:
        """Record one agent step.

        Parameters
        ----------
        action : str
            The action the agent issued this step.
        progress : float
            game_progress value returned by the env after the step.
        reasoning : str
            The agent's reasoning text for this step.
        """
        self._actions.append((action or '').lower().strip())
        self._progress.append(float(progress))
        self._incoherence.append(_reasoning_incoherence(reasoning))

    # ------------------------------------------------------------------
    # Component scores (each in [0, 1])
    # ------------------------------------------------------------------

    def stagnation_score(self) -> float:
        """Fraction of window steps with no progress increase (trailing run)."""
        prog = list(self._progress)
        if len(prog) < 2:
            return 0.0
        steps_flat = 0
        for i in range(len(prog) - 1, 0, -1):
            if prog[i] > prog[i - 1]:
                break
            steps_flat += 1
        return min(1.0, steps_flat / len(prog))

    def loop_score(self) -> float:
        """1 - (unique actions / total actions) — high means repetitive."""
        actions = list(self._actions)
        if not actions:
            return 0.0
        return 1.0 - len(set(actions)) / len(actions)

    def incoherence_score(self) -> float:
        """Rolling mean of per-step reasoning incoherence."""
        inc = list(self._incoherence)
        return sum(inc) / len(inc) if inc else 0.0

    def drift_score(self) -> float:
        """Weighted composite of the three component scores."""
        return (self.w_stagnation * self.stagnation_score()
                + self.w_loop      * self.loop_score()
                + self.w_coherence * self.incoherence_score())

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def check_drift(self) -> tuple[bool, dict]:
        """Check whether the agent is drifting.

        Returns
        -------
        (is_drifting, details)
            is_drifting : bool
                True when drift_score >= threshold and the window is full.
            details : dict
                drift_score, stagnation, loop_rate, incoherence, window,
                threshold.  Adds 'reason' key when not flagging.
        """
        if len(self._actions) < self.window:
            return (False, {
                'reason':          'insufficient_history',
                'steps_collected': len(self._actions),
                'window_required': self.window,
            })

        stag  = self.stagnation_score()
        loop  = self.loop_score()
        inc   = self.incoherence_score()
        score = self.w_stagnation * stag + self.w_loop * loop + self.w_coherence * inc
        is_drifting = score >= self.threshold

        details: dict = {
            'drift_score':  round(score, 3),
            'stagnation':   round(stag, 3),
            'loop_rate':    round(loop, 3),
            'incoherence':  round(inc, 3),
            'window':       self.window,
            'threshold':    self.threshold,
        }
        if not is_drifting:
            details['reason'] = 'below_threshold'

        return (is_drifting, details)
