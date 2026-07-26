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
       How goal-directed the agent's reasoning still is, given its stated
       goal/subgoal and recent history. Scored by LLMCoherenceJudge — a
       small LLM call made every step — rather than a regex heuristic,
       since "is this reasoning still goal-directed" is a semantic judgment
       that pattern-matching vague/specific vocabulary approximates poorly.

Composite
---------
  drift_score = w_stag * stagnation + w_loop * loop_rate + w_coh * incoherence

  is_drifting when drift_score >= threshold AND window is full.

All three signals are already available in the run loop (progress from info,
action from the model output, incoherence from LLMCoherenceJudge.score()).
"""

import json
import re
from collections import deque

# ---------------------------------------------------------------------------
# LLM-based coherence judge
# ---------------------------------------------------------------------------

class LLMCoherenceJudge:
    """Score reasoning incoherence via a small LLM call (replaces regex).

    Same call pattern as mechanisms.info_seeking.InfoSeekingMechanism: a
    cheap, low-max-tokens prompt, run every step (confirmed cost-acceptable),
    asking whether the agent's current reasoning is still goal-directed given
    its self-reported goal/subgoal and recent action history.
    """

    _PROMPT_TEMPLATE = (
        "You are scoring whether a game-playing agent's reasoning is still "
        'goal-directed.\n'
        'Goal: {goal}\n'
        'Subgoal: {subgoal}\n'
        'Recent actions (oldest first): {recent_history}\n'
        'Current reasoning: {reasoning}\n\n'
        'On a scale from 0.0 (fully coherent and goal-directed) to 1.0 '
        '(vague, aimless, or unrelated to the stated goal/subgoal), how '
        'incoherent is this reasoning?\n'
        'Return ONLY a JSON object with one key: {{"incoherence": <float 0.0-1.0>}}.'
    )

    def score(self, goal, subgoal, recent_history, reasoning, llm_client):
        """Return (incoherence: float in [0,1], tokens: int) for this step.

        Falls back to a neutral 0.5 (with 0 tokens charged) if reasoning is
        empty, the LLM call fails, or the response can't be parsed — a judge
        failure should read as "unknown", not silently as "fully coherent".
        """
        if not reasoning or not reasoning.strip():
            return 0.5, 0

        prompt = self._PROMPT_TEMPLATE.format(
            goal=goal or 'unknown',
            subgoal=subgoal or 'unknown',
            recent_history='; '.join(recent_history[-5:]) if recent_history else '(none)',
            reasoning=reasoning,
        )
        try:
            resp = llm_client.generate(prompt, max_tokens=32, reasoning_enabled=False)
            parsed = llm_client.parse_completion(resp)
        except Exception:
            return 0.5, 0

        content = parsed.get('content', '') if isinstance(parsed, dict) else str(parsed)
        tokens = parsed.get('tokens', 0) if isinstance(parsed, dict) else 0

        start, end = content.find('{'), content.rfind('}')
        if 0 <= start < end:
            try:
                obj = json.loads(content[start:end + 1])
                value = float(obj.get('incoherence'))
                return max(0.0, min(1.0, value)), tokens
            except Exception:
                pass

        match = re.search(r'0?\.\d+|\d(?:\.\d+)?', content)
        if match:
            try:
                value = float(match.group(0))
                return max(0.0, min(1.0, value)), tokens
            except Exception:
                pass

        return 0.5, tokens


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

    def add(self, action: str, progress: float, incoherence_score: float) -> None:
        """Record one agent step.

        Parameters
        ----------
        action : str
            The action the agent issued this step.
        progress : float
            game_progress value returned by the env after the step.
        incoherence_score : float
            Pre-computed incoherence in [0, 1] for this step's reasoning,
            from LLMCoherenceJudge.score() (or any equivalent signal).
        """
        self._actions.append((action or '').lower().strip())
        self._progress.append(float(progress))
        self._incoherence.append(float(incoherence_score))

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
