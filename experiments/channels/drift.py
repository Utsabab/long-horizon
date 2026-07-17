"""Control-Path Drift detector (research agenda §2.1 — drift channel).

Drift: the agent loses the goal or current subgoal and drifts into collecting
unrelated objects or exploring without purpose.

Detection strategy — oracle path distance
-----------------------------------------
Each game has a known walkthrough: the exact sequence of parser commands that
complete the game. We use that sequence as an oracle. An agent action that
appears in the walkthrough set is "on-path"; one that does not is "off-path".

  off_path_rate()     — fraction of the last `window` actions not in the
                        walkthrough set.  1.0 = fully drifted, 0.0 = on-path.

  steps_since_on_path() — how many consecutive recent steps were all off-path.

  check_drift()       — returns (is_drifting, details) where is_drifting is
                        True when off_path_rate >= threshold.

Caveat: "off-path" ≠ "wrong". Text adventures allow
multiple valid routes; the walkthrough is one solution. The drift score is a
cheap behavioural proxy, not a hard label. An LLM judge is needed for
definitive drift labelling.
"""

import re
from collections import deque
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Direction abbreviation → full word (parser accepts both; walkthrough uses full words)
_DIR_ABBREV = {
    'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
    'ne': 'northeast', 'nw': 'northwest', 'se': 'southeast', 'sw': 'southwest',
    'u': 'up', 'd': 'down',
}

# Leading movement verb that the walkthrough omits
_GO_PREFIX_RE = re.compile(r'^go\s+')

# Verb synonyms: map LLM variants → walkthrough canonical form
_VERB_MAP = {
    'pick up': 'take', 'grab': 'take', 'get': 'take',
    'examine': 'look at', 'x ': 'look at ', 'inspect': 'look at',
    'put down': 'drop', 'place': 'put',
}


def _normalize(action: str) -> str:
    """Lowercase, strip, collapse whitespace, then canonicalize common variants.

    Handles:
    - Direction abbreviations (n → north, u → up, …)
    - 'go north' → 'north'
    - Verb synonyms (pick up → take, examine → look at, …)
    """
    a = re.sub(r'\s+', ' ', (action or '').lower().strip())
    # Direction abbreviation as the whole command
    if a in _DIR_ABBREV:
        return _DIR_ABBREV[a]
    # Strip leading "go " (walkthrough uses bare direction words)
    a = _GO_PREFIX_RE.sub('', a)
    # If now just an abbreviation, expand it
    if a in _DIR_ABBREV:
        return _DIR_ABBREV[a]
    # Verb synonym substitution
    for variant, canonical in _VERB_MAP.items():
        if a.startswith(variant):
            a = canonical + a[len(variant):]
            break
    return a


def load_walkthrough(path) -> list[str]:
    """Return the list of normalized walkthrough actions.

    Raises FileNotFoundError if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f'Walkthrough file not found: {path}\n'
            'Make sure the textquests submodule is initialised: '
            'git submodule update --init'
        )
    with open(path, encoding='utf-8') as f:
        lines = f.read().splitlines()
    return [_normalize(line) for line in lines if line.strip()]

class DriftDetector:
    """Detect control-path drift via oracle path distance.

    Parameters
    ----------
    walkthrough_path : Path-like
        Path to ``<game>_walkthrough.txt`` inside the textquests data folder.
    window : int
        Number of recent agent actions to examine.  Default 20.
    threshold : float
        Off-path fraction in [0, 1] at or above which drift is flagged.
        Default 0.8 means 80 % of the window must be off-path.
    """

    def __init__(self, walkthrough_path, window: int = 20, threshold: float = 0.8):
        wt = load_walkthrough(walkthrough_path)
        self._wt_set: set[str] = set(wt)
        self._wt_len: int = len(wt)
        self.window = window
        self.threshold = threshold
        self._history: deque[str] = deque(maxlen=window)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def walkthrough_length(self) -> int:
        """Number of actions in the full walkthrough."""
        return self._wt_len

    @property
    def history(self) -> list[str]:
        return list(self._history)

    def add(self, action: str) -> None:
        """Record one agent action (call after every env.step)."""
        self._history.append(_normalize(action))

    def is_on_path(self, action: str) -> bool:
        """True if the (normalized) action appears anywhere in the walkthrough."""
        return _normalize(action) in self._wt_set

    def off_path_rate(self) -> float:
        """Fraction of the current window that are off-path actions."""
        if not self._history:
            return 0.0
        off = sum(1 for a in self._history if a not in self._wt_set)
        return off / len(self._history)

    def steps_since_on_path(self) -> int:
        """Number of consecutive tail steps that are all off-path."""
        count = 0
        for action in reversed(self._history):
            if action in self._wt_set:
                break
            count += 1
        return count

    def check_drift(self) -> tuple[bool, dict]:
        """Check whether the agent is drifting.

        Returns
        -------
        (is_drifting, details)
            is_drifting : bool
                True when ``off_path_rate >= threshold`` and the window is full.
            details : dict
                off_path_rate, steps_since_on_path, window, threshold,
                recent_actions (list), and a 'reason' key when not drifting.
        """
        if len(self._history) < self.window:
            return (False, {
                'reason': 'insufficient_history',
                'steps_collected': len(self._history),
                'window_required': self.window,
            })

        rate = self.off_path_rate()
        consecutive = self.steps_since_on_path()
        is_drifting = rate >= self.threshold

        details: dict = {
            'off_path_rate': round(rate, 3),
            'steps_since_on_path': consecutive,
            'window': self.window,
            'threshold': self.threshold,
            'recent_actions': list(self._history),
        }
        if not is_drifting:
            details['reason'] = 'below_threshold'

        return (is_drifting, details)
