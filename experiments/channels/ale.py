"""Accepted-Local-Error (ALE) detector.

ALE (research agenda §2.1): the agent takes something false as true and acts
on it—a wrong belief, a mistaken map edge, or a command that assumes a state
that does not hold.

Detection strategy
------------------
1. check_repetition() – flags when the same action appears k times in a row
   with no progress change (cheap, purely behavioural).
2. check_ale()        – given the belief the model self-reported this step
   (see agent.SYSTEM_PROMPT_TEMPLATE / llm.extract_step_fields), calls an
   external env_checker(belief) to test whether that belief contradicts the
   game's true state.

The belief is no longer regex-mined from reasoning prose: the model reports
it directly as a structured field each turn, which is far more robust than
pattern-matching "I have X" / "X is in Y" out of free text.

``env_checker`` is a callable provided by the runner; it has access to the
live game state and returns (is_contradiction: bool, evidence: dict).
"""

from collections import deque


def _belief_to_tuple(belief):
    """Convert a structured {type, item, location} belief dict to the typed
    tuple env_checker.make_env_checker() expects, or None if unusable.

      ('have_item', item)
      ('item_in_loc', item_or_None, location)
    """
    if not belief or not isinstance(belief, dict):
        return None
    btype = belief.get('type')
    if btype == 'have_item' and belief.get('item'):
        return ('have_item', belief['item'])
    if btype == 'item_in_loc' and belief.get('location'):
        return ('item_in_loc', belief.get('item'), belief['location'])
    return None


class AcceptedLocalErrorDetector:
    """Detect ALE by combining repetition detection with belief verification."""

    def __init__(self, k=3, window=10):
        """
        Args:
            k: number of consecutive identical actions that triggers repetition.
            window: sliding-window size for action/progress/belief history.
        """
        self.k = k
        self.window = window
        self.actions = deque(maxlen=window)
        self.progress = deque(maxlen=window)
        self.beliefs = deque(maxlen=window)

    def add(self, action, progress, belief=None):
        """Record one game step.

        Args:
            action: the action taken this step.
            progress: game_progress after the step.
            belief: structured belief dict self-reported by the model this
                    step ({type, item, location}), or None.
        """
        self.actions.append(action)
        self.progress.append(progress)
        self.beliefs.append(belief)

    def check_repetition(self):
        """Return True if the last k actions are identical with no progress."""
        if len(self.actions) < self.k:
            return False
        last_k = list(self.actions)[-self.k:]
        if not all(a == last_k[0] for a in last_k):
            return False
        return len(self.progress) >= 2 and max(self.progress) == min(self.progress)

    def check_ale(self, env_checker):
        """Check whether the current situation is an ALE.

        Args:
            env_checker: callable(belief) -> (is_contradiction: bool, evidence: dict)
                         The runner supplies this; it has access to the live game state.

        Returns:
            (is_ale: bool, details: dict)
        """
        if not self.check_repetition():
            return (False, {'reason': 'no_repetition'})

        for belief in reversed(self.beliefs):
            belief_tuple = _belief_to_tuple(belief)
            if belief_tuple:
                contradiction, evidence = env_checker(belief_tuple)
                return (contradiction, {'belief': belief_tuple, 'contradiction': contradiction, 'evidence': evidence})

        return (False, {'reason': 'no_belief_detected'})
