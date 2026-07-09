"""Accepted-Local-Error (ALE) detector.

ALE (research agenda §2.1): the agent takes something false as true and acts
on it—a wrong belief, a mistaken map edge, or a command that assumes a state
that does not hold.

Detection strategy
------------------
1. check_repetition() – flags when the same action appears k times in a row
   with no progress change (cheap, purely behavioural).
2. extract_belief()   – parses the most recent reasoning text to find the
   belief the agent is acting on.
3. check_ale()        – calls an external env_checker(belief) to test whether
   that belief contradicts the game's true state.

``env_checker`` is a callable provided by the runner; it has access to the
live game state and returns (is_contradiction: bool, evidence: dict).
"""

import re
from collections import deque


class AcceptedLocalErrorDetector:
    """Detect ALE by combining repetition detection with belief verification."""

    # Patterns that extract a structured belief from a reasoning string.
    # Each match is converted into a typed tuple:
    #   ('have_item', item)
    #   ('item_in_loc', item, location)
    #   ('clause', text)
    # "I have ..." is ambiguous between possession ("I have the lamp") and
    # perfect-tense narration of past actions ("I have already tried going
    # east", "I have looked at the poodle"). The narration form is far more
    # common in a stuck agent's reasoning (it's recapping what it already
    # tried), so a have_item match is rejected — not just accepted — when the
    # captured phrase contains one of these recap/movement markers anywhere,
    # rather than only right after "have" (a front-anchored exclusion missed
    # "I have already looked ..." since "already" sits between the two).
    _RECAP_MARKERS = re.compile(
        r'\b(?:already|tried|tries|trying|looked|talked|spoke|spoken|explored|exited|'
        r'examined|searched|walked|moved|returned|arrived|entered|reached|got|gotten|'
        r'gone|come|been|made it|checked|visited)\b',
        re.I,
    )

    BELIEF_PATTERNS = [
        re.compile(r'\bI (?:have|picked up|picked|took|grabbed) (?:the )?(?P<item>[\w \'\-]+)', re.I),
        re.compile(r'\bI (?:dropped|left) (?:the )?(?P<item>[\w \'\-]+)', re.I),
        re.compile(r'\b(?P<item>[\w \'\-]+) is in the (?P<loc>[\w \'\-]+)', re.I),
        re.compile(r'\bIt (?:\'s| is) in the (?P<loc>[\w \'\-]+)', re.I),
        re.compile(r'\bI (?:remember|know) that (?P<clause>.+)', re.I),
    ]

    def __init__(self, k=3, window=10):
        """
        Args:
            k: number of consecutive identical actions that triggers repetition.
            window: sliding-window size for action/progress/reasoning history.
        """
        self.k = k
        self.window = window
        self.actions = deque(maxlen=window)
        self.progress = deque(maxlen=window)
        self.reasonings = deque(maxlen=window)

    def add(self, action, progress, reasoning=None):
        """Record one game step."""
        self.actions.append(action)
        self.progress.append(progress)
        self.reasonings.append(reasoning or '')

    def check_repetition(self):
        """Return True if the last k actions are identical with no progress."""
        if len(self.actions) < self.k:
            return False
        last_k = list(self.actions)[-self.k:]
        if not all(a == last_k[0] for a in last_k):
            return False
        return len(self.progress) >= 2 and max(self.progress) == min(self.progress)

    def extract_belief(self, reasoning_text):
        """Parse a typed belief tuple from a reasoning string, or return None."""
        if not reasoning_text:
            return None
        for pattern in self.BELIEF_PATTERNS:
            m = pattern.search(reasoning_text)
            if not m:
                continue
            gd = m.groupdict()
            # Check 'loc' before 'item': the "X is in the Y" pattern captures both
            # groups when it matches, so item_in_loc must win over have_item for it
            # (checking 'item' first would misclassify every location belief).
            if gd.get('loc'):
                return ('item_in_loc', (gd.get('item') or '').strip() or None, gd['loc'].strip())
            if gd.get('item'):
                if self._RECAP_MARKERS.search(gd['item']):
                    continue  # narration of a past action, not a possession claim
                return ('have_item', gd['item'].strip())
            if gd.get('clause'):
                return ('clause', gd['clause'].strip())
        return None

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

        for reasoning in reversed(self.reasonings):
            belief = self.extract_belief(reasoning)
            if belief:
                contradiction, evidence = env_checker(belief)
                return (contradiction, {'belief': belief, 'contradiction': contradiction, 'evidence': evidence})

        return (False, {'reason': 'no_belief_detected'})
