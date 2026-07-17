"""M2: Memory / context mechanism (research agenda §2.2).

Drift channel: the agent loses its goal or current subgoal.
M2 targets drift by maintaining a structured scratchpad — an explicit record
of what has been seen and tried — that replaces raw scrolling history in the
prompt.

How it works
------------
``update(obs_before, obs_after, action, reasoning)`` is called after every
env.step().  It updates the scratchpad heuristically using regex patterns on
the observation and reasoning text — no extra LLM call required.

``render()`` returns a compact, structured string that the runner injects into
the system prompt before asking the model for its next action.  With the
scratchpad visible, the model always sees its current goal, what it has tried
here, and what it has discovered — making it much harder to drift invisibly.

On / Off switch
-----------------------------------------
  Off (baseline): runner calls build_step_messages with no scratchpad.
  On  (M2):       runner calls build_step_messages(scratchpad=memory.render()).

Targets channel: control-path drift (§2.1).
"""

import re
from collections import defaultdict

# ---------------------------------------------------------------------------
# Heuristic pattern constants
# ---------------------------------------------------------------------------

# Parser rejection phrases — when the observation matches, the action failed.
_FAILURE_RE = re.compile(
    r"(?:you can'?t|that'?s not|i don'?t (?:understand|know the word|see any)|"
    r"nothing happens|impossible|it is already|there is no|you don'?t see|"
    r"you cannot|that doesn'?t|invalid|no verb in|i don'?t know what)",
    re.I,
)

# Discovery phrases — "There is a X", "You see a X", etc.
_DISCOVERY_RE = re.compile(
    r'(?:there (?:is|are)|you (?:see|notice|find|spot)|'
    r'lying here|sitting here|resting here|is here)\s+'
    r'(?:a |an |the |some )?(?P<item>[A-Za-z][A-Za-z0-9\s\'\-]{2,35})',
    re.I,
)

# Subgoal intent phrases in the agent's reasoning.
_SUBGOAL_RE = re.compile(
    r'(?:I need to|I should|I must|my (?:next )?(?:goal|plan|task) is|'
    r"I(?:'m| am) (?:trying|going|planning) to|I want to|I will|let me)\s+"
    r'(?P<intent>[A-Za-z].{5,80}?)(?:\.|,|;|$)',
    re.I,
)

_MAX_TRIED = 10     # max tried-actions to show per location
_MAX_DISC  = 10     # max discoveries to retain
_MAX_VISIT = 25     # max visited locations to show


# ---------------------------------------------------------------------------
# MemoryMechanism
# ---------------------------------------------------------------------------

class MemoryMechanism:
    """Structured scratchpad (M2) for the memory / context harness mechanism.

    Parameters
    ----------
    goal : str, optional
        High-level goal statement.  If not supplied, defaults to "complete the
        game" until the runner sets it from the first observation.
    max_visited : int
        Maximum number of visited locations to show in render().
    max_discoveries : int
        Maximum number of key discoveries to retain.
    """

    def __init__(self, goal: str | None = None,
                 max_visited: int = _MAX_VISIT,
                 max_discoveries: int = _MAX_DISC):
        self.goal: str = goal or 'complete the game'
        self.subgoal: str | None = None
        self.max_visited = max_visited
        self.max_discoveries = max_discoveries

        self._current_location: str | None = None
        self._visited: list[str] = []           # ordered, deduplicated
        self._tried:  dict[str, list[str]] = defaultdict(list)   # loc → actions
        self._failed: dict[str, list[str]] = defaultdict(list)   # loc → actions
        self._discoveries: list[str] = []
        self._disc_seen: set[str] = set()       # dedup key

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_location(self, obs: str) -> str | None:
        """First non-empty, non-boilerplate line is usually the room name."""
        skip_prefixes = ('ZORK', 'Infocom', 'Copyright', 'Release', '>')
        for line in obs.splitlines():
            line = line.strip()
            if line and not any(line.startswith(p) for p in skip_prefixes):
                # Room names are typically short (< 60 chars) and mostly words
                if len(line) < 60 and re.search(r'[A-Za-z]', line):
                    return line
        return None

    def _extract_subgoal(self, reasoning: str) -> str | None:
        m = _SUBGOAL_RE.search(reasoning or '')
        if m:
            intent = m.group('intent').strip().rstrip('.,;')
            if 10 <= len(intent) <= 80:
                return intent
        return None

    def _extract_discoveries(self, obs: str) -> list[str]:
        found = []
        for m in _DISCOVERY_RE.finditer(obs):
            item = m.group('item').strip().rstrip('.,;').lower()
            if item and item not in self._disc_seen and len(item) >= 3:
                found.append(item)
                self._disc_seen.add(item)
        return found

    def _action_failed(self, obs: str) -> bool:
        return bool(_FAILURE_RE.search(obs or ''))

    def set_goal(self, goal: str) -> None:
        """Override the high-level goal (e.g. extracted from first observation)."""
        self.goal = goal

    def update(self, obs_before: str, obs_after: str, action: str,
               reasoning: str) -> None:
        """Update memory after one env.step().

        Parameters
        ----------
        obs_before : str
            Observation the agent saw BEFORE taking the action (the context
            in which the action was chosen — used for tried/failed tracking).
        obs_after : str
            Observation returned by env.step() (where the agent is now).
        action : str
            The action the agent took.
        reasoning : str
            The agent's reasoning text for this step.
        """
        # --- Current location (where we landed) ---
        new_loc = self._extract_location(obs_after)
        if new_loc:
            self._current_location = new_loc
            if new_loc not in self._visited:
                self._visited.append(new_loc)

        # --- Tried / failed actions (at the location we were in before) ---
        prev_loc = self._extract_location(obs_before) or self._current_location
        if prev_loc and action:
            tried = self._tried[prev_loc]
            if action not in tried:
                tried.append(action)
            if self._action_failed(obs_after):
                failed = self._failed[prev_loc]
                if action not in failed:
                    failed.append(action)

        # --- Key discoveries from the new observation ---
        new_discs = self._extract_discoveries(obs_after)
        self._discoveries.extend(new_discs)
        if len(self._discoveries) > self.max_discoveries:
            self._discoveries = self._discoveries[-self.max_discoveries:]

        # --- Subgoal from reasoning ---
        subgoal = self._extract_subgoal(reasoning)
        if subgoal:
            self.subgoal = subgoal

    def render(self) -> str:
        """Return a compact structured string for prompt injection.

        The returned string is meant to be appended to (or inserted into) the
        system prompt so the model always sees its current context.
        """
        lines = ['=== AGENT MEMORY (M2) ===']

        lines.append(f'Goal:    {self.goal}')
        if self.subgoal:
            lines.append(f'Subgoal: {self.subgoal}')

        if self._current_location:
            lines.append(f'You are: {self._current_location}')

        if self._visited:
            shown = self._visited[-self.max_visited:]
            lines.append(f'Visited ({len(self._visited)} rooms): '
                         + ', '.join(shown))

        if self._current_location:
            tried  = self._tried.get(self._current_location, [])
            failed = self._failed.get(self._current_location, [])
            if tried:
                lines.append('Tried here: ' + ', '.join(tried[-_MAX_TRIED:]))
            if failed:
                lines.append('Failed here: ' + ', '.join(failed[-_MAX_TRIED:]))

        if self._discoveries:
            lines.append('Discoveries:')
            for d in self._discoveries:
                lines.append(f'  • {d}')

        lines.append('=== END MEMORY ===')
        return '\n'.join(lines)
