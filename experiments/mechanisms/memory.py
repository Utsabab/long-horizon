"""M2: Memory / context mechanism (research agenda §2.2).

Drift channel: the agent loses its goal or current subgoal.
M2 targets drift by maintaining a structured scratchpad — an explicit record
of what has been seen and tried — that replaces raw scrolling history in the
prompt.

How it works
------------
``update(...)`` is called after every env.step(). It no longer scrapes
observation/reasoning text with regex to guess location, subgoal, or
discoveries — every field is handed in already structured:

  - location / inventory_items come from mechanisms.state_ext.StateExternalizer
    (M3), a read-only ground-truth probe of the live environment.
  - goal / subgoal come from the model's own structured self-report each turn
    (agent.SYSTEM_PROMPT_TEMPLATE / llm.extract_step_fields).
  - action_failed is a structural signal the runner computes by comparing
    the M3 probe before and after the step (state unchanged), not by
    pattern-matching rejection phrases in the observation text.

``render()`` returns a compact, structured string that the runner injects into
the system prompt before asking the model for its next action. With the
scratchpad visible, the model always sees its current goal, what it has tried
here, and what it has discovered — making it much harder to drift invisibly.

On / Off switch
-----------------------------------------
  Off (baseline): runner calls build_step_messages with no scratchpad.
  On  (M2):       runner calls build_step_messages(scratchpad=memory.render()).

Targets channel: control-path drift (§2.1).
"""

from collections import defaultdict

_MAX_TRIED = 10     # max tried-actions to show per location
_MAX_ITEMS = 10     # max inventory items to retain
_MAX_VISIT = 25     # max visited locations to show


class MemoryMechanism:
    """Structured scratchpad (M2) for the memory / context harness mechanism.

    Parameters
    ----------
    goal : str, optional
        High-level goal statement.  If not supplied, defaults to "complete the
        game" until a step reports one.
    max_visited : int
        Maximum number of visited locations to show in render().
    max_discoveries : int
        Maximum number of inventory items to retain (kept as
        ``max_discoveries`` for config-key backward compatibility).
    """

    def __init__(self, goal: str | None = None,
                 max_visited: int = _MAX_VISIT,
                 max_discoveries: int = _MAX_ITEMS):
        self.goal: str = goal or 'complete the game'
        self.subgoal: str | None = None
        self.max_visited = max_visited
        self.max_items = max_discoveries

        self._current_location: str | None = None
        self._visited: list[str] = []           # ordered, deduplicated
        self._tried:  dict[str, list[str]] = defaultdict(list)   # loc → actions
        self._failed: dict[str, list[str]] = defaultdict(list)   # loc → actions
        self._inventory_items: list[str] = []

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def set_goal(self, goal: str) -> None:
        """Override the high-level goal (e.g. seeded before the first step)."""
        self.goal = goal

    @property
    def inventory_items(self) -> list[str]:
        """Most recent M3-verified inventory, for IntegrationDetector."""
        return list(self._inventory_items)

    @property
    def recent_actions(self) -> list[str]:
        """Flattened recent tried-actions across all locations, most-recent last."""
        actions = []
        for loc in self._visited:
            actions.extend(self._tried.get(loc, []))
        return actions

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, location: str | None, inventory_items: list[str] | None,
               action: str, action_failed: bool,
               goal: str | None = None, subgoal: str | None = None) -> None:
        """Update memory after one env.step(), fed structured fields directly.

        Parameters
        ----------
        location : str or None
            Ground-truth room name from mechanisms.state_ext (M3), i.e. where
            the agent is now.
        inventory_items : list[str] or None
            Ground-truth inventory from mechanisms.state_ext (M3).
        action : str
            The action the agent took this step.
        action_failed : bool
            Structural signal (computed by the runner from before/after M3
            state) indicating the action had no observable effect.
        goal : str, optional
            Self-reported goal for this step; updates self.goal if given.
        subgoal : str, optional
            Self-reported subgoal for this step; updates self.subgoal if given.
        """
        if goal:
            self.goal = goal
        if subgoal:
            self.subgoal = subgoal

        prev_location = self._current_location

        if location:
            self._current_location = location
            if location not in self._visited:
                self._visited.append(location)

        tracking_loc = prev_location or self._current_location
        if tracking_loc and action:
            tried = self._tried[tracking_loc]
            if action not in tried:
                tried.append(action)
            if action_failed:
                failed = self._failed[tracking_loc]
                if action not in failed:
                    failed.append(action)

        if inventory_items is not None:
            self._inventory_items = list(inventory_items)[-self.max_items:]

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

        if self._inventory_items:
            lines.append('Carrying: ' + ', '.join(self._inventory_items))

        lines.append('=== END MEMORY ===')
        return '\n'.join(lines)
