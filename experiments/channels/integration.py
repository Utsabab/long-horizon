"""Integration (synthesis) detector (research agenda §2.1 — integration channel).

Integration: the agent's individual beliefs are correct and it still holds
the goal, yet it never combines the known pieces into the solution. Example:
it has found the lamp, the key, and the spell, and knows what it is trying
to do, but never assembles them into the winning sequence. Distinct from
drift (the goal is held) and from Accepted-Local-Error (the beliefs are
right); this is M6 (Planning/synthesis)'s primary target.

Two-stage detection
--------------------
1. precheck() — cheap, no LLM/regex: plain attribute checks over M2 memory
   state (discoveries, inventory) and recent action history. Does the agent
   hold >= min_unused known items/discoveries that haven't been referenced
   in any of its recent actions? If not, there is nothing worth spending an
   LLM call to investigate.
2. LLMIntegrationJudge.check() — an LLM call, run only when precheck() trips
   — asks whether those held-but-unused pieces should already have been
   combined given the agent's current subgoal, and if so, what combination
   is missing.
"""

import json


class IntegrationDetector:
    """Detect 'known pieces never combined' via a cheap pre-filter."""

    def __init__(self, min_unused=2, recent_window=10):
        """
        Args:
            min_unused: minimum number of unused discoveries/inventory items
                        that must be present before the (costly) LLM judge is
                        worth running.
            recent_window: how many recent actions count as "recent" when
                        checking whether an item has been referenced.
        """
        self.min_unused = min_unused
        self.recent_window = recent_window

    def precheck(self, discoveries, inventory_items, recent_actions):
        """Cheap attribute pre-filter.

        Returns:
            (tripped: bool, unused: list[str]) — unused items/discoveries
            that no recent action mentions (case-insensitive substring
            match).
        """
        candidates = list(dict.fromkeys((discoveries or []) + (inventory_items or [])))
        recent = [a.lower() for a in (recent_actions or [])[-self.recent_window:]]
        unused = [
            item for item in candidates
            if item and not any(item.lower() in action for action in recent)
        ]
        return len(unused) >= self.min_unused, unused


class LLMIntegrationJudge:
    """LLM call (run only when IntegrationDetector.precheck() trips) that asks
    whether the agent should already be combining its known-but-unused pieces.
    """

    _PROMPT_TEMPLATE = (
        'You are checking whether a game-playing agent is failing to combine '
        'pieces it already knows about into a solution.\n'
        'Current subgoal: {subgoal}\n'
        'Known but seemingly unused discoveries/items: {unused}\n'
        'Recent actions (oldest first): {recent_history}\n\n'
        'Given the subgoal, should the agent already be combining some of '
        'these known pieces (e.g. using an item on another, combining two '
        'discoveries) instead of continuing as it has been? '
        'Return ONLY a JSON object with this schema: '
        '{{"gap": true|false, "suggested_combination": "<short suggestion or empty string>"}}.'
    )

    def check(self, subgoal, unused, recent_actions, llm_client):
        """Return (result: {'gap': bool, 'suggested_combination': str|None}, tokens: int)."""
        prompt = self._PROMPT_TEMPLATE.format(
            subgoal=subgoal or 'unknown',
            unused=', '.join(unused) if unused else '(none)',
            recent_history='; '.join((recent_actions or [])[-10:]) or '(none)',
        )
        try:
            resp = llm_client.generate(prompt, max_tokens=96, reasoning_enabled=False)
            parsed = llm_client.parse_completion(resp)
        except Exception:
            return {'gap': False, 'suggested_combination': None}, 0

        content = parsed.get('content', '') if isinstance(parsed, dict) else str(parsed)
        tokens = parsed.get('tokens', 0) if isinstance(parsed, dict) else 0

        start, end = content.find('{'), content.rfind('}')
        if 0 <= start < end:
            try:
                obj = json.loads(content[start:end + 1])
                gap = bool(obj.get('gap'))
                suggestion = str(obj.get('suggested_combination') or '').strip() or None
                return {'gap': gap, 'suggested_combination': suggestion}, tokens
            except Exception:
                pass

        return {'gap': False, 'suggested_combination': None}, tokens
