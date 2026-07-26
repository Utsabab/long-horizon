"""M6: Planning / synthesis mechanism (research agenda §2.2).

Integration channel (primary): the agent's individual beliefs are correct and
it still holds the goal, yet it never combines the known pieces into the
solution (e.g. it has the lamp, the key, and the spell, but never assembles
them into the winning sequence).

M6 is triggered two ways (mechanisms.adaptive.AdaptiveComputeController):
  - primary: channels.integration.IntegrationDetector confirms a combination
    gap via its LLM judge, which also proposes a `suggested_combination` hint.
  - secondary: sustained high drift risk alone (no confirmed integration gap),
    the many-to-many path where M6 also helps recover from drift.

Either way, PlanningMechanism asks the model for a short refreshed subgoal
given the current goal/memory/state, and writes it back into memory so the
next turn's scratchpad reflects the new plan.
"""

import json

_PROMPT_TEMPLATE = (
    'You are helping a game-playing agent re-plan its immediate next step.\n'
    'Goal: {goal}\n'
    'Current subgoal (may be stale): {subgoal}\n'
    'Known state: {state}\n'
    'Recent actions (oldest first): {recent_history}\n'
    '{hint_line}'
    '\nThe agent appears stuck: it may hold everything it needs but is not '
    'combining known pieces, or its reasoning has drifted from the goal. '
    'Propose ONE short, concrete new subgoal (a single sentence, e.g. '
    '"unlock the grate with the key then go down") that would move the '
    'agent toward the goal.\n'
    'Return ONLY a JSON object with this schema: '
    '{{"subgoal": "<short new subgoal>"}}.'
)


class PlanningMechanism:
    """M6: asks the LLM for a refreshed subgoal and updates memory with it."""

    def __init__(self, window=10):
        self.window = window

    def replan(self, goal, memory, state, llm_client, suggested_combination=None):
        """Produce a refreshed subgoal and write it into memory.

        Args:
            goal: current high-level goal string.
            memory: mechanisms.memory.MemoryMechanism instance (read for
                context, and updated with the new subgoal on success).
            state: dict from mechanisms.state_ext.StateExternalizer.probe()
                (location/inventory), used to ground the prompt.
            llm_client: an OpenRouterClient instance.
            suggested_combination: optional hint string from
                channels.integration.LLMIntegrationJudge.

        Returns:
            (new_subgoal: str|None, tokens: int)
            new_subgoal is None if the call failed or produced nothing usable
            (memory is left unchanged in that case).
        """
        recent = memory.recent_actions[-self.window:] if memory else []
        state_desc = ''
        if state:
            loc = state.get('location') or 'unknown'
            items = ', '.join(state.get('inventory_items') or []) or 'nothing'
            state_desc = f'at "{loc}", carrying: {items}'
        hint_line = f'Hint: consider combining: {suggested_combination}\n' if suggested_combination else ''

        prompt = _PROMPT_TEMPLATE.format(
            goal=goal or 'complete the game',
            subgoal=(memory.subgoal if memory else None) or 'none set',
            state=state_desc or 'unknown',
            recent_history='\n'.join(recent) if recent else 'none',
            hint_line=hint_line,
        )

        try:
            resp = llm_client.generate(prompt, max_tokens=96, reasoning_enabled=False)
            parsed = llm_client.parse_completion(resp)
        except Exception:
            return None, 0

        content = parsed.get('content', '') if isinstance(parsed, dict) else str(parsed)
        tokens = parsed.get('tokens', 0) if isinstance(parsed, dict) else 0

        new_subgoal = None
        start, end = content.find('{'), content.rfind('}')
        if 0 <= start < end:
            try:
                obj = json.loads(content[start:end + 1])
                candidate = str(obj.get('subgoal') or '').strip()
                if candidate:
                    new_subgoal = candidate
            except Exception:
                pass

        if new_subgoal and memory is not None:
            memory.subgoal = new_subgoal

        return new_subgoal, tokens
