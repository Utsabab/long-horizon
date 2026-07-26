"""M1: Information-seeking mechanism (lead mechanism).

Research agenda §2.2 / §6:
  Off: agent acts directly on its beliefs.
  On (Prompted): agent is told the belief the harness just confirmed false,
                 and the verified true state, then asked to act on the truth.
  On (Enforced): harness will not let the agent act on an unchecked belief.

This module implements the Prompted setting. channels.env_checker has already
confirmed the agent's belief contradicts the live game state (via an
inventory/look probe); this mechanism hands that confirmed belief + the
verified evidence to the model and asks it to propose one next action
consistent with the truth, not the false assumption.

The Enforced setting (blocking every action, not just ones already flagged by
repetition, until it's pre-verified) is a future extension tracked as a TODO.

Targets channel: Accepted-Local-Error, Control-path drift (§2.1).
"""

from mechanisms.action_templating import normalize_action

_PROMPT_TEMPLATE = (
    'You are an assistant helping an agent playing a text-based game.\n'
    'The agent has been acting on a belief that the game just confirmed is FALSE:\n'
    '  Believed: {believed}\n'
    '  Actual state (just verified in-game): {actual}\n\n'
    'Recent history (oldest first):\n{history}\n\n'
    'Given the corrected information above, propose ONE concrete next parser command '
    '(e.g. EXAMINE DOOR, GO NORTH) that is consistent with the true state, not the false '
    "belief. Do not repeat an action that assumed the false belief.\n"
    'End your response with the proposed command on its own line.'
)


def _describe_belief(belief):
    """Render a channels.ale belief tuple as a short human-readable clause."""
    kind = belief[0] if belief else None
    if kind == 'have_item':
        return f"the agent is holding '{belief[1]}'"
    if kind == 'item_in_loc':
        item, loc = belief[1], belief[2]
        return f"'{item or 'something'}' is in the '{loc}'"
    return str(belief[1]) if len(belief) > 1 else str(belief)


class InfoSeekingMechanism:
    """M1 (Prompted): tells the model its belief is confirmed false and the
    verified true state, then asks it to propose a corrected next action.

    Typical usage (inside a runner's repetition-detection block, after
    channels.ale.AcceptedLocalErrorDetector.check_ale() confirms a contradiction):

        is_ale, details = detector.check_ale(env_checker)
        if is_ale:
            proposed_action, response = info_seeking.intervene(
                format_history(history), details['belief'], details['evidence'],
            )
            if proposed_action:
                obs, reward, done, info = env.step(proposed_action)
    """

    def __init__(self, llm_client, window=10):
        """
        Args:
            llm_client: an OpenRouterClient instance.
            window: how many recent history lines to include in the prompt.
        """
        self.llm = llm_client
        self.window = window

    def intervene(self, history_lines, belief, evidence):
        """Run the info-seeking prompt and extract a proposed game action.

        Args:
            history_lines: list of 'OBS/ACT/REAS' formatted strings
                           (as returned by agent.format_history).
            belief: the belief tuple channels.ale.AcceptedLocalErrorDetector
                    extracted and confirmed false.
            evidence: the evidence dict env_checker returned alongside that
                      confirmation; must contain 'probe_text' (the raw
                      inventory/look response that proved the belief wrong).

        Returns:
            (proposed_action: str, parsed_response: dict)
            proposed_action is empty string if extraction failed.
        """
        recent = history_lines[-self.window:]
        believed = _describe_belief(belief)
        actual = (evidence or {}).get('probe_text', '').strip()
        prompt = _PROMPT_TEMPLATE.format(believed=believed, actual=actual, history='\n'.join(recent))

        # reasoning_enabled=False: this call only needs a short reflection + one
        # action line, not extended chain-of-thought; with thinking enabled the
        # model can burn the whole token budget before emitting an answer.
        resp = self.llm.generate(prompt, max_tokens=384, reasoning_enabled=False)
        parsed = dict(self.llm.parse_completion(resp))

        content = parsed.get('content', '') if isinstance(parsed, dict) else str(parsed)
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        proposed = normalize_action(lines[-1]) if lines else ''

        # Prefix the logged reasoning with the verified fact itself (not just the
        # model's paraphrase of it), so once this step lands in history, the
        # correction persists verbatim for every later turn to see — not just a
        # one-time nudge that can be forgotten or reasoned away.
        verified_fact = f"[Verified] Belief ({believed}) contradicted the game state. Actual state: {actual}"
        parsed['content'] = f'{verified_fact}\n{content}'.strip()

        return proposed, parsed
