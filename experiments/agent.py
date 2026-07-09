"""Agent utilities: system prompts and conversation history formatting.

These are kept separate from the runner so mechanisms can reuse them
without pulling in the full runner dependency graph.
"""

SYSTEM_PROMPT_TEMPLATE = (
    'You are playing {game_name}. '
    'Return ONLY a JSON object with keys "reasoning" and "action". '
    'The action must be a short parser command such as LOOK, OPEN MAILBOX, '
    'TAKE LEAFLET, or GO NORTH. '
    'Do not include markdown, bullets, or any extra text.'
)


def get_system_prompt(game_name):
    """Return the system prompt for a given game."""
    return SYSTEM_PROMPT_TEMPLATE.format(game_name=game_name)


def format_history(history):
    """Convert a list of step records into OBS/ACT/REAS prompt lines.

    Args:
        history: list of dicts, each with at least 'obs' and 'action' keys.

    Returns:
        list of strings, one per history step.
    """
    return [
        f"OBS: {h['obs']}\nACT: {h['action']}\nREAS: {h.get('reasoning', '')}"
        for h in history
        if isinstance(h, dict) and 'obs' in h
    ]


def build_step_messages(game_name, history, observation):
    """Build the full messages list for a single game step.

    Returns a list of {role, content} dicts ready for the LLM client.
    """
    history_lines = format_history(history)
    user_content = '\n'.join(history_lines + [f'OBSERVE: {observation}'])
    return [
        {'role': 'system', 'content': get_system_prompt(game_name)},
        {'role': 'user', 'content': user_content},
    ]
