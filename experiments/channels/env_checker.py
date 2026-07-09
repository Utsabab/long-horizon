"""Ground-truth belief checker for AcceptedLocalErrorDetector.check_ale().

Verifies a belief tuple extracted by AcceptedLocalErrorDetector.extract_belief()
against the live TextQuestsEnv state, without mutating that state.
"""


def _probe(env, command):
    """Issue a read-only probe command and return its raw response text."""
    saved = env.get_state()
    try:
        text, _, _, _ = env._pre_step(command)
    finally:
        env.set_state(saved)
    return text


def _check_have_item(env, item):
    text = _probe(env, 'inventory')
    has_item = bool(item) and item.lower() in text.lower()
    contradiction = not has_item
    return contradiction, {
        'reason': 'inventory_probe',
        'item': item,
        'probe_text': text,
        'has_item': has_item,
    }


def _check_item_in_loc(env, item, location):
    text = _probe(env, 'look')
    if not location or location.lower() not in text.lower():
        return False, {
            'reason': 'unverifiable_location',
            'claimed_location': location,
            'probe_text': text,
        }
    item_mentioned = bool(item) and item.lower() in text.lower()
    contradiction = bool(item) and not item_mentioned
    return contradiction, {
        'reason': 'location_probe',
        'item': item,
        'claimed_location': location,
        'probe_text': text,
        'item_mentioned': item_mentioned,
    }


def make_env_checker(env):
    """Return a callable(belief) -> (is_contradiction, evidence) bound to env.

    belief is a typed tuple from AcceptedLocalErrorDetector.extract_belief():
      ('have_item', item)
      ('item_in_loc', item_or_None, location)
      ('clause', text)  -- unverifiable structurally
    """
    def checker(belief):
        kind = belief[0] if belief else None
        if kind == 'have_item':
            return _check_have_item(env, belief[1])
        if kind == 'item_in_loc':
            return _check_item_in_loc(env, belief[1], belief[2])
        return False, {'reason': 'unverifiable_clause', 'belief': belief}

    return checker
