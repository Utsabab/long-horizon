"""M3: State externalization mechanism (research agenda §2.2).

Accepted-Local-Error channel: the agent misrepresents state — it acts on a
stale or imagined picture of where it is or what it's carrying instead of
the game's actual ground truth.

M3 targets this by probing the live environment for ground-truth location
and inventory every step — the same read-only save/restore pattern already
used by channels.env_checker._probe() to verify ALE beliefs — and handing
that verified state to the rest of the harness (M2's scratchpad, the
integration channel, the runner's own bookkeeping) instead of letting any
component infer location/inventory from prose.

Zero LLM cost, zero permanent env mutation.
"""


def _probe(env, command):
    """Issue a read-only probe command and return its raw response text.

    Identical pattern to channels.env_checker._probe(): save state, issue
    the command, restore state — so the probe never permanently mutates the
    live environment.
    """
    saved = env.get_state()
    try:
        text, _, _, _ = env._pre_step(command)
    finally:
        env.set_state(saved)
    return text


def _parse_inventory_items(inventory_text):
    """Split a raw inventory probe response into a flat list of item names.

    This is mechanical list-splitting on a probe response with a fairly
    regular "you are carrying: a, b, and c" shape — not semantic inference
    over open-ended prose. Any line that doesn't look like a list is skipped
    rather than guessed at.
    """
    if not inventory_text:
        return []
    items = []
    for line in inventory_text.splitlines():
        line = line.strip().strip('.')
        if not line or ':' not in line and not line[:1].isupper():
            continue
        _, _, rest = line.partition(':')
        rest = rest if ':' in line else line
        for chunk in rest.replace(' and ', ',').split(','):
            item = chunk.strip().strip('.')
            for article in ('a ', 'an ', 'the '):
                if item.lower().startswith(article):
                    item = item[len(article):]
                    break
            item = item.strip()
            if item and len(item) < 60:
                items.append(item)
    return items


# Boilerplate line prefixes emitted by the game engine itself (banner/copyright
# lines), not part of any room description — filtered out mechanically, same
# list mechanisms.memory._extract_location used to use for the same purpose.
_SKIP_PREFIXES = ('ZORK', 'Infocom', 'Copyright', 'Release', '>')


def _first_room_line(look_text):
    """Return the first non-boilerplate line of a 'look' probe response.

    Text adventure engines render the room name as the first line of a
    'look' response by convention, so this is picking a fixed structural
    position in a probe response — not inferring meaning from free-form
    reasoning prose.
    """
    for line in (look_text or '').splitlines():
        line = line.strip()
        if line and not any(line.startswith(p) for p in _SKIP_PREFIXES):
            if len(line) < 60:
                return line
    return None


class StateExternalizer:
    """M3: read-only per-step ground-truth probe of location + inventory."""

    def probe(self, env):
        """Return the current ground-truth state:

            {
              'location': str|None,       # short room name (first line of 'look')
              'location_text': str,       # full raw 'look' probe response
              'inventory_text': str,      # full raw 'inventory' probe response
              'inventory_items': list[str],
            }
        """
        location_text = _probe(env, 'look')
        inventory_text = _probe(env, 'inventory')
        return {
            'location': _first_room_line(location_text),
            'location_text': location_text,
            'inventory_text': inventory_text,
            'inventory_items': _parse_inventory_items(inventory_text),
        }
