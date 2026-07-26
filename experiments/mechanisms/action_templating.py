"""M5: Action templating mechanism (secondary; research agenda §2.2).

Interface channel: the agent can't produce valid actions — e.g. it wraps its
command in a label the game parser chokes on ("ACT: SEARCH ROAD" instead of
"SEARCH ROAD"), or prefixes it with a stray "> " prompt marker it copied from
formatting examples.

M5 is a guardrail, not an inference step: it mechanically strips a small set
of known literal prefixes right before env.step() is called, on every path
(normal generation and the M1 intervention path alike). This generalizes the
_LABEL_PREFIX stripping that used to live only inside
mechanisms.info_seeking.InfoSeekingMechanism.intervene().
"""

import re

# Matches a leading label the model sometimes prepends to its command line
# despite the prompt asking for a bare parser command (e.g. "ACT: SEARCH
# ROAD", "> GO NORTH"). Left in place, the game parser fails outright (e.g.
# "This story doesn't know the word 'act:.'"), wasting the step. This is
# mechanical literal-prefix stripping, not semantic inference over prose.
_LABEL_PREFIX = re.compile(r'^(?:>\s*|(?:ACT(?:ION)?|COMMAND)\s*:\s*)', re.I)


def normalize_action(raw):
    """Strip known parser-hostile prefixes from a proposed action.

    Args:
        raw: the action string as proposed by the model or a mechanism
             (e.g. InfoSeekingMechanism.intervene(), PlanningMechanism.replan()).

    Returns:
        str: the cleaned action, safe to pass to env.step().
    """
    if not raw:
        return raw
    return _LABEL_PREFIX.sub('', raw.strip()).strip()
