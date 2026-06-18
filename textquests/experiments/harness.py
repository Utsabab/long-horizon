from collections import deque
import re

class AcceptedLocalErrorDetector:
    """Detects repeated actions and extracts beliefs to identify Accepted-Local-Error (ALE)."""
    BELIEF_PATTERNS = [
        re.compile(r"\bI (?:have|picked up|picked|took|grabbed) (?:the )?(?P<item>[\w '\-]+)", re.I),
        re.compile(r"\bI (?:dropped|left) (?:the )?(?P<item>[\w '\-]+)", re.I),
        re.compile(r"\b(?P<item>[\w '\-]+) is in the (?P<loc>[\w '\-]+)", re.I),
        re.compile(r"\bIt (?:'s| is) in the (?P<loc>[\w '\-]+)", re.I),
        re.compile(r"\bI (?:remember|know) that (?P<clause>.+)", re.I),
    ]

    def __init__(self, k=3, window=10):
        self.k = k
        self.window = window
        self.actions = deque(maxlen=window)
        self.progress = deque(maxlen=window)
        self.reasonings = deque(maxlen=window)

    def add(self, action, progress, reasoning=None):
        self.actions.append(action)
        self.progress.append(progress)
        self.reasonings.append(reasoning or "")

    def check_repetition(self):
        if len(self.actions) < self.k:
            return False
        last = list(self.actions)[-self.k:]
        if all(a == last[0] for a in last):
            # ensure progress hasn't changed in the recent window
            if len(self.progress) >= 2 and max(self.progress) == min(self.progress):
                return True
        return False

    def extract_belief(self, reasoning_text):
        if not reasoning_text:
            return None
        for pat in self.BELIEF_PATTERNS:
            m = pat.search(reasoning_text)
            if m:
                gd = m.groupdict()
                # prefer item-based beliefs
                if 'item' in gd and gd['item']:
                    return ('have_item', gd['item'].strip())
                if 'loc' in gd and gd['loc']:
                    return ('item_in_loc', gd.get('item', '').strip() or None, gd['loc'].strip())
                if 'clause' in gd and gd['clause']:
                    return ('clause', gd['clause'].strip())
        return None

    def check_ale(self, env_checker):
        """Check for Accepted-Local-Error by using env_checker(belief) -> (contradiction_bool, evidence)

        Returns (is_ale:bool, details:dict)
        """
        if not self.check_repetition():
            return (False, {'reason': 'no_repetition'})
        # look for most recent belief in reasonings
        for reasoning in reversed(self.reasonings):
            belief = self.extract_belief(reasoning)
            if belief:
                # env_checker should accept belief and return (contradiction_bool, evidence)
                contradiction, evidence = env_checker(belief)
                details = {'belief': belief, 'contradiction': contradiction, 'evidence': evidence}
                return (contradiction, details)
        return (False, {'reason': 'no_belief_detected'})


if __name__ == '__main__':
    d = AcceptedLocalErrorDetector(k=3, window=5)
    d.add('open door', 0, 'I have the key')
    d.add('open door', 0, 'I have the key')
    d.add('open door', 0, 'I have the key')
    print(d.check_repetition(), d.extract_belief('I have the key'), d.check_ale(lambda b: (True, 'inventory shows no key')))
