#!/usr/bin/env python3
"""Simple transcript analyzer for Accepted-Local-Error (ALE).
Scans *_transcript.json files and extracts candidate beliefs from agent reasoning;
then checks for supporting evidence earlier in the transcript to decide if belief
is contradicted by observed history.

Outputs results to logs/scans_ale_results.jsonl
"""
import json, glob, os, re

LOG_DIR = 'textquests/experiments/logs'
OUT = os.path.join(LOG_DIR, 'scans_ale_results.jsonl')

BELIEF_RE = re.compile(r"\bI (?:have|picked up|picked|took|grabbed) (?:the )?(?P<item>[\w '\-]+)|(?P<pair_item>[\w '\-]+) is in the (?P<loc>[\w '\-]+)", re.I)

files = glob.glob(os.path.join(LOG_DIR, '*_transcript.json'))
if not files:
    print('No transcripts found in', LOG_DIR)
    raise SystemExit(1)

results = []
for f in files:
    data = json.load(open(f))
    # build list of observations up to each step
    observations = []
    inventory_history = []
    for i, step in enumerate(data):
        obs = step.get('obs','') or ''
        observations.append(obs.lower())
        # try to extract inventory mentions
        if 'inventory' in obs.lower():
            inventory_history.append((i, obs))

    # now scan reasonings for beliefs
    beliefs = []
    for i, step in enumerate(data):
        reasoning = step.get('reasoning') or ''
        if not reasoning:
            continue
        for m in BELIEF_RE.finditer(reasoning):
            item = None
            loc = None
            if m.group('item'):
                item = m.group('item').strip().lower()
            if m.group('pair_item') and m.group('loc'):
                item = m.group('pair_item').strip().lower()
                loc = m.group('loc').strip().lower()
            belief_text = reasoning.strip().replace('\n',' ')[:400]
            # check prior obs for evidence
            prior_obs = ' '.join(observations[:i+1])
            support = None
            evidence = None
            if item:
                # check if item appeared in inventory previously
                for idx, inv in inventory_history:
                    if item in inv.lower():
                        support = True
                        evidence = {'type':'inventory', 'step': idx, 'text': inv}
                        break
                # check if item ever mentioned as taken
                if support is None:
                    if any(item in o for o in observations[:i+1]):
                        support = True
                        evidence = {'type':'observation', 'text': 'found mention in prior obs'}
                if support is None:
                    support = False
                    evidence = {'type':'none', 'text': 'no prior evidence in transcript'}
            elif loc:
                # check if item-in-loc mentions exist
                if any(loc in o and item and item in o for o in observations[:i+1]):
                    support = True
                    evidence = {'type':'observation', 'text':'prior mention matched'}
                else:
                    support = False
                    evidence = {'type':'none', 'text':'no prior evidence in transcript'}

            beliefs.append({'step': i, 'belief_text': belief_text, 'item': item, 'loc': loc, 'supported': support, 'evidence': evidence})

    out = {'file': f, 'beliefs': beliefs}
    results.append(out)
    # append to output file
    with open(OUT, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(out) + '\n')
    print('Scanned', f, 'found', len(beliefs), 'beliefs')

print('Wrote results to', OUT)
