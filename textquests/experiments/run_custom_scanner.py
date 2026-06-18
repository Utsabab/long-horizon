#!/usr/bin/env python3
import os, json, glob, yaml
from experiments.openrouter_client import OpenRouterClient

cfg = yaml.safe_load(open('textquests/experiments/config.yaml'))
client = OpenRouterClient(api_key=cfg['openrouter']['api_key'], model=cfg['openrouter']['model'], base_url=cfg['openrouter']['base_url'])

LOG_DIR = 'textquests/experiments/logs'
OUT = os.path.join(LOG_DIR, 'scans_results.jsonl')

files = glob.glob(os.path.join(LOG_DIR, '*_transcript.json'))
if not files:
    print('No transcript files found in', LOG_DIR)
    raise SystemExit(1)

with open(OUT, 'w', encoding='utf-8') as out:
    for f in files:
        data = json.load(open(f))
        # data is list of step records
        lines = []
        for i, step in enumerate(data):
            obs = step.get('obs','')
            act = step.get('action','')
            reas = step.get('reasoning','')
            lines.append(f"Step {i}: OBS: {obs.strip()} ACT: {act.strip()} REAS: {str(reas).strip()}")
        transcript_text = "\n".join(lines[-200:])
        prompt = (
            "Here is a transcript of an agent playing a text-based game.\n\n" +
            transcript_text +
            "\n\nQuestion: Does this transcript show the agent repeatedly accepting a local error or looping on a failed action? If yes, answer 'Yes' and provide a short explanation and step numbers. If no, answer 'No'."
        )
        resp = client.generate(prompt, max_tokens=256, reasoning_enabled=True)
        parsed = client.parse_completion(resp)
        out_record = {'file': f, 'response': parsed}
        out.write(json.dumps(out_record) + "\n")
        print('Scanned', f)
print('Done. Results in', OUT)
