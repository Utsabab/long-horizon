import json
import os
from pathlib import Path

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

class JSONLogger:
    def __init__(self, out_dir):
        ensure_dir(out_dir)
        self.out_dir = out_dir

    def log_step(self, run_id, step_record):
        path = Path(self.out_dir) / f"{run_id}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(step_record, default=str) + "\n")

    def save_transcript(self, run_id, transcript):
        path = Path(self.out_dir) / f"{run_id}_transcript.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=2, default=str)
