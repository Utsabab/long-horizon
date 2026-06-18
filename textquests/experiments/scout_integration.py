import json
import subprocess
from pathlib import Path

def export_transcript_for_scout(transcript, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2)
    return out_path

def run_scout_scan(scanner_file, transcripts_dir, model=None):
    cmd = ["scout", "scan", scanner_file, "-T", transcripts_dir]
    if model:
        cmd += ["--model", model]
    # run in subprocess; user must have inspect-scout installed and configured
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr
