import json
from pathlib import Path

LOG = Path("output/interactions.json")

def log_interaction(candidate_idx, job_idx, score, ts):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"c": candidate_idx, "j": job_idx, "s": float(score), "t": int(ts)}
    data = json.loads(LOG.read_text()) if LOG.exists() else []
    data.append(entry)
    LOG.write_text(json.dumps(data, indent=2))