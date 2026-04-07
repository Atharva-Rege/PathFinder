import json
from pathlib import Path

LOG = Path("output/interactions.json")

def log_interaction(candidate_idx=None, job_idx=None, score=0.0, ts=0, candidature_idx=None):
    # Accept both candidate_idx and legacy candidature_idx to avoid runtime keyword mismatches.
    if candidate_idx is None:
        candidate_idx = candidature_idx
    if candidate_idx is None:
        raise ValueError("candidate_idx is required")
    if job_idx is None:
        raise ValueError("job_idx is required")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"c": int(candidate_idx), "j": int(job_idx), "s": float(score), "t": int(ts)}
    data = json.loads(LOG.read_text()) if LOG.exists() else []
    data.append(entry)
    LOG.write_text(json.dumps(data, indent=2))