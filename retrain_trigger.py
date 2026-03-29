import json
from pathlib import Path

LOG_PATH = Path("output/interactions.json")

#threshold 3 for now (?)
def should_retrain(threshold=3):
    if not LOG_PATH.exists():
        return False

    data = json.loads(LOG_PATH.read_text())
    return len(data) >= threshold