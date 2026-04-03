import json
from pathlib import Path

LOG_PATH = Path("output/interactions.json")

STATE_PATH = Path("output/retrain_state.json")

def should_retrain(threshold=3):
    if not LOG_PATH.exists():
        return False

    data = json.loads(LOG_PATH.read_text())
    last_count = 0

    if STATE_PATH.exists():
        last_count = json.loads(STATE_PATH.read_text()).get("last_count", 0)

    if len(data) - last_count >= threshold:
        STATE_PATH.write_text(json.dumps({"last_count": len(data)}))
        return True

    return False