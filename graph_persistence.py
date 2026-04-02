from pathlib import Path
import torch


def save_graph(data, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def load_graph(path: str | Path):
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def save_mappings(mappings, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mappings, path)


def load_mappings(path: str | Path):
    return torch.load(Path(path), map_location="cpu", weights_only=False)