from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader

from graph_runtime import load_runtime_artifacts


DATA_DIR = Path(__file__).resolve().parent / "required_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
INTERACTIONS_PATH = OUTPUT_DIR / "interactions.json"
MODEL_PATH = Path(__file__).resolve().parent / "model.pt"


EPOCHS = 10
LR = 1e-3


def load_interactions():
    if not INTERACTIONS_PATH.exists():
        return []
    with open(INTERACTIONS_PATH, "r") as f:
        return json.load(f)


def main():
    print("Starting incremental retraining...")

    artifacts = load_runtime_artifacts(
        data_dir=DATA_DIR,
        model_path=MODEL_PATH,
        config=None,
    )

    model = artifacts.model
    data = artifacts.data

    
    print("Graph stats:",
          data["candidate"].node_id.numel(),
          data["job"].node_id.numel(),
          data["candidature"].node_id.numel())

    model.train()

    interactions = load_interactions()

    if len(interactions) == 0:
        print("No interactions found. Skipping retraining.")
        return

    # Use recent interactions only
    interactions = interactions[-100:]

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    predict_edge = ("candidature", "has_application", "job")

    edge_list = []
    timestamps = []

    for interaction in interactions:
        edge_list.append([
            interaction["c"],   
            interaction["j"]
        ])
        timestamps.append(interaction["t"])

    if len(edge_list) == 0:
        print("No valid interactions for training.")
        return

    edge_index = torch.tensor(edge_list).t().long()
    num_pos = edge_index.shape[1]

    # Negative sampling 
    all_jobs = data["job"].node_id
    neg_jobs = all_jobs[torch.randint(0, len(all_jobs), (num_pos,))]

    neg_edge_index = torch.stack([
        edge_index[0],  
        neg_jobs
    ])

    # Combine positive + negative
    edge_index = torch.cat([edge_index, neg_edge_index], dim=1)

    edge_label = torch.cat([
        torch.ones(num_pos),
        torch.zeros(num_pos)
    ])

    
    edge_label_time = torch.tensor(timestamps + timestamps).long()

    
    assert edge_index.shape[1] == edge_label.shape[0], "Edge-label mismatch"

    loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[20, 10],
        edge_label_index=(predict_edge, edge_index),
        edge_label=edge_label,
        edge_label_time=edge_label_time,
        temporal_strategy="uniform",
        time_attr="timestamp",
        batch_size=128,
        shuffle=True,
    )

    device = next(model.parameters()).device

    for epoch in range(EPOCHS):
        total_loss = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)

            loss = F.binary_cross_entropy_with_logits(
                pred,
                batch[predict_edge]["edge_label"]
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    torch.save(model, MODEL_PATH)
    print("Retraining complete. Model updated.")


if __name__ == "__main__":
    main()

