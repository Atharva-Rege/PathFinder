# Job Prediction with Graph Neural Networks (GNNs)

A Binary Node Classifier (a GNN) that predicts whether the developer is a web or a machine learning developer: implemented various Graph Neural Net

### Models Implemented

Implemented modular PyTorch Geometric (`torch_geometric`) classes for:

* **GCN (Graph Convolutional Network):** Uses mean aggregation; generally stable.
* **GAT (Graph Attention Network):** Uses attention mechanisms to weigh neighbor importance; handles padded zeros effectively.
* **GIN (Graph Isomorphism Network):** A highly expressive model using sum aggregation and MLPs; tested with and without Batch Normalization.

## Observations

| Experiment | Model | Observation |
| --- | --- | --- |
| **Padding** | **GAT** | **Best Performer.** Achieved ~84% accuracy. Attention mechanism effectively filters out padded noise. |
| **Padding** | **GIN** | **Unstable.** Suffered from loss spikes and poor convergence due to noise amplification in sum aggregation. |
| **Aggregation** | **GIN** | **Stable & Effective.** Compressing features to 5 statistical values stabilized the GIN model, yielding competitive performance (~82%). |

### While Attention (GAT) can handle sparse, padded high-dimensional data, MLP-based aggregation (GIN) struggles with it and benefits significantly from dense feature engineering (Statistical Aggregation).
