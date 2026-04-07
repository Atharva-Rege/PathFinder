# PathFinder

## Overview

PathFinder is a Graph Neural Network (GNN)-based recommendation system designed to match candidates with relevant jobs.
It models candidates, jobs, and their attributes as a temporal heterogeneous graph and performs link prediction to generate personalized recommendations.

The system is built to address key challenges in recruitment such as cold-start problems and time-sensitive recommendations, while supporting dynamic updates and online learning to continuously improve from user interactions.

---

## Key Features

* Graph-based recommendation using PyTorch Geometric
* Temporal heterogeneous graph modeling (skills, experience, time, etc.)
* Dual-sided recommendation (candidate ↔ recruiter)
* Real-time candidate and job ingestion
* Link prediction for candidate–job matching
* Online learning with incremental retraining
* Dynamic graph updates with continuous persistence

---

## System Architecture

```
User Input → Graph Update → GNN Model → Ranking → Interaction Logging → Retraining → Updated Model
```
---
## Project Structure

PathFinder/

* **activation.py**: Contains activation function implementations used in the model

* **api_server.py**: Backend API server for handling inference requests

* **candidate_input.py**: Processes candidate input into structured features

* **evaluate.py**: Evaluates model performance using various metrics

* **event_processor.py**: Handles and processes interaction events

* **graph_builder.py**: Constructs the initial heterogeneous graph

* **graph_persistence.py**: Saves and loads graph state

* **graph_runtime.py**: Handles dynamic graph updates during runtime

* **infer_cli.py**: CLI interface for inference and interaction loop

* **inference_types.py**: Defines data structures for inference pipeline

* **interaction_logger.py**: Logs user interactions for training

* **main.py**: Entry point for running the system

* **main_notebook.ipynb**: Notebook for experimentation and analysis

* **model.py**: Defines the GNN architecture (GraphSAGE + GAT)

* **prepare_runtime_features.py**: Prepares features for runtime inference

* **ranker.py**: Implements ranking logic for recommendations

* **retrain.py**: Performs incremental retraining using new interactions

* **retrain_trigger.py**: Triggers retraining based on interaction threshold

* **run_model_api.py**: Runs model inference through API interface

* **train.py**: Script for initial model training

* **required_data.rar**: Contains dataset required for graph construction

* **requirements.txt**: Lists all project dependencies

* **pyrightconfig.json**: Configuration for static type checking

* **backend/**: Backend implementation (Django services)

* **frontend/**: Frontend application (React interface)

* **Learning-Phase/**: Experimental scripts and development phase code


---

## How It Works

### 1. Graph Construction

* Entities are represented as nodes:

  * Candidate, Job, Skill, Company, Experience, Salary, Category, Concept, Time, Contract, Origin, Shortlist
* Relationships are modeled as edges (18 types, bidirectional)
* Shortlist nodes explicitly represent candidate–job interactions with timestamps
* Temporal nodes enforce recency constraints
* Implemented using PyTorch Geometric’s `HeteroData`

---

### 2. Model Architecture

* Combines GraphSAGE and Graph Attention (GAT) layers
* Aggregates structural and semantic information across node types
* Uses cosine similarity between embeddings for ranking
* Learns link prediction between shortlist and job nodes

---

### 3. Inference (Recommendation)

* New candidates/jobs are dynamically added to the graph
* Features are initialized from processed input (resume/job description embeddings)
* Edges are automatically created (e.g., candidate → skill)
* A temporary shortlist node is created for querying
* Top-K recommendations are generated via link prediction

---

### 4. Online Learning Pipeline

```
Ranking → Interaction Logging → Threshold Check → Retraining → Model Reload
```

* User interactions are stored in `interactions.json`
* Each interaction becomes a new graph edge
* Retraining is triggered after a predefined threshold
* The global graph is continuously updated and persisted

---

### 5. Incremental Retraining

* Uses mini-batch training with neighbor sampling
* Combines:

  * Positive samples (real interactions)
  * Negative samples (random job nodes)
* Preserves existing weights and fine-tunes with new data
* Avoids full retraining

---

## Technologies Used

* Python
* PyTorch
* PyTorch Geometric (PyG)
* NumPy, Pandas
* Sentence Transformers
* Django (Backend)
* React + Three.js (Frontend)

---

## Evaluation Metrics

* AUC
* Precision
* Recall
* F1 Score
* Accuracy

---
## Frontend Output

<p align="center">
  <img src="./images/fe.jpg" width="700"/>
</p>

## Sample Output

<p align="center">
  <img src="./images/sample_op.jpg" width="700"/>
</p>
