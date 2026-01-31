# A GNN-based Job Predictor

Your goal is to develop a Binary Node Classifier (a GNN) for the given dataset.

## Dataset Details

Nodes are developers who have starred at most minuscule 10 repositories, and edges are mutual follower relationships between them. The vertex features are extracted based on the location; repositories starred, employer and e-mail address.

### edges.csv
Contains the edge list of the GitHub social network, where each row represents an undirected connection between two developers (nodes).

### target.csv
Contains node-level names and labels for each developer.

### features.json
Contain dense representation of features of every developer.

## Goal

Train a Binary Node Classifier (a GNN) that predicts whether the developer is a web or a machine learning developer.
