import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero, GATConv, GCNConv
from torch_geometric.nn import norm as N

from activation import ReGLU, GEGLU
import torch.nn as nn


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, operator="sage", batch_norm=False, linear_unit_label='relu', num_layers=3):
        super().__init__()
        self.batch_norm = batch_norm
        self.linear_unit_label = linear_unit_label
        if linear_unit_label == 'relu':
            self.linear_unit = F.relu
        elif linear_unit_label == 'leakyrelu':
            self.linear_unit = F.leaky_relu
        elif linear_unit_label == 'gelu':
            self.linear_unit = F.gelu
        elif linear_unit_label == 'geglu':
            self.linear_unit = GEGLU()
        elif linear_unit_label == 'reglu':
            self.linear_unit = ReGLU()
        else:
            print("Error: linear_unit not recognized")
            print("Must be 'relu', 'leakyrelu', 'gelu' or 'geglue'")

        # loop to declare x number of conv layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            if operator == "sage":
                self.layers.append(SAGEConv(hidden_channels, hidden_channels))
            elif operator == "gat":
                self.layers.append(GATConv(hidden_channels, hidden_channels, add_self_loops=False))
            elif operator == "gcn":
                self.layers.append(GCNConv(hidden_channels, hidden_channels))
            else:
                print("Error: operator not recognized")
                print("Must be 'sage', 'gat' or 'gcn'")
            if self.batch_norm == 'batch_norm':
                self.norms.append(N.BatchNorm(hidden_channels))
            elif self.batch_norm == 'layer_norm':
                self.norms.append(N.LayerNorm(hidden_channels))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.batch_norm:
                x = self.norms[i](x)
            x = self.linear_unit(x)

        return x


class Classifier(torch.nn.Module):
    # Our final classifier applies the dot-product between source and destination
    # node embeddings to derive edge-level predictions
    def forward(self, x_user: Tensor, x_item: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_item = x_item[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(edge_feat_user, edge_feat_item)


class Model(torch.nn.Module):
    def __init__(self,
                 hidden_channels,
                 data,
                 remove_feat=False,
                 freeze=False,
                 list_abl=None,
                 predict_edge=("candidature", "has_application", "job"),
                 batch_norm=False,
                 linear_unit_label='relu',
                 num_layers=3,
                 conv_operator="sage"
                 ):
        super().__init__()
        if list_abl is None:
            list_abl = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and items
        self.freeze = freeze
        self.remove_feat = remove_feat
        self.predict_edge = predict_edge
        self.list_abl = list_abl

        self.user_lin = torch.nn.Linear(384, hidden_channels)
        self.job_lin = torch.nn.Linear(384, hidden_channels)

        self.user_lin_emb = torch.nn.Linear(hidden_channels, hidden_channels)
        self.job_lin_emb = torch.nn.Linear(hidden_channels, hidden_channels)

        self.user_emb = torch.nn.Embedding(data["candidate"].num_nodes, hidden_channels)
        self.job_emb = torch.nn.Embedding(data["job"].num_nodes, hidden_channels)

        if self.predict_edge[0] == "candidature":
            self.candidature_emb = torch.nn.Embedding(data["candidature"].num_nodes, hidden_channels)

            if self.freeze:
                # embedding is a 0 tensor of size num_nodes x hidden_channels
                self.candidature_emb.weight.data.fill_(0)
                # self.candidature_emb.weight.data = self.candidature_emb.weight.data.long()
                self.candidature_emb.freeze = True

        if self.freeze:
            # embedding is a 0 tensor of size num_nodes x hidden_channels
            self.user_emb.weight.data.fill_(0)
            self.job_emb.weight.data.fill_(0)
            self.user_emb.freeze = True
            self.job_emb.freeze = True

        if list_abl[0] == 1:
            self.skill_emb = torch.nn.Embedding(data["skill"].num_nodes, hidden_channels)
        if list_abl[1] == 1:
            self.contract_emb = torch.nn.Embedding(data["contract"].num_nodes, hidden_channels)
        if list_abl[2] == 1:
            self.origin_emb = torch.nn.Embedding(data["origin"].num_nodes, hidden_channels)
        if list_abl[3] == 1:
            self.experience_emb = torch.nn.Embedding(data["experience"].num_nodes, hidden_channels)
        if list_abl[4] == 1:
            self.salary_emb = torch.nn.Embedding(data["salary"].num_nodes, hidden_channels)
        # if list_abl[5] == 1:
        #     self.zip_emb = torch.nn.Embedding(data["zip"].num_nodes, hidden_channels)
        if list_abl[5] == 1:
            self.category_emb = torch.nn.Embedding(data["category"].num_nodes, hidden_channels)
        # if list_abl[7] == 1:
        #     self.recruiter_emb = torch.nn.Embedding(data["recruiter"].num_nodes, hidden_channels)
        if list_abl[6] == 1:
            self.company_emb = torch.nn.Embedding(data["company"].num_nodes, hidden_channels)
        if list_abl[7] == 1:
            self.concept_emb = torch.nn.Embedding(data["concept"].num_nodes, hidden_channels)
        if list_abl[8] == 1:
            self.time_node_emb = torch.nn.Embedding(data["time"].num_nodes, hidden_channels)
            self.time_lin_emb = torch.nn.Linear(1, hidden_channels)

        # Instantiate homogeneous GNN
        self.gnn = GNN(hidden_channels, batch_norm=batch_norm, linear_unit_label=linear_unit_label, num_layers=num_layers, operator=conv_operator)
        # Convert GNN model into a heterogeneous variant
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        list_abl = self.list_abl
        x_dict = {}

        # remove feature
        if self.remove_feat:
            x_dict["candidate"] = self.user_lin(data["candidate"].x)
            x_dict["job"] = self.job_lin(data["job"].x)
        else:
            x_dict["candidate"] = self.user_lin(data["candidate"].x) + self.user_lin_emb(
                self.user_emb(data["candidate"].node_id))
            x_dict["job"] = self.job_lin(data["job"].x) + self.job_lin_emb(self.job_emb(data["job"].node_id))

        if self.predict_edge[0] == "candidature":
            x_dict["candidature"] = self.candidature_emb(data["candidature"].node_id)

        if list_abl[0] == 1:
            x_dict["skill"] = self.skill_emb(data["skill"].node_id)
        if list_abl[1] == 1:
            x_dict["contract"] = self.contract_emb(data["contract"].node_id)
        if list_abl[2] == 1:
            x_dict["origin"] = self.origin_emb(data["origin"].node_id)
        if list_abl[3] == 1:
            x_dict["experience"] = self.experience_emb(data["experience"].node_id)
        if list_abl[4] == 1:
            x_dict["salary"] = self.salary_emb(data["salary"].node_id)
        # if list_abl[5] == 1:
        #     x_dict["zip"] = self.zip_emb(data["zip"].node_id)
        if list_abl[5] == 1:
            x_dict["category"] = self.category_emb(data["category"].node_id)
        # if list_abl[7] == 1:
        #     x_dict["recruiter"] = self.recruiter_emb(data["recruiter"].node_id)
        if list_abl[6] == 1:
            x_dict["company"] = self.company_emb(data["company"].node_id)
        if list_abl[7] == 1:
            x_dict["concept"] = self.concept_emb(data["concept"].node_id)
        if list_abl[8] == 1:
            x_dict["time"] = self.time_lin_emb(data["time"].x)

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict[self.predict_edge[0]],
            x_dict[self.predict_edge[2]],
            data[self.predict_edge[0], self.predict_edge[1], self.predict_edge[2]].edge_label_index,
        )

        return pred
