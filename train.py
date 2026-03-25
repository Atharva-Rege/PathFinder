from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import LinkNeighborLoader

from evaluate import evaluate, evaluate_top_k
from utils import transform


def interlace(list1, list2):
    result = [None] * (len(list1) + len(list2))
    result[::2] = list1
    result[1::2] = list2
    return result


# def resample_usr(sampled_data, predict_edge=("candidature", "has_application", "job")):
#     """
#         Resample the user to create negative samples.
#         Args:
#             sampled_data (torch_geometric.data.HeteroData): The dataset.
#             predict_edge (tuple): The edge to predict.
#         Returns:
#             The dataset with negative samples.
#     """
#     edge_label_index = sampled_data[predict_edge[0], predict_edge[1], predict_edge[2]].edge_label_index
#     z_user = edge_label_index[0].tolist() + edge_label_index[0].tolist()
#     z_item = []
#     for i in range(0, len(edge_label_index[0])):
#         rand_item_list = edge_label_index[1].tolist()
#         if len(rand_item_list) != 1:
#             del rand_item_list[i]
#         rng = np.random.default_rng()
#         z_item.append(rng.choice(rand_item_list))

#     z_item_ = z_item + edge_label_index[1].tolist()
#     edge_index = [z_user, z_item_]
#     sampled_data[predict_edge[0], predict_edge[1], predict_edge[2]].edge_label_index = torch.tensor(edge_index)
#     sampled_data[predict_edge[0], predict_edge[1], predict_edge[2]].edge_label = torch.tensor(
#         [1] * len(edge_label_index[0]) + [0] * len(edge_label_index[0]))
#     sampled_data[predict_edge[0], predict_edge[1], predict_edge[2]].input_id = torch.tensor(
#         list(range(0, len(edge_label_index[0] * 2))))
#     return sampled_data


def split_data(data,
               predict_edge=("candidature", "has_application", "job"),
               num_val=0.1,
               num_test=0.1, ):
    # For this, we first split the set of edges into
    # training (80%), validation (10%), and testing edges (10%).
    # Across the training edges, we use 70% of edges for message passing,
    # and 30% of edges for supervision.
    # We further want to generate fixed negative edges for evaluation with a ratio of 2:1.
    # Negative edges during training will be generated on-the-fly.
    # We can leverage the `RandomLinkSplit()` transform for this from PyG:
    print(predict_edge)
    print(data[predict_edge[0], predict_edge[1], predict_edge[2]])
    print(predict_edge[2], f"rev_{predict_edge[1]}", predict_edge[0])
    print(data.edge_types)
    transform = T.RandomLinkSplit(
        num_val=num_val,
        num_test=num_test,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=1.0,
        is_undirected=True,
        add_negative_train_samples=False,
        edge_types=(predict_edge[0], predict_edge[1], predict_edge[2]),
        rev_edge_types=(predict_edge[2], f"rev_{predict_edge[1]}", predict_edge[0])
    )
    train_data, val_data, test_data = transform(data)

    return train_data, val_data, test_data


def ts_loader(data,
              percentile_min,
              percentile_max,
              num_neigh=None,
              predict_edge=("candidature", "has_application", "job"),
              strategy='uniform',
              batch_size=128,
              ):
    """
        Create a loader for train, test and val.
        Args:
            percentile_max: 
            percentile_min: 
            data (torch_geometric.data.HeteroData): The dataset.
            num_neigh : The number of neighbors to sample for each edge type.
            predict_edge : The edge to predict.
            strategy (str): The strategy to use for temporal sampling.
            batch_size (int): The batch size.
        Returns:
            The loader.
    """
    if num_neigh is None:
        num_neigh = [20, 10]

    # Define seed edges:
    edge_label_index = data[predict_edge[0], predict_edge[1], predict_edge[2]].edge_index
    
    # add negative sampling
    edge_label_index_0_arr = edge_label_index[0].numpy()
    edge_label_index_1_arr = edge_label_index[1].numpy()
    z_user = interlace(edge_label_index_0_arr.tolist(), edge_label_index_0_arr.tolist())
    z_item = []

    for i in range(0, len(edge_label_index[0])):
        true_item = edge_label_index_1_arr[i]
        rand_item_list = [item for item in edge_label_index_1_arr.tolist() if item != true_item]
        rng = np.random.default_rng()
        rand_item_choice = rng.choice(rand_item_list) if rand_item_list else true_item
        z_item.append(rand_item_choice)

    z_item_ = interlace(edge_label_index_1_arr.tolist(), z_item)

    edge_label_time = torch.tensor(
        interlace(data[predict_edge[0]].timestamp.tolist(), data[predict_edge[0]].timestamp.tolist()))

    edge_label = torch.tensor(interlace([1] * len(edge_label_index[0]), [0] * len(edge_label_index[0])))
    edge_label_index = torch.tensor([z_user, z_item_])

    print("edge_label_time :", edge_label_time)
    print('edge_label_time.size() :', edge_label_time.size())
    print('edge_label_index[0].size() :', edge_label_index[0].size())
    print('edge_label_index[1].size() :', edge_label_index[1].size())

    # After negative sampling
    edge_label_time = edge_label_time.long()

    edge_label_index_0_arr = edge_label_index[0].numpy()
    edge_label_index_1_arr = edge_label_index[1].numpy()

    # Filter the edges based on the time
    ts_min = np.percentile(edge_label_time, percentile_min)
    ts_max = np.percentile(edge_label_time, percentile_max)

    ts_filtered_edge = (edge_label_time > ts_min) & (edge_label_time <= ts_max)

    edge_label_index = torch.Tensor(np.array([edge_label_index_0_arr[ts_filtered_edge],
                                              edge_label_index_1_arr[ts_filtered_edge]]))

    edge_label = edge_label[ts_filtered_edge]
    edge_label_time = edge_label_time[ts_filtered_edge]

    loader = LinkNeighborLoader(
        data=data,
        num_neighbors=num_neigh,
        neg_sampling_ratio=0,
        edge_label_index=((predict_edge[0], predict_edge[1], predict_edge[2]), edge_label_index),
        temporal_strategy=strategy,
        time_attr='timestamp',
        edge_label_time=edge_label_time,
        edge_label=edge_label,
        batch_size=batch_size,
        transform=transform,
        shuffle=False
    )
    return loader


def traintestval_loader(data,
                        num_neigh=None,
                        predict_edge=("candidature", "has_application", "job"),
                        strat='uniform',
                        temporal=False,
                        batch_size=128,
                        neg_sampling_ratio=0):
    if num_neigh is None:
        num_neigh = [20, 10]
    # Define seed edges
    edge_label_index = data[predict_edge[0], predict_edge[1], predict_edge[2]].edge_label_index
    edge_label = data[predict_edge[0], predict_edge[1], predict_edge[2]].edge_label

    edge_label_time = data[predict_edge[0]].timestamp[
        data[predict_edge[0], predict_edge[1], predict_edge[2]].edge_label_index[0]]
    edge_label_time = edge_label_time.long()

    if temporal:
        loader = LinkNeighborLoader(
            data=data,
            num_neighbors=num_neigh,
            neg_sampling_ratio=neg_sampling_ratio,
            edge_label_index=((predict_edge[0], predict_edge[1], predict_edge[2]), edge_label_index),
            temporal_strategy=strat,
            time_attr='timestamp',
            edge_label_time=edge_label_time,
            edge_label=edge_label,
            batch_size=batch_size,
            shuffle=False
        )
    else:
        loader = LinkNeighborLoader(
            data=data,
            num_neighbors=num_neigh,
            neg_sampling_ratio=neg_sampling_ratio,
            edge_label_index=((predict_edge[0], predict_edge[1], predict_edge[2]), edge_label_index),
            edge_label=edge_label,
            batch_size=batch_size,
            shuffle=True
        )

    return loader


def train(model,
          train_loader,
          val_loader,
          test_loader,
          all_data,
          name_experiment,
          strategy,
          output_dir,
          wd=0.00001,
          predict_edge=("candidature", "has_application", "job"),
          num_neigh=None,
          use_ts_loader=False,
          save_model_bool=False,
          train_metric=0,
          max_epoch=2000,
          max_val_decrease=20,
          lr=0.0001,
          hyperparameters=None,
          error_analysis=False
          ):
    """
        Train the model.
        Args:
            output_dir:
            strategy:
            error_analysis:
            max_val_decrease:
            max_epoch:
            hyperparameters:
            train_metric:
            save_model_bool:
            use_ts_loader:
            model (torch.nn.Module): The model instance.
            train_loader : torch_geometric.loader.LinkNeighborLoader
                instance.
            test_loader : torch_geometric.loader.LinkNeighborLoader
            val_loader : torch_geometric.loader.LinkNeighborLoader
            all_data : torch_geometric.data.HeteroData
            name_experiment (str): The name of the experiment.
            wd (float): Weight decay value.
            predict_edge (tuple): The edge to predict.
            num_neigh (list): The number of neighbors to sample.
        Returns:
            The trained model."""
    if num_neigh is None:
        num_neigh = [20, 10]
    if hyperparameters is None:
        hyperparameters = {}

    writer = SummaryWriter(output_dir + f'/runs/{name_experiment}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    model.train()

    if train_metric == 0:
        best_metric_early_stop = 100
    else:
        best_metric_early_stop = -1

    # old way of early stopping
    counter = 0

    auc, precision, recall, acc, f1, val_loss = evaluate(model, val_loader, predict_edge=predict_edge)

    if writer is not None:
        writer.add_scalar('Loss/val', val_loss, 0)
        writer.add_scalar('Metrics/auc', auc, 0)
        writer.add_scalar('Metrics/precision', precision, 0)
        writer.add_scalar('Metrics/recall', recall, 0)
        writer.add_scalar('Metrics/accuracy', acc, 0)
        writer.add_scalar('Metrics/F1', f1, 0)

    for epoch in range(max_epoch):
        if counter >= max_val_decrease:
            break
        total_loss = total_examples = 0

        # about batch size
        all_batch_size = []

        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()

            sampled_data = sampled_data.to(device)
            pred = model(sampled_data)
            ground_truth = sampled_data[predict_edge[0], predict_edge[1], predict_edge[2]].edge_label

            all_batch_size.append(len(ground_truth))

            loss = F.binary_cross_entropy_with_logits(pred.float(), ground_truth.float())
            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        print(
            f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}, [Avg,Max,Min] batch size: [{np.mean(all_batch_size):.4f},{np.max(all_batch_size):.4f},{np.min(all_batch_size):.4f}]")

        auc, precision, recall, acc, f1, val_loss = evaluate(model, val_loader, predict_edge=predict_edge)

        if writer is not None:
            writer.add_scalar('Loss/train', total_loss / total_examples, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/auc', auc, epoch)
            writer.add_scalar('Metrics/precision', precision, epoch)
            writer.add_scalar('Metrics/recall', recall, epoch)
            writer.add_scalar('Metrics/accuracy', acc, epoch)
            writer.add_scalar('Metrics/F1', f1, epoch)

        if train_metric == 0:
            metric_early_stop = val_loss

        elif train_metric == 1:
            metric_early_stop = f1
        elif train_metric == 2:
            metric_early_stop = auc
        else:
            # Should produce an error
            metric_early_stop = None

        if train_metric == 0:
            if val_loss < best_metric_early_stop:
                best_metric_early_stop = metric_early_stop
                counter = 0
                if save_model_bool:
                    # export the model
                    Path(output_dir + f"/models_temp/{name_experiment}").mkdir(parents=True, exist_ok=True)
                    torch.save(model, output_dir + f"/models_temp/{name_experiment}/model.pt")
            else:
                counter += 1
        else:
            if metric_early_stop > best_metric_early_stop:
                best_metric_early_stop = metric_early_stop
                counter = 0
                if save_model_bool:
                    Path(output_dir + f"/models_temp/{name_experiment}").mkdir(parents=True, exist_ok=True)
                    torch.save(model, output_dir + f"/models_temp/{name_experiment}/model.pt")

            else:
                counter += 1

    # re-add hyperparameter
    setting = {"lr": lr}

    setting.update(hyperparameters)

    print("Model trained")

    if save_model_bool:
        # import the best model
        model = torch.load(output_dir + f"/models_temp/{name_experiment}/model.pt")

    auc, precision, recall, acc, f1, val_loss = evaluate(model, test_loader, predict_edge=predict_edge)

    # evaluate model
    precision_at_10, recall_at_10, average_precision_score_at_10, ndcg_score_at_10, mrr, recall_at_10_ts = evaluate_top_k(
        model,
        test_loader,
        all_data,
        name_experiment,
        output_dir=output_dir,
        predict_edge=predict_edge,
        num_neigh=num_neigh,
        ts_loader=use_ts_loader,
        strategy=strategy,
        error_analysis=error_analysis
    )

    writer.add_hparams(
        setting, {
            "hparam/AUC": float(auc),
            "hparam/Precision": float(precision),
            "hparam/Recall": float(recall),
            "hparam/accuracy": float(acc),
            "hparam/F1": float(f1),
            "hparam/val_loss": float(val_loss),
            "precision_at_10": float(precision_at_10),
            "recall_at_10": float(recall_at_10),
            "average_precision_score_at_10": float(average_precision_score_at_10),
            "ndcg_score_at_10": float(ndcg_score_at_10),
            "MRR": float(mrr),
            "recall_at_10_ts": float(recall_at_10_ts),
        })

    print("Model evaluated")
    return model
