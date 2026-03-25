import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from torch_geometric.loader import LinkNeighborLoader

from utils import transform


def evaluate(model, test_loader, verbose=True, predict_edge=("candidature", "has_application", "job")):
    model.eval()
    all_predictions = []
    ground_truths = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iterator = tqdm.tqdm(test_loader) if verbose else test_loader
    total_loss = total_examples = 0

    for sampled_data in iterator:
        with torch.no_grad():
            sampled_data = sampled_data.to(device)
            pred = model(sampled_data)
            all_predictions.append(pred)
            gt = sampled_data[predict_edge[0], predict_edge[1], predict_edge[2]].edge_label
            ground_truths.append(gt)
            loss = F.binary_cross_entropy_with_logits(pred.float(), gt.float())
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

    all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

    false_positive_rate, true_positive_rate, thresholds = roc_curve(ground_truth, all_predictions)
    if verbose:
        print(all_predictions)
        print("false_positive_rate: ", false_positive_rate)
        print("true_positive_rate: ", true_positive_rate)
        print("thresholds: ", thresholds)

    try:
        try:
            auc = roc_auc_score(ground_truth, all_predictions)
        except Exception:
            auc = roc_auc_score(ground_truth, all_predictions, multi_class="ovr")
    except Exception:
        auc = 0

    f1 = f1_score(ground_truth, [int(x > 0) for x in all_predictions])
    prec = precision_score(ground_truth, [int(x > 0) for x in all_predictions])
    acc = accuracy_score(ground_truth, [int(x > 0) for x in all_predictions])
    recall = recall_score(ground_truth, [int(x > 0) for x in all_predictions])

    if verbose:
        print()
        print(f"Validation AUC: {auc:.4f}")
        print(f"Validation Precision: {prec:.4f}")
        print(f"Validation Recall: {recall:.4f}")
        print(f"Validation Accuracy: {acc:.4f}")
        print(f"Validation F1: {f1:.4f}")
        print(f"Validation total loss: {total_loss / total_examples:.4f}")

    model.train()
    return auc, prec, recall, acc, f1, total_loss / total_examples


def evaluate_top_k(
    model,
    test_loader,
    all_data,
    name_experiment,
    predict_edge,
    num_neigh,
    ts_loader,
    strategy,
    output_dir,
    k=10,
    batch_size=3 * 128,
    error_analysis=False,
    verbose=False,
):
    model.eval()

    all_precisions = []
    all_recalls = []
    all_average_precision_scores = []
    all_ndcg = []
    all_rank = []
    error_analysis_results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = test_loader.data
    if verbose:
        print(test_data)

    edges = test_data[predict_edge[0], predict_edge[1], predict_edge[2]]
    if ts_loader:
        test_users = test_loader.input_data.row
        test_items = test_loader.input_data.col
        test_labels = test_loader.input_data.label
    else:
        test_users = edges["edge_label_index"][0]
        test_items = edges["edge_label_index"][1]
        test_labels = edges["edge_label"]

    true_edges_user = test_users[test_labels == 1]
    true_edges_job = test_items[test_labels == 1]

    pbar = tqdm.tqdm(list(set(test_users.tolist())))
    counter = 0
    for u in pbar:
        counter += 1
        if counter % 100 == 0 and error_analysis:
            Path(output_dir + f"/error_analysis/{name_experiment}").mkdir(parents=True, exist_ok=True)
            with open(output_dir + f"/error_analysis/{name_experiment}/error_analysis.json", "w") as f:
                json.dump(error_analysis_results, f)

        index_truth = [i for i, j in enumerate(true_edges_user) if j == u]
        index_true_item = true_edges_job[index_truth]
        all_items = list(set(list(all_data[predict_edge[2]]["node_id"])))

        ground_truth = [1 if i in index_true_item else 0 for i in all_items]
        if sum(ground_truth) == 0:
            print("PROBLEM!")
            continue

        edge_label_index = torch.tensor(np.array([np.repeat(u, len(all_items)), np.array(all_items)]))
        edge_label = torch.Tensor(ground_truth)
        if ts_loader:
            edge_label_time = torch.tensor(np.repeat(test_data["candidature"]["timestamp"][int(u)].item(), len(all_items)))
        else:
            raise NotImplementedError

        ranking_loader = LinkNeighborLoader(
            data=test_data,
            num_neighbors=num_neigh,
            edge_label_index=((predict_edge[0], predict_edge[1], predict_edge[2]), edge_label_index),
            edge_label=edge_label,
            edge_label_time=edge_label_time,
            temporal_strategy=strategy,
            time_attr="timestamp",
            neg_sampling_ratio=0,
            batch_size=batch_size,
            transform=transform,
            shuffle=False,
        )

        local_pred = []
        local_ground_truth = []
        local_all_jobs = []

        for sampled_data in ranking_loader:
            pred = model(sampled_data.to(device))
            local_pred += pred.tolist()
            local_ground_truth += sampled_data[predict_edge[0], predict_edge[1], predict_edge[2]]["edge_label"]
            local_all_jobs += sampled_data[predict_edge[2]]["node_id"][
                sampled_data[predict_edge[0], predict_edge[1], predict_edge[2]]["edge_label_index"][1]
            ].tolist()

        if predict_edge[0] == "candidature":
            true_user = test_data["candidate", "applied_with", "candidature"]["edge_index"].transpose(0, 1)[
                test_data["candidate", "applied_with", "candidature"]["edge_index"][1] == u
            ][0][0].item()
        else:
            true_user = int(u)

        local_ground_truth = [x.item() for x in local_ground_truth]
        sorted_predictions = sorted(zip(local_pred, local_ground_truth, local_all_jobs), key=lambda x: -x[0])

        if error_analysis:
            error_analysis_results.append(
                [true_user, [x[0] for x in sorted_predictions], [x[1] for x in sorted_predictions], [x[2] for x in sorted_predictions]]
            )

        top_k = sorted_predictions[:k]
        top_k_gt = [x[1] for x in top_k]
        precision = sum(top_k_gt) / len(top_k_gt)
        recall = sum(top_k_gt) / sum(local_ground_truth)

        local_ranks = [i + 1 for i, x in enumerate(sorted_predictions) if x[1] == 1]
        all_rank += local_ranks

        ndcg = 0
        for i, x in enumerate(top_k_gt):
            if x == 1:
                ndcg = 1 / (i + 1)
                break

        average_precision = 0
        count_top = 1
        for i, x in enumerate(top_k_gt):
            if x == 1:
                average_precision += count_top / (i + 1)
                count_top += 1

        average_precision = average_precision / sum(local_ground_truth) if sum(local_ground_truth) != 0 else 0

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_average_precision_scores.append(average_precision)
        all_ndcg.append(ndcg)

        pbar.set_description(
            "R@10: "
            + "%.5f" % (sum(all_recalls) / len(all_recalls))
            + " MAP@10: "
            + "%.5f" % (sum(all_average_precision_scores) / len(all_average_precision_scores))
            + " MRR: "
            + "%.5f" % (sum([1 / r for r in all_rank]) / len(all_rank))
            + " NDCG@10: "
            + "%.5f" % (sum(all_ndcg) / len(all_ndcg))
        )

    precision_at_10 = sum(all_precisions) / len(all_precisions)
    recall_at_10 = sum(all_recalls) / len(all_recalls)
    average_precision_score_at_10 = sum(all_average_precision_scores) / len(all_average_precision_scores)
    ndcg_score_at_10 = sum(all_ndcg) / len(all_ndcg)
    mrr = sum([1 / r for r in all_rank]) / len(all_rank)
    recall_at_10_ts = len([1 for i in all_rank if i <= 10]) / len(all_rank)

    model.train()
    return precision_at_10, recall_at_10, average_precision_score_at_10, ndcg_score_at_10, mrr, recall_at_10_ts
