import numpy as np
import torch


def transform(data):
    forward_label_edge = ("candidature", "has_application", "job")
    reverse_label_edge = ("candidate", "applied_with", "candidature")

    if (
        forward_label_edge in data.edge_types
        and "edge_label_index" in data[forward_label_edge]
        and "edge_index" in data[forward_label_edge]
    ):
        candidature_seeds = data[forward_label_edge]["edge_label_index"][0].numpy()
        new_array = np.array(
            [
                x.numpy()
                for x in data[forward_label_edge]["edge_index"].transpose(0, 1)
                if x[0].item() not in candidature_seeds
            ]
        )
        if len(new_array.shape) == 1:
            data[forward_label_edge]["edge_index"] = torch.empty((2, 0), dtype=torch.int64)
        else:
            data[forward_label_edge]["edge_index"] = torch.LongTensor(new_array).transpose(0, 1)

        rev_edge = ("job", "rev_has_application", "candidature")
        if rev_edge in data.edge_types and "edge_index" in data[rev_edge]:
            new_rev_array = np.array(
                [
                    x.numpy()
                    for x in data[rev_edge]["edge_index"].transpose(0, 1)
                    if x[1].item() not in candidature_seeds
                ]
            )
            if len(new_rev_array.shape) == 1:
                data[rev_edge]["edge_index"] = torch.empty((2, 0), dtype=torch.int64)
            else:
                data[rev_edge]["edge_index"] = torch.LongTensor(new_rev_array).transpose(0, 1)

    elif (
        reverse_label_edge in data.edge_types
        and "edge_label_index" in data[reverse_label_edge]
        and "edge_index" in data[reverse_label_edge]
    ):
        candidature_seeds = data[reverse_label_edge]["edge_label_index"][1].numpy()
        new_array = np.array(
            [
                x.numpy()
                for x in data[reverse_label_edge]["edge_index"].transpose(0, 1)
                if x[1].item() not in candidature_seeds
            ]
        )
        if len(new_array.shape) == 1:
            data[reverse_label_edge]["edge_index"] = torch.empty((2, 0), dtype=torch.int64)
        else:
            data[reverse_label_edge]["edge_index"] = torch.LongTensor(new_array).transpose(0, 1)

        rev_edge = ("candidature", "rev_applied_with", "candidate")
        if rev_edge in data.edge_types and "edge_index" in data[rev_edge]:
            new_rev_array = np.array(
                [
                    x.numpy()
                    for x in data[rev_edge]["edge_index"].transpose(0, 1)
                    if x[0].item() not in candidature_seeds
                ]
            )
            if len(new_rev_array.shape) == 1:
                data[rev_edge]["edge_index"] = torch.empty((2, 0), dtype=torch.int64)
            else:
                data[rev_edge]["edge_index"] = torch.LongTensor(new_rev_array).transpose(0, 1)

    for n_from, rel, n_to in data.edge_types:
        rev = rel[4:] if rel.startswith("rev_") else "rev_" + rel
        if (
            rel.startswith("rev_")
            and (n_to, rev, n_from) in data.edge_types
            and "edge_index" in data[n_to, rev, n_from]
            and "edge_index" in data[n_from, rel, n_to]
            and len(data[n_to, rev, n_from]["edge_index"]) == 0
            and len(data[n_from, rel, n_to]["edge_index"]) != 0
        ):
            data[n_to, rev, n_from]["edge_index"] = torch.LongTensor(
                np.array(
                    [
                        data[n_from, rel, n_to]["edge_index"][1].numpy(),
                        data[n_from, rel, n_to]["edge_index"][0].numpy(),
                    ]
                )
            )
    return data
