import numpy as np
import torch


def transform(data):
    candidacy = data["candidature", "has_application", "job"]["edge_label_index"][0].numpy()
    new_array = np.array(
        [
            x.numpy()
            for x in data["candidature", "has_application", "job"]["edge_index"].transpose(0, 1)
            if x[0].item() not in candidacy
        ]
    )
    if len(new_array.shape) == 1:
        data["candidature", "has_application", "job"]["edge_index"] = torch.empty((2, 0), dtype=torch.int64)
    else:
        data["candidature", "has_application", "job"]["edge_index"] = torch.LongTensor(new_array).transpose(0, 1)

    new_rev_array = np.array(
        [
            x.numpy()
            for x in data["job", "rev_has_application", "candidature"]["edge_index"].transpose(0, 1)
            if x[1].item() not in candidacy
        ]
    )
    if len(new_rev_array.shape) == 1:
        data["job", "rev_has_application", "candidature"]["edge_index"] = torch.empty((2, 0), dtype=torch.int64)
    else:
        data["job", "rev_has_application", "candidature"]["edge_index"] = torch.LongTensor(new_rev_array).transpose(
            0, 1
        )

    for n_from, rel, n_to in data.edge_types:
        rev = rel[4:] if rel.startswith("rev_") else "rev_" + rel
        if rel.startswith("rev_") and len(data[n_to, rev, n_from]["edge_index"]) == 0 and len(
            data[n_from, rel, n_to]["edge_index"]
        ) != 0:
            data[n_to, rev, n_from]["edge_index"] = torch.LongTensor(
                np.array(
                    [
                        data[n_from, rel, n_to]["edge_index"][1].numpy(),
                        data[n_from, rel, n_to]["edge_index"][0].numpy(),
                    ]
                )
            )
    return data
