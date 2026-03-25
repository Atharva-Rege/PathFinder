from pathlib import Path

import torch

from graph_builder import build_graph
from model import Model
from train import train, split_data, traintestval_loader, ts_loader

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "required_data"


def main() -> None:
    config = {
        "data_dir": DEFAULT_DATA_DIR,
        "output_dir": "output",
        "experiment_name": "hardcoded_shortlist_job",
        "predict_edge": ("candidature", "has_application", "job"),
        "use_candidature_node": True,
        "use_temporal": True,
        "use_ts_loader": True,
        "ts_nodes_all": True,
        "remove_feat": False,
        "freeze": False,
        "save_model": True,
        "seed": 42,
        "hidden_channels": 64,
        "lr": 1e-4,
        "wd": 1e-5,
        "batch_norm": "layer_norm",
        "linear_unit": "gelu",
        "num_layers": 3,
        "conv_operator": "sage",
        "strategy": "uniform",
        "num_neigh": [20, 10],
        "batch_size": 128,
        "eval_batch_size": 3 * 128,
        "train_metric": 0,
        "max_epoch": 20,
        "max_val_decrease": 20,
        # model.py expects 9 switches: skill, contract, origin, experience,
        # salary, category, company, concept, time
        "model_list_abl": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        "error_analysis": False,
    }

    torch.manual_seed(config["seed"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph_abl_list = config["model_list_abl"][:8]
    use_time_nodes = bool(config["model_list_abl"][8])

    data = build_graph(
        config["data_dir"],
        abl_list=graph_abl_list,
        candidature_node=config["use_candidature_node"],
        ts_nodes=config["use_temporal"],
        ts_nodes_all=config["ts_nodes_all"],
        ts_attr=use_time_nodes,
        error_analysis=str(output_dir / "error_analysis" / config["experiment_name"])
        if config["error_analysis"]
        else None,
    )
    print("Graph built")

    model = Model(
        hidden_channels=config["hidden_channels"],
        data=data,
        remove_feat=config["remove_feat"],
        freeze=config["freeze"],
        list_abl=config["model_list_abl"],
        predict_edge=config["predict_edge"],
        batch_norm=config["batch_norm"],
        linear_unit_label=config["linear_unit"],
        num_layers=config["num_layers"],
        conv_operator=config["conv_operator"],
    ).to(device)
    print("Model instantiated")

    if config["use_ts_loader"]:
        train_loader = ts_loader(
            data,
            0,
            80,
            num_neigh=config["num_neigh"],
            predict_edge=config["predict_edge"],
            strategy=config["strategy"],
            batch_size=config["batch_size"],
        )
        print("Train loader loaded")

        test_loader = ts_loader(
            data,
            90,
            100,
            num_neigh=config["num_neigh"],
            predict_edge=config["predict_edge"],
            strategy=config["strategy"],
            batch_size=config["eval_batch_size"],
        )
        print("Test loader loaded")

        val_loader = ts_loader(
            data,
            80,
            90,
            num_neigh=config["num_neigh"],
            predict_edge=config["predict_edge"],
            strategy=config["strategy"],
            batch_size=config["eval_batch_size"],
        )
        print("Val loader loaded")
    else:
        train_data, val_data, test_data = split_data(
            data,
            predict_edge=config["predict_edge"],
        )
        print("Data split")

        train_loader = traintestval_loader(
            train_data,
            num_neigh=config["num_neigh"],
            predict_edge=config["predict_edge"],
            strat=config["strategy"],
            temporal=config["use_temporal"],
            batch_size=config["batch_size"],
            neg_sampling_ratio=1,
        )
        print("Train loader loaded")

        test_loader = traintestval_loader(
            test_data,
            num_neigh=config["num_neigh"],
            predict_edge=config["predict_edge"],
            strat=config["strategy"],
            temporal=config["use_temporal"],
            batch_size=config["eval_batch_size"],
            neg_sampling_ratio=1,
        )
        print("Test loader loaded")

        val_loader = traintestval_loader(
            val_data,
            num_neigh=config["num_neigh"],
            predict_edge=config["predict_edge"],
            strat=config["strategy"],
            temporal=config["use_temporal"],
            batch_size=config["eval_batch_size"],
            neg_sampling_ratio=1,
        )
        print("Val loader loaded")

    train(
        model,
        train_loader,
        val_loader,
        test_loader,
        data,
        config["experiment_name"],
        strategy=config["strategy"],
        output_dir=str(output_dir),
        wd=config["wd"],
        predict_edge=config["predict_edge"],
        num_neigh=config["num_neigh"],
        use_ts_loader=config["use_ts_loader"],
        save_model_bool=config["save_model"],
        train_metric=config["train_metric"],
        max_epoch=config["max_epoch"],
        max_val_decrease=config["max_val_decrease"],
        lr=config["lr"],
        hyperparameters=config,
        error_analysis=config["error_analysis"],
    )
    print("Training complete")


if __name__ == "__main__":
    main()
