from __future__ import annotations

from datetime import datetime
from typing import Any

import torch
from torch_geometric.loader import LinkNeighborLoader

from inference_types import GraphMappings, RankedCandidate, RankedJob
from utils import transform


def _print_ranking_summary(
    *,
    predict_edge: tuple[str, str, str],
    candidature_idx: int,
    num_jobs: int,
    num_jobs_before_filter: int,
    use_temporal: bool,
    candidature_timestamp: int | None,
    job_recency_days: int | None,
) -> None:
    print("\nRanking summary")
    print(f"  Predict edge: {predict_edge}")
    print(f"  Candidature index used for ranking: {candidature_idx}")
    print(f"  Number of jobs scored: {num_jobs}")
    print(f"  Number of jobs before temporal filtering: {num_jobs_before_filter}")
    print(f"  Temporal sampling enabled: {use_temporal}")
    print(f"  Job recency window (days): {job_recency_days}")
    if candidature_timestamp is not None:
        print(f"  Candidature timestamp: {datetime.fromtimestamp(candidature_timestamp).isoformat(sep=' ')}")


def _resolve_ranking_config(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    return {
        "predict_edge": config.get("predict_edge", ("candidature", "has_application", "job")),
        "num_neigh": config.get("num_neigh", [20, 10]),
        "strategy": config.get("strategy", "uniform"),
        "batch_size": config.get("eval_batch_size", 3 * 128),
        "use_temporal": config.get("use_temporal", True),
        "job_recency_days": config.get("job_recency_days", 365),
    }


def rank_jobs_for_candidature(
    *,
    model: Any,
    data: Any,
    candidature_idx: int,
    config: dict[str, Any] | None = None,
    mappings: GraphMappings | None = None,
    top_k: int = 10,
) -> list[RankedJob]:
    """
    Score one temporary candidature node against all jobs and return top results.
    """
    if model is None:
        raise ValueError("`model` must be loaded before ranking.")
    if data is None:
        raise ValueError("`data` must be a valid HeteroData instance.")

    ranking_config = _resolve_ranking_config(config)
    predict_edge = ranking_config["predict_edge"]
    device = next(model.parameters()).device

    all_job_indices_all = data[predict_edge[2]].node_id.clone().long()
    if all_job_indices_all.numel() == 0:
        return []

    all_job_indices = all_job_indices_all
    candidature_timestamp: int | None = None
    if ranking_config["use_temporal"]:
        candidature_timestamp = int(data[predict_edge[0]].timestamp[int(candidature_idx)].item())
        job_timestamps = data[predict_edge[2]].timestamp[all_job_indices_all].long()
        valid_job_mask = job_timestamps <= candidature_timestamp

        job_recency_days = ranking_config["job_recency_days"]
        if job_recency_days is not None:
            min_allowed_timestamp = candidature_timestamp - int(job_recency_days) * 24 * 60 * 60
            valid_job_mask = valid_job_mask & (job_timestamps >= min_allowed_timestamp)

        all_job_indices = all_job_indices_all[valid_job_mask]
        if all_job_indices.numel() == 0:
            return []

    candidature_column = torch.full(
        (all_job_indices.numel(),),
        int(candidature_idx),
        dtype=torch.long,
    )
    edge_label_index = torch.stack([candidature_column, all_job_indices], dim=0)
    edge_label = torch.zeros(all_job_indices.numel(), dtype=torch.float)

    loader_kwargs: dict[str, Any] = {
        "data": data,
        "num_neighbors": ranking_config["num_neigh"],
        "edge_label_index": (predict_edge, edge_label_index),
        "edge_label": edge_label,
        "neg_sampling_ratio": 0,
        "batch_size": ranking_config["batch_size"],
        "shuffle": False,
    }

    if ranking_config["use_temporal"]:
        edge_label_time = torch.full(
            (all_job_indices.numel(),),
            candidature_timestamp,
            dtype=torch.long,
        )
        loader_kwargs.update(
            {
                "edge_label_time": edge_label_time,
                "temporal_strategy": ranking_config["strategy"],
                "time_attr": "timestamp",
                "transform": transform,
            }
        )

    _print_ranking_summary(
        predict_edge=predict_edge,
        candidature_idx=int(candidature_idx),
        num_jobs=int(all_job_indices.numel()),
        num_jobs_before_filter=int(all_job_indices_all.numel()),
        use_temporal=ranking_config["use_temporal"],
        candidature_timestamp=candidature_timestamp,
        job_recency_days=ranking_config["job_recency_days"],
    )

    ranking_loader = LinkNeighborLoader(**loader_kwargs)

    model.eval()
    scored_jobs: list[tuple[float, int]] = []
    with torch.no_grad():
        for sampled_data in ranking_loader:
            sampled_data = sampled_data.to(device)
            predictions = model(sampled_data).detach().cpu()
            batch_job_indices = sampled_data[predict_edge[2]]["node_id"][
                sampled_data[predict_edge[0], predict_edge[1], predict_edge[2]]["edge_label_index"][1]
            ].detach().cpu()

            for score_tensor, job_idx_tensor in zip(predictions, batch_job_indices):
                scored_jobs.append((float(score_tensor.item()), int(job_idx_tensor.item())))

    scored_jobs.sort(key=lambda item: item[0], reverse=True)
    top_scored_jobs = scored_jobs[:top_k]

    ranked_jobs: list[RankedJob] = []
    job_lookup = mappings.job_idx_to_id if mappings is not None else {}
    for rank, (score, job_idx) in enumerate(top_scored_jobs, start=1):
        ranked_jobs.append(
            RankedJob(
                job_id=job_lookup.get(job_idx, job_idx),
                score=score,
                rank=rank,
                metadata={"job_index": job_idx},
            )
        )

    return ranked_jobs


def rank_candidates_for_job(
    *,
    model: Any,
    data: Any,
    candidature_idx: int,
    config: dict[str, Any] | None = None,
    mappings: GraphMappings | None = None,
    top_k: int = 10,
) -> list[RankedCandidate]:
    """
    Score one temporary job-query candidature node against all candidates.
    """
    if model is None:
        raise ValueError("`model` must be loaded before ranking.")
    if data is None:
        raise ValueError("`data` must be a valid HeteroData instance.")

    ranking_config = _resolve_ranking_config(config)
    predict_edge = ("candidate", "applied_with", "candidature")
    device = next(model.parameters()).device

    all_candidate_indices_all = data[predict_edge[0]].node_id.clone().long()
    if all_candidate_indices_all.numel() == 0:
        return []

    all_candidate_indices = all_candidate_indices_all
    candidature_timestamp: int | None = None
    if ranking_config["use_temporal"]:
        candidature_timestamp = int(data[predict_edge[2]].timestamp[int(candidature_idx)].item())
        candidate_timestamps = data[predict_edge[0]].timestamp[all_candidate_indices_all].long()
        valid_candidate_mask = candidate_timestamps <= candidature_timestamp

        job_recency_days = ranking_config["job_recency_days"]
        if job_recency_days is not None:
            min_allowed_timestamp = candidature_timestamp - int(job_recency_days) * 24 * 60 * 60
            valid_candidate_mask = valid_candidate_mask & (candidate_timestamps >= min_allowed_timestamp)

        all_candidate_indices = all_candidate_indices_all[valid_candidate_mask]
        if all_candidate_indices.numel() == 0:
            return []

    candidature_column = torch.full(
        (all_candidate_indices.numel(),),
        int(candidature_idx),
        dtype=torch.long,
    )
    edge_label_index = torch.stack([all_candidate_indices, candidature_column], dim=0)
    edge_label = torch.zeros(all_candidate_indices.numel(), dtype=torch.float)

    loader_kwargs: dict[str, Any] = {
        "data": data,
        "num_neighbors": ranking_config["num_neigh"],
        "edge_label_index": (predict_edge, edge_label_index),
        "edge_label": edge_label,
        "neg_sampling_ratio": 0,
        "batch_size": ranking_config["batch_size"],
        "shuffle": False,
    }

    if ranking_config["use_temporal"]:
        edge_label_time = torch.full(
            (all_candidate_indices.numel(),),
            candidature_timestamp,
            dtype=torch.long,
        )
        loader_kwargs.update(
            {
                "edge_label_time": edge_label_time,
                "temporal_strategy": ranking_config["strategy"],
                "time_attr": "timestamp",
            }
        )

    print("\nCandidate ranking summary")
    print(f"  Predict edge: {predict_edge}")
    print(f"  Query candidature index used for ranking: {candidature_idx}")
    print(f"  Number of candidates scored: {int(all_candidate_indices.numel())}")
    print(f"  Number of candidates before temporal filtering: {int(all_candidate_indices_all.numel())}")
    print(f"  Temporal sampling enabled: {ranking_config['use_temporal']}")
    print(f"  Job recency window (days): {ranking_config['job_recency_days']}")
    if candidature_timestamp is not None:
        print(f"  Query timestamp: {datetime.fromtimestamp(candidature_timestamp).isoformat(sep=' ')}")

    ranking_loader = LinkNeighborLoader(**loader_kwargs)

    model.eval()
    scored_candidates: list[tuple[float, int]] = []
    original_predict_edge = model.predict_edge
    with torch.no_grad():
        model.predict_edge = predict_edge
        for sampled_data in ranking_loader:
            sampled_data = sampled_data.to(device)
            predictions = model(sampled_data).detach().cpu()
            batch_candidate_indices = sampled_data[predict_edge[0]]["node_id"][
                sampled_data[predict_edge[0], predict_edge[1], predict_edge[2]]["edge_label_index"][0]
            ].detach().cpu()

            for score_tensor, candidate_idx_tensor in zip(predictions, batch_candidate_indices):
                scored_candidates.append((float(score_tensor.item()), int(candidate_idx_tensor.item())))
    model.predict_edge = original_predict_edge

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    top_scored_candidates = scored_candidates[:top_k]

    candidate_lookup = mappings.candidate_idx_to_id if mappings is not None else {}
    ranked_candidates: list[RankedCandidate] = []
    for rank, (score, candidate_idx) in enumerate(top_scored_candidates, start=1):
        ranked_candidates.append(
            RankedCandidate(
                candidate_id=candidate_lookup.get(candidate_idx, candidate_idx),
                score=score,
                rank=rank,
                metadata={
                    "candidate_index": candidate_idx,
                    "query_candidature_index": int(candidature_idx),
                },
            )
        )

    return ranked_candidates
