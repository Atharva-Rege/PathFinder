from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from graph_builder import build_graph, load_data, load_nodes
from inference_types import (
    GraphInsertionSummary,
    GraphMappings,
    NormalizedCandidateInput,
    NormalizedJobInput,
    RuntimeArtifacts,
)

from graph_persistence import load_graph, load_mappings, save_graph, save_mappings
import os
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = MODULE_DIR / "required_data"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "output"


DEFAULT_RUNTIME_CONFIG: dict[str, Any] = {
    "data_dir": DEFAULT_DATA_DIR,
    "output_dir": DEFAULT_OUTPUT_DIR,
    "experiment_name": "hardcoded_shortlist_job_notebook",
    "predict_edge": ("candidature", "has_application", "job"),
    "sentence_transformer_model": "sentence-transformers/all-MiniLM-L6-v2",
    "use_candidature_node": True,
    "use_temporal": True,
    "ts_nodes_all": True,
    "num_neigh": [20, 10],
    "strategy": "uniform",
    "eval_batch_size": 3 * 128,
    "job_recency_days": 365,
    "model_list_abl": [1, 1, 1, 1, 1, 1, 1, 1, 1],
}


def _merge_runtime_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_RUNTIME_CONFIG)
    if config is not None:
        merged.update(config)
    return merged


def _hydrate_mappings(mappings: GraphMappings) -> GraphMappings:
    """
    Backfill fields that may be missing from older persisted GraphMappings objects.
    """
    if not hasattr(mappings, "candidate_idx_to_id"):
        mappings.candidate_idx_to_id = {}
    if not hasattr(mappings, "candidate_id_to_idx"):
        mappings.candidate_id_to_idx = {}
    if not hasattr(mappings, "job_idx_to_id"):
        mappings.job_idx_to_id = {}
    if not hasattr(mappings, "job_id_to_idx"):
        mappings.job_id_to_idx = {}

    if not mappings.candidate_id_to_idx and mappings.candidate_idx_to_id:
        mappings.candidate_id_to_idx = {
            candidate_id: candidate_idx
            for candidate_idx, candidate_id in mappings.candidate_idx_to_id.items()
        }
    return mappings


def _build_graph_mappings(data_dir: str | Path, config: dict[str, Any]) -> GraphMappings:
    (
        candidate,
        candidate_skills,
        job_candidates,
        job_skills,
        candidate_pre,
        job,
        job_pre,
        skill,
        hierarchy,
        candidate_exp,
        job_exp,
        candidate_origin,
        candidate_contract,
        job_contract,
        job_category,
        candidate_salary,
        job_salary,
        job_company,
        job_company_full,
        skill_concept,
    ) = load_data(data_dir)
    _ = candidate, job, skill, hierarchy, job_candidates, candidate_skills, job_skills, job_company

    (
        unique_user_id,
        unique_skill_id,
        unique_job_id,
        unique_candidature_id,
        unique_contract_id,
        unique_exp_id,
        unique_origin_id,
        unique_salary_id,
        unique_category_id,
        unique_company_id,
        unique_concept_id,
        unique_time_id,
    ) = load_nodes(
        candidate_skills,
        job_candidates,
        job_skills,
        candidate_pre,
        candidate_exp,
        job_exp,
        candidate_origin,
        candidate_contract,
        job_contract,
        job_category,
        candidate_salary,
        job_salary,
        job_company,
        job_company_full,
        skill_concept,
        job_pre,
    )
    _ = unique_user_id, unique_candidature_id, unique_concept_id

    abl = config["model_list_abl"]
    use_time_nodes = bool(abl[8])

    return GraphMappings(
        skill_to_idx={str(row.skillID): int(row.mappedID) for row in unique_skill_id.itertuples()} if abl[0] else {},
        contract_to_idx={str(row.contractID): int(row.mappedID) for row in unique_contract_id.itertuples()} if abl[1] else {},
        origin_to_idx={str(row.originID): int(row.mappedID) for row in unique_origin_id.itertuples()} if abl[2] else {},
        experience_to_idx={str(row.expID): int(row.mappedID) for row in unique_exp_id.itertuples()} if abl[3] else {},
        salary_to_idx={str(row.salaryID): int(row.mappedID) for row in unique_salary_id.itertuples()} if abl[4] else {},
        category_to_idx={str(row.categoryID): int(row.mappedID) for row in unique_category_id.itertuples()} if abl[5] else {},
        company_to_idx={str(row.companyID): int(row.mappedID) for row in unique_company_id.itertuples()} if abl[6] else {},
        time_to_idx={int(row.timeID): int(row.mappedID) for row in unique_time_id.itertuples()} if use_time_nodes else {},
        time_base_yearmonth=(
            min(
                datetime.fromtimestamp(int(row.timestamp)).year * 12
                + datetime.fromtimestamp(int(row.timestamp)).month
                - 1
                for row in unique_time_id.itertuples()
            )
            if use_time_nodes and len(unique_time_id) > 0
            else None
        ),
        candidate_idx_to_id={int(row.mappedID): row.nameID for row in unique_user_id.itertuples()},
        candidate_id_to_idx={row.nameID: int(row.mappedID) for row in unique_user_id.itertuples()},
        job_idx_to_id={int(row.mappedID): row.jobID for row in unique_job_id.itertuples()},
        job_id_to_idx={row.jobID: int(row.mappedID) for row in unique_job_id.itertuples()},
    )


def _default_model_path(config: dict[str, Any]) -> str:
    return str(Path(config["output_dir"]) / "models_temp" / config["experiment_name"] / "model.pt")


def _graph_store_dir(config: dict[str, Any]) -> Path:
    return Path(config["output_dir"]) / "graph_store" / config["experiment_name"]


def _graph_snapshot_path(config: dict[str, Any]) -> Path:
    return _graph_store_dir(config) / "graph.pt"


def _mappings_path(config: dict[str, Any]) -> Path:
    return _graph_store_dir(config) / "mappings.pt"


def _load_text_encoder(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for candidate and job text encoding."
        ) from exc

    return SentenceTransformer(model_name)


def _encode_text(
    text: str,
    *,
    text_encoder: Any | None,
    feature_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if text_encoder is None:
        raise ValueError("`text_encoder` must be loaded before adding node features.")

    encoded = text_encoder.encode(text or "", convert_to_numpy=True)
    features = torch.tensor(encoded, dtype=dtype)
    if features.numel() != feature_dim:
        raise ValueError(
            f"Feature dimension mismatch: expected {feature_dim}, got {features.numel()}."
        )
    return features


def _append_1d_tensor(tensor: torch.Tensor, value: int | float, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if dtype is None:
        dtype = tensor.dtype
    value_tensor = torch.tensor([value], dtype=dtype, device=tensor.device)
    return torch.cat([tensor, value_tensor], dim=0)


def _append_2d_row(tensor: torch.Tensor, row: torch.Tensor) -> torch.Tensor:
    row = row.to(device=tensor.device, dtype=tensor.dtype)
    if row.dim() == 1:
        row = row.unsqueeze(0)
    return torch.cat([tensor, row], dim=0)


def _append_edge(
    data: HeteroData,
    edge_type: tuple[str, str, str],
    src_idx: int,
    dst_idx: int,
) -> None:
    edge_tensor = torch.tensor([[src_idx], [dst_idx]], dtype=torch.long)
    if "edge_index" in data[edge_type]:
        data[edge_type].edge_index = torch.cat([data[edge_type].edge_index, edge_tensor], dim=1)
    else:
        data[edge_type].edge_index = edge_tensor


def _add_bidirectional_edge(
    data: HeteroData,
    forward_edge: tuple[str, str, str],
    reverse_edge: tuple[str, str, str],
    src_idx: int,
    dst_idx: int,
) -> None:
    _append_edge(data, forward_edge, src_idx, dst_idx)
    _append_edge(data, reverse_edge, dst_idx, src_idx)


def _timestamp_to_absolute_yearmonth(timestamp: int) -> int:
    dt_value = datetime.fromtimestamp(int(timestamp))
    return dt_value.year * 12 + dt_value.month - 1


def _ensure_graph_store_initialized(
    *,
    data_dir: str | Path,
    config: dict[str, Any],
) -> tuple[HeteroData, GraphMappings]:
    snapshot_path = _graph_snapshot_path(config)
    mappings_path = _mappings_path(config)

    if snapshot_path.exists() and mappings_path.exists():
        try:
            graph = torch.load(snapshot_path, map_location="cpu", weights_only=False)
        except TypeError:
            graph = torch.load(snapshot_path, map_location="cpu")
        try:
            mappings = torch.load(mappings_path, map_location="cpu", weights_only=False)
        except TypeError:
            mappings = torch.load(mappings_path, map_location="cpu")
        return graph, _hydrate_mappings(mappings)

    graph_abl_list = config["model_list_abl"][:8]
    use_time_nodes = bool(config["model_list_abl"][8])
    graph = build_graph(
        data_dir,
        abl_list=graph_abl_list,
        candidature_node=config["use_candidature_node"],
        ts_nodes=config["use_temporal"],
        ts_nodes_all=config["ts_nodes_all"],
        ts_attr=use_time_nodes,
    )
    mappings = _build_graph_mappings(data_dir, config)
    save_runtime_state(data=graph, mappings=mappings, config=config)
    return graph, mappings


def save_runtime_state(
    *,
    data: HeteroData,
    mappings: GraphMappings,
    config: dict[str, Any],
) -> None:
    store_dir = _graph_store_dir(config)
    store_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, _graph_snapshot_path(config))
    torch.save(mappings, _mappings_path(config))


def load_runtime_artifacts(
    *,
    data_dir: str | Path,
    model_path: str | None = None,
    config: dict[str, Any] | None = None,
) -> RuntimeArtifacts:

    runtime_config = _merge_runtime_config(config)
    runtime_config["data_dir"] = data_dir

    graph, mappings = _ensure_graph_store_initialized(
        data_dir=data_dir,
        config=runtime_config,
    )

    resolved_model_path = model_path or _default_model_path(runtime_config)

    try:
        model = torch.load(resolved_model_path, map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(resolved_model_path, map_location="cpu")

    model = extend_model_for_new_nodes(model, graph)
    model.eval()

    text_encoder = _load_text_encoder(runtime_config["sentence_transformer_model"])

    return RuntimeArtifacts(
        data=graph,
        model=model,
        text_encoder=text_encoder,
        mappings=mappings,
        config=runtime_config,
    )


def ensure_time_node(
    data: HeteroData,
    timestamp: int,
    *,
    mappings: GraphMappings | None = None,
    copy_data: bool = False,
) -> tuple[HeteroData, int | None, bool]:
    """
    Ensure the month bucket for `timestamp` exists in the graph.
    """
    if data is None:
        raise ValueError("`data` must be a valid HeteroData instance.")
    if "time" not in data.node_types:
        return data, None, False
    if mappings is None:
        mappings = GraphMappings()

    if copy_data:
        data = deepcopy(data)

    absolute_yearmonth = _timestamp_to_absolute_yearmonth(timestamp)
    if mappings.time_base_yearmonth is None:
        mappings.time_base_yearmonth = absolute_yearmonth

    derived_yearmonth = absolute_yearmonth - mappings.time_base_yearmonth
    existing_idx = mappings.time_to_idx.get(derived_yearmonth)
    if existing_idx is not None:
        return data, existing_idx, False

    time_idx = int(data["time"].node_id.numel())
    data["time"].node_id = _append_1d_tensor(data["time"].node_id, time_idx, dtype=torch.long)

    if "x" in data["time"]:
        time_feature = torch.tensor([derived_yearmonth], dtype=data["time"].x.dtype)
        data["time"].x = _append_2d_row(data["time"].x, time_feature)

    if "timestamp" in data["time"]:
        month_start = datetime(absolute_yearmonth // 12, absolute_yearmonth % 12 + 1, 1)
        data["time"].timestamp = _append_1d_tensor(
            data["time"].timestamp,
            int(month_start.timestamp()),
            dtype=torch.long,
        )

    mappings.time_to_idx[derived_yearmonth] = time_idx
    return data, time_idx, True


def add_candidate_to_graph(
    data: HeteroData,
    candidate: NormalizedCandidateInput,
    *,
    mappings: GraphMappings | None = None,
    text_encoder: Any | None = None,
    copy_data: bool = False,
) -> tuple[HeteroData, int, GraphInsertionSummary]:
    """
    Append a candidate node and connect it to existing attribute nodes.
    """
    if data is None:
        raise ValueError("`data` must be a valid HeteroData instance.")
    if mappings is None:
        mappings = GraphMappings()

    if copy_data:
        data = deepcopy(data)

    candidate_id = candidate.candidate_id or f"candidate_{int(data['candidate'].node_id.numel())}"
    if candidate_id in mappings.candidate_id_to_idx:
        raise ValueError(f"Candidate ID `{candidate_id}` already exists in the persistent graph.")

    data, time_idx, time_created = ensure_time_node(
        data,
        candidate.timestamp,
        mappings=mappings,
        copy_data=False,
    )

    candidate_idx = int(data["candidate"].node_id.numel())
    insertion_summary = GraphInsertionSummary(
        candidate_idx=candidate_idx,
        candidate_time_idx=time_idx,
        time_node_created=time_created,
    )

    feature_dim = int(data["candidate"].x.size(1))
    candidate_features = _encode_text(
        candidate.description,
        text_encoder=text_encoder,
        feature_dim=feature_dim,
        dtype=data["candidate"].x.dtype,
    )
    data["candidate"].x = _append_2d_row(data["candidate"].x, candidate_features)
    data["candidate"].node_id = _append_1d_tensor(data["candidate"].node_id, candidate_idx, dtype=torch.long)
    mappings.candidate_idx_to_id[candidate_idx] = candidate_id
    mappings.candidate_id_to_idx[candidate_id] = candidate_idx

    if "timestamp" in data["candidate"]:
        data["candidate"].timestamp = _append_1d_tensor(
            data["candidate"].timestamp,
            candidate.timestamp,
            dtype=torch.long,
        )

    for skill in candidate.skills:
        skill_idx = mappings.skill_to_idx.get(skill)
        if skill_idx is not None:
            insertion_summary.matched_skill_indices.append(skill_idx)
            _add_bidirectional_edge(
                data,
                ("candidate", "has", "skill"),
                ("skill", "rev_has", "candidate"),
                candidate_idx,
                skill_idx,
            )

    if candidate.contract is not None:
        contract_idx = mappings.contract_to_idx.get(candidate.contract)
        if contract_idx is not None:
            insertion_summary.contract_idx = contract_idx
            _add_bidirectional_edge(
                data,
                ("candidate", "work_on", "contract"),
                ("contract", "rev_work_on", "candidate"),
                candidate_idx,
                contract_idx,
            )

    if candidate.origin is not None:
        origin_idx = mappings.origin_to_idx.get(candidate.origin)
        if origin_idx is not None:
            insertion_summary.origin_idx = origin_idx
            _add_bidirectional_edge(
                data,
                ("candidate", "was_found_on", "origin"),
                ("origin", "rev_was_found_on", "candidate"),
                candidate_idx,
                origin_idx,
            )

    if candidate.experience is not None:
        experience_idx = mappings.experience_to_idx.get(candidate.experience)
        if experience_idx is not None:
            insertion_summary.experience_idx = experience_idx
            _add_bidirectional_edge(
                data,
                ("candidate", "has_gain", "experience"),
                ("experience", "rev_has_gain", "candidate"),
                candidate_idx,
                experience_idx,
            )

    if candidate.salary_category is not None:
        salary_idx = mappings.salary_to_idx.get(candidate.salary_category)
        if salary_idx is not None:
            insertion_summary.salary_idx = salary_idx
            _add_bidirectional_edge(
                data,
                ("candidate", "is_worth", "salary"),
                ("salary", "rev_is_worth", "candidate"),
                candidate_idx,
                salary_idx,
            )

    if time_idx is not None:
        _add_bidirectional_edge(
            data,
            ("candidate", "has_time", "time"),
            ("time", "rev_has_time", "candidate"),
            candidate_idx,
            time_idx,
        )

    return data, candidate_idx, insertion_summary


def add_job_to_graph(
    data: HeteroData,
    job: NormalizedJobInput,
    *,
    mappings: GraphMappings | None = None,
    text_encoder: Any | None = None,
    copy_data: bool = False,
) -> tuple[HeteroData, int, GraphInsertionSummary]:
    """
    Append a job node and connect it to existing attribute nodes.
    """
    if data is None:
        raise ValueError("`data` must be a valid HeteroData instance.")
    if mappings is None:
        mappings = GraphMappings()

    if copy_data:
        data = deepcopy(data)

    if job.job_id in mappings.job_id_to_idx:
        raise ValueError(f"Job ID `{job.job_id}` already exists in the persistent graph.")

    data, time_idx, time_created = ensure_time_node(
        data,
        job.timestamp,
        mappings=mappings,
        copy_data=False,
    )

    job_idx = int(data["job"].node_id.numel())
    insertion_summary = GraphInsertionSummary(
        candidate_idx=-1,
        job_idx=job_idx,
        candidate_time_idx=time_idx,
        time_node_created=time_created,
    )

    feature_dim = int(data["job"].x.size(1))
    job_features = _encode_text(
        job.description,
        text_encoder=text_encoder,
        feature_dim=feature_dim,
        dtype=data["job"].x.dtype,
    )
    data["job"].x = _append_2d_row(data["job"].x, job_features)
    data["job"].node_id = _append_1d_tensor(data["job"].node_id, job_idx, dtype=torch.long)

    if "timestamp" in data["job"]:
        data["job"].timestamp = _append_1d_tensor(
            data["job"].timestamp,
            job.timestamp,
            dtype=torch.long,
        )

    mappings.job_id_to_idx[job.job_id] = job_idx
    mappings.job_idx_to_id[job_idx] = job.job_id

    for skill in job.skills:
        skill_idx = mappings.skill_to_idx.get(skill)
        if skill_idx is not None:
            insertion_summary.matched_skill_indices.append(skill_idx)
            _add_bidirectional_edge(
                data,
                ("job", "has", "skill"),
                ("skill", "rev_has", "job"),
                job_idx,
                skill_idx,
            )

    if job.contract is not None:
        contract_idx = mappings.contract_to_idx.get(job.contract)
        if contract_idx is not None:
            insertion_summary.contract_idx = contract_idx
            _add_bidirectional_edge(
                data,
                ("job", "has_type", "contract"),
                ("contract", "rev_has_type", "job"),
                job_idx,
                contract_idx,
            )

    if job.experience is not None:
        experience_idx = mappings.experience_to_idx.get(job.experience)
        if experience_idx is not None:
            insertion_summary.experience_idx = experience_idx
            _add_bidirectional_edge(
                data,
                ("job", "requires", "experience"),
                ("experience", "rev_requires", "job"),
                job_idx,
                experience_idx,
            )

    if job.salary_category is not None:
        salary_idx = mappings.salary_to_idx.get(job.salary_category)
        if salary_idx is not None:
            insertion_summary.salary_idx = salary_idx
            _add_bidirectional_edge(
                data,
                ("job", "is_worth", "salary"),
                ("salary", "rev_is_worth", "job"),
                job_idx,
                salary_idx,
            )

    if job.category is not None:
        category_idx = mappings.category_to_idx.get(job.category)
        if category_idx is not None:
            insertion_summary.category_idx = category_idx
            _add_bidirectional_edge(
                data,
                ("job", "is_in", "category"),
                ("category", "rev_is_in", "job"),
                job_idx,
                category_idx,
            )

    if job.company is not None:
        company_idx = mappings.company_to_idx.get(job.company)
        if company_idx is not None:
            insertion_summary.company_idx = company_idx
            _add_bidirectional_edge(
                data,
                ("job", "is_for", "company"),
                ("company", "rev_is_for", "job"),
                job_idx,
                company_idx,
            )

    if time_idx is not None:
        _add_bidirectional_edge(
            data,
            ("job", "has_time", "time"),
            ("time", "rev_has_time", "job"),
            job_idx,
            time_idx,
        )

    return data, job_idx, insertion_summary


def add_temporary_candidature(
    data: HeteroData,
    *,
    candidate_idx: int,
    timestamp: int,
    mappings: GraphMappings | None = None,
    copy_data: bool = False,
) -> tuple[HeteroData, int, GraphInsertionSummary]:
    """
    Add a candidature node and link it to the candidate and time nodes.
    """
    if data is None:
        raise ValueError("`data` must be a valid HeteroData instance.")
    if mappings is None:
        mappings = GraphMappings()

    if copy_data:
        data = deepcopy(data)

    data, time_idx, time_created = ensure_time_node(
        data,
        timestamp,
        mappings=mappings,
        copy_data=False,
    )

    candidature_idx = int(data["candidature"].node_id.numel())
    insertion_summary = GraphInsertionSummary(
        candidate_idx=candidate_idx,
        candidature_idx=candidature_idx,
        candidature_time_idx=time_idx,
        time_node_created=time_created,
    )
    data["candidature"].node_id = _append_1d_tensor(
        data["candidature"].node_id,
        candidature_idx,
        dtype=torch.long,
    )

    if "timestamp" in data["candidature"]:
        data["candidature"].timestamp = _append_1d_tensor(
            data["candidature"].timestamp,
            timestamp,
            dtype=torch.long,
        )

    _add_bidirectional_edge(
        data,
        ("candidate", "applied_with", "candidature"),
        ("candidature", "rev_applied_with", "candidate"),
        candidate_idx,
        candidature_idx,
    )

    if time_idx is not None:
        _add_bidirectional_edge(
            data,
            ("candidature", "has_time", "time"),
            ("time", "rev_has_time", "candidature"),
            candidature_idx,
            time_idx,
        )

    return data, candidature_idx, insertion_summary


def add_temporary_job_query_candidature(
    data: HeteroData,
    *,
    job_idx: int,
    timestamp: int,
    mappings: GraphMappings | None = None,
    copy_data: bool = False,
) -> tuple[HeteroData, int, GraphInsertionSummary]:
    """
    Add a query-time candidature node linked to a job and time for candidate ranking.
    """
    if data is None:
        raise ValueError("`data` must be a valid HeteroData instance.")
    if mappings is None:
        mappings = GraphMappings()

    if copy_data:
        data = deepcopy(data)

    data, time_idx, time_created = ensure_time_node(
        data,
        timestamp,
        mappings=mappings,
        copy_data=False,
    )

    candidature_idx = int(data["candidature"].node_id.numel())
    insertion_summary = GraphInsertionSummary(
        candidate_idx=-1,
        candidature_idx=candidature_idx,
        candidature_time_idx=time_idx,
        job_idx=job_idx,
        time_node_created=time_created,
    )
    data["candidature"].node_id = _append_1d_tensor(
        data["candidature"].node_id,
        candidature_idx,
        dtype=torch.long,
    )

    if "timestamp" in data["candidature"]:
        data["candidature"].timestamp = _append_1d_tensor(
            data["candidature"].timestamp,
            timestamp,
            dtype=torch.long,
        )

    _add_bidirectional_edge(
        data,
        ("candidature", "has_application", "job"),
        ("job", "rev_has_application", "candidature"),
        candidature_idx,
        job_idx,
    )

    if time_idx is not None:
        _add_bidirectional_edge(
            data,
            ("candidature", "has_time", "time"),
            ("time", "rev_has_time", "candidature"),
            candidature_idx,
            time_idx,
        )

    return data, candidature_idx, insertion_summary


def attach_candidature_to_job(
    data: HeteroData,
    *,
    candidature_idx: int,
    job_idx: int,
    copy_data: bool = False,
) -> HeteroData:
    """
    Link an existing candidature node to a job node.
    """
    if data is None:
        raise ValueError("`data` must be a valid HeteroData instance.")

    if copy_data:
        data = deepcopy(data)

    _add_bidirectional_edge(
        data,
        ("candidature", "has_application", "job"),
        ("job", "rev_has_application", "candidature"),
        candidature_idx,
        job_idx,
    )
    return data


def attach_candidate_to_candidature(
    data: HeteroData,
    *,
    candidate_idx: int,
    candidature_idx: int,
    copy_data: bool = False,
) -> HeteroData:
    """
    Link an existing candidate node to a candidature node.
    """
    if data is None:
        raise ValueError("`data` must be a valid HeteroData instance.")

    if copy_data:
        data = deepcopy(data)

    _add_bidirectional_edge(
        data,
        ("candidate", "applied_with", "candidature"),
        ("candidature", "rev_applied_with", "candidate"),
        candidate_idx,
        candidature_idx,
    )
    return data


def extend_model_for_new_nodes(model: torch.nn.Module, data: HeteroData) -> torch.nn.Module:
    """
    Resize model embeddings if inference adds brand-new nodes.
    """
    if model is None:
        raise ValueError("`model` must be loaded before extending embeddings.")

    def resize_embedding(embedding: nn.Embedding, target_size: int) -> nn.Embedding:
        current_size, emb_dim = embedding.weight.shape
        if target_size <= current_size:
            return embedding

        resized = nn.Embedding(target_size, emb_dim).to(embedding.weight.device)
        resized.weight.data.zero_()
        resized.weight.data[:current_size] = embedding.weight.data
        return resized

    model.user_emb = resize_embedding(model.user_emb, int(data["candidate"].node_id.numel()))
    model.job_emb = resize_embedding(model.job_emb, int(data["job"].node_id.numel()))

    if hasattr(model, "candidature_emb") and "candidature" in data.node_types:
        model.candidature_emb = resize_embedding(
            model.candidature_emb,
            int(data["candidature"].node_id.numel()),
        )

    if hasattr(model, "time_node_emb") and "time" in data.node_types:
        model.time_node_emb = resize_embedding(
            model.time_node_emb,
            int(data["time"].node_id.numel()),
        )

    return model
