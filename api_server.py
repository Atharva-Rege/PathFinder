from __future__ import annotations

from copy import deepcopy
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from candidate_input import normalize_candidate_input, normalize_job_input
from graph_runtime import (
    add_candidate_to_graph,
    add_job_to_graph,
    add_temporary_candidature,
    add_temporary_job_query_candidature,
    attach_candidate_to_candidature,
    attach_candidature_to_job,
    extend_model_for_new_nodes,
    load_runtime_artifacts,
    save_runtime_state,
    update_candidate_in_graph,
    update_job_in_graph,
)
from inference_types import CandidateInput, JobInput
from interaction_logger import log_interaction
from ranker import rank_candidates_for_job, rank_jobs_for_candidature
from retrain_trigger import should_retrain


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = MODULE_DIR / "required_data"


class CandidatePayload(BaseModel):
    candidate_id: str | None = None
    description: str = ""
    skills: list[str] = Field(default_factory=list)
    contract: str | None = None
    origin: str | None = None
    experience: str | None = None
    salary_current: float | int | None = None
    timestamp: int | None = None


class JobPayload(BaseModel):
    job_id: str
    description: str = ""
    skills: list[str] = Field(default_factory=list)
    contract: str | None = None
    experience: str | None = None
    salary: float | int | None = None
    category: str | None = None
    company: str | None = None
    timestamp: int | None = None


class RecommendJobsRequest(BaseModel):
    candidate_id: str | None = None
    candidate: CandidatePayload | None = None
    job_ids: list[str] | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    persist_graph: bool = True
    persist_selection_rank: int | None = Field(default=None, ge=1, le=100)

    @model_validator(mode="after")
    def validate_candidate_reference(self) -> "RecommendJobsRequest":
        if self.candidate_id is None and self.candidate is None:
            raise ValueError("Provide either `candidate_id` or `candidate`.")
        if self.candidate_id is not None and self.candidate is not None:
            raise ValueError("Provide only one of `candidate_id` or `candidate`.")
        return self


class RecommendCandidatesRequest(BaseModel):
    job_id: str | None = None
    job: JobPayload | None = None
    candidate_ids: list[str] | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    persist_graph: bool = True
    persist_selection_rank: int | None = Field(default=None, ge=1, le=100)

    @model_validator(mode="after")
    def validate_job_reference(self) -> "RecommendCandidatesRequest":
        if self.job_id is None and self.job is None:
            raise ValueError("Provide either `job_id` or `job`.")
        if self.job_id is not None and self.job is not None:
            raise ValueError("Provide only one of `job_id` or `job`.")
        return self


class AddCandidateRequest(BaseModel):
    candidate: CandidatePayload
    persist_graph: bool = True
    force_update: bool = False


class AddJobRequest(BaseModel):
    job: JobPayload
    persist_graph: bool = True
    force_update: bool = False


class InteractionRequest(BaseModel):
    candidate_idx: int | None = None
    candidate_id: str | None = None
    job_idx: int | None = None
    job_id: str | None = None
    score: float
    timestamp: int | None = None

    @model_validator(mode="after")
    def validate_references(self) -> "InteractionRequest":
        has_candidate_idx = self.candidate_idx is not None
        has_candidate_id = self.candidate_id is not None
        has_job_idx = self.job_idx is not None
        has_job_id = self.job_id is not None

        if has_candidate_idx == has_candidate_id:
            raise ValueError("Provide exactly one of `candidate_idx` or `candidate_id`.")
        if has_job_idx == has_job_id:
            raise ValueError("Provide exactly one of `job_idx` or `job_id`.")
        return self


class RuntimeService:
    def __init__(self) -> None:
        self.data_dir = Path(os.getenv("RUNTIME_DATA_DIR", str(DEFAULT_DATA_DIR)))
        self.model_path = os.getenv("RUNTIME_MODEL_PATH")
        self._artifacts = None

    def reload(self):
        try:
            self._artifacts = load_runtime_artifacts(
                data_dir=self.data_dir,
                model_path=self.model_path,
                config=None,
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model weights are not available yet. Set RUNTIME_MODEL_PATH or place model at the default runtime path. "
                    f"Original error: {exc}"
                ),
            ) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load runtime artifacts: {exc}") from exc
        return self._artifacts

    @property
    def artifacts(self):
        if self._artifacts is None:
            return self.reload()
        return self._artifacts


app = FastAPI(title="PathFinder Recommendation API", version="1.0.0")
service = RuntimeService()


def _resolve_candidate_idx(candidate_id: str, mappings) -> int:
    candidate_idx = mappings.candidate_id_to_idx.get(candidate_id)
    if candidate_idx is None:
        raise HTTPException(status_code=404, detail=f"Unknown candidate_id `{candidate_id}`")
    return int(candidate_idx)


def _resolve_job_idx(job_id: str, mappings) -> int:
    job_idx = mappings.job_id_to_idx.get(job_id)
    if job_idx is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id `{job_id}`")
    return int(job_idx)


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        artifacts = service.artifacts
        return {
            "ok": True,
            "model_loaded": True,
            "graph_nodes": {
                "candidate": int(artifacts.data["candidate"].node_id.numel()),
                "job": int(artifacts.data["job"].node_id.numel()),
                "candidature": int(artifacts.data["candidature"].node_id.numel()),
            },
        }
    except HTTPException as exc:
        return {"ok": False, "model_loaded": False, "detail": exc.detail}


@app.post("/candidates")
def add_candidate(request: AddCandidateRequest) -> dict[str, Any]:
    artifacts = service.artifacts

    raw_candidate = CandidateInput(
        candidate_id=request.candidate.candidate_id,
        description=request.candidate.description,
        skills=request.candidate.skills,
        contract=request.candidate.contract,
        origin=request.candidate.origin,
        experience=request.candidate.experience,
        salary_current=request.candidate.salary_current,
        timestamp=request.candidate.timestamp,
    )

    normalized_candidate = normalize_candidate_input(raw_candidate, graph_context=artifacts.mappings)
    candidate_exists = (
        normalized_candidate.candidate_id is not None
        and normalized_candidate.candidate_id in artifacts.mappings.candidate_id_to_idx
    )

    try:
        if request.force_update and candidate_exists:
            data, candidate_idx, summary = update_candidate_in_graph(
                artifacts.data,
                normalized_candidate,
                mappings=artifacts.mappings,
                text_encoder=artifacts.text_encoder,
                copy_data=False,
            )
            action = "updated"
        else:
            data, candidate_idx, summary = add_candidate_to_graph(
                artifacts.data,
                normalized_candidate,
                mappings=artifacts.mappings,
                text_encoder=artifacts.text_encoder,
                copy_data=False,
            )
            action = "inserted"
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    artifacts.data = data
    artifacts.model = extend_model_for_new_nodes(artifacts.model, artifacts.data)

    if request.persist_graph:
        save_runtime_state(data=artifacts.data, mappings=artifacts.mappings, config=artifacts.config)

    return {
        "status": "ok",
        "action": action,
        "candidate_index": int(candidate_idx),
        "candidate_id": artifacts.mappings.candidate_idx_to_id.get(int(candidate_idx)),
        "summary": {
            "normalization": normalized_candidate.to_dict(),
            "insertion": summary.__dict__,
        },
    }


@app.post("/jobs")
def add_job(request: AddJobRequest) -> dict[str, Any]:
    artifacts = service.artifacts

    raw_job = JobInput(
        job_id=request.job.job_id,
        description=request.job.description,
        skills=request.job.skills,
        contract=request.job.contract,
        experience=request.job.experience,
        salary=request.job.salary,
        category=request.job.category,
        company=request.job.company,
        timestamp=request.job.timestamp,
    )

    normalized_job = normalize_job_input(raw_job, graph_context=artifacts.mappings)
    job_exists = normalized_job.job_id in artifacts.mappings.job_id_to_idx

    try:
        if request.force_update and job_exists:
            data, job_idx, summary = update_job_in_graph(
                artifacts.data,
                normalized_job,
                mappings=artifacts.mappings,
                text_encoder=artifacts.text_encoder,
                copy_data=False,
            )
            action = "updated"
        else:
            data, job_idx, summary = add_job_to_graph(
                artifacts.data,
                normalized_job,
                mappings=artifacts.mappings,
                text_encoder=artifacts.text_encoder,
                copy_data=False,
            )
            action = "inserted"
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    artifacts.data = data
    artifacts.model = extend_model_for_new_nodes(artifacts.model, artifacts.data)

    if request.persist_graph:
        save_runtime_state(data=artifacts.data, mappings=artifacts.mappings, config=artifacts.config)

    return {
        "status": "ok",
        "action": action,
        "job_index": int(job_idx),
        "job_id": request.job.job_id,
        "summary": {
            "normalization": normalized_job.to_dict(),
            "insertion": summary.__dict__,
        },
    }


@app.post("/recommend/jobs-for-candidate")
def recommend_jobs_for_candidate(request: RecommendJobsRequest) -> dict[str, Any]:
    artifacts = service.artifacts

    data = deepcopy(artifacts.data)
    insertion_summary: dict[str, Any] | None = None

    if request.candidate_id is not None:
        candidate_idx = _resolve_candidate_idx(request.candidate_id, artifacts.mappings)
    else:
        assert request.candidate is not None
        query_candidate_id = request.candidate.candidate_id
        is_existing_candidate = bool(
            query_candidate_id and query_candidate_id in artifacts.mappings.candidate_id_to_idx
        )
        if not query_candidate_id:
            query_candidate_id = f"__query_candidate_{time.time_ns()}"

        raw_candidate = CandidateInput(
            candidate_id=query_candidate_id,
            description=request.candidate.description,
            skills=request.candidate.skills,
            contract=request.candidate.contract,
            origin=request.candidate.origin,
            experience=request.candidate.experience,
            salary_current=request.candidate.salary_current,
            timestamp=request.candidate.timestamp,
        )

        normalized_candidate = normalize_candidate_input(raw_candidate, graph_context=artifacts.mappings)
        try:
            if is_existing_candidate:
                data, candidate_idx, candidate_summary = update_candidate_in_graph(
                    data,
                    normalized_candidate,
                    mappings=artifacts.mappings,
                    text_encoder=artifacts.text_encoder,
                    copy_data=True,
                )
            else:
                data, candidate_idx, candidate_summary = add_candidate_to_graph(
                    data,
                    normalized_candidate,
                    mappings=artifacts.mappings,
                    text_encoder=artifacts.text_encoder,
                    copy_data=True,
                )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        insertion_summary = {
            "normalization": normalized_candidate.to_dict(),
            "candidate_insertion": candidate_summary.__dict__,
        }

    ranking_timestamp = int(time.time())
    allowed_job_indices: list[int] | None = None
    if request.job_ids is not None:
        resolved_job_indices: list[int] = []
        unknown_job_ids: list[str] = []
        for job_id in request.job_ids:
            job_idx = artifacts.mappings.job_id_to_idx.get(job_id)
            if job_idx is None:
                unknown_job_ids.append(job_id)
            else:
                resolved_job_indices.append(int(job_idx))

        if unknown_job_ids:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "Some job_ids were not found.",
                    "unknown_job_ids": unknown_job_ids,
                },
            )
        allowed_job_indices = resolved_job_indices

    try:
        data, candidature_idx, candidature_summary = add_temporary_candidature(
            data,
            candidate_idx=int(candidate_idx),
            timestamp=ranking_timestamp,
            mappings=artifacts.mappings,
            copy_data=False,
        )

        model = extend_model_for_new_nodes(artifacts.model, data)

        ranked_jobs = rank_jobs_for_candidature(
            model=model,
            data=data,
            candidature_idx=candidature_idx,
            allowed_job_indices=allowed_job_indices,
            config=artifacts.config,
            mappings=artifacts.mappings,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to rank jobs: {exc}") from exc

    persisted_selection: dict[str, Any] | None = None
    if request.persist_selection_rank is not None:
        selected = next((item for item in ranked_jobs if item.rank == request.persist_selection_rank), None)
        if selected is None:
            raise HTTPException(status_code=400, detail="persist_selection_rank was not found in ranked results")
        try:
            data = attach_candidature_to_job(
                data,
                candidature_idx=candidature_idx,
                job_idx=int(selected.metadata["job_index"]),
                copy_data=False,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        persisted_selection = {
            "rank": int(selected.rank),
            "job_id": selected.job_id,
            "job_index": int(selected.metadata["job_index"]),
        }

    if request.persist_graph:
        save_runtime_state(data=data, mappings=artifacts.mappings, config=artifacts.config)

    return {
        "request_ts": int(time.time()),
        "top_k": request.top_k,
        "candidate": {
            "candidate_index": int(candidate_idx),
            "candidate_id": artifacts.mappings.candidate_idx_to_id.get(int(candidate_idx), request.candidate_id),
            "query_candidature_index": int(candidature_idx),
            "query_candidature_insertion": candidature_summary.__dict__,
            "insertion": insertion_summary,
        },
        "persisted_selection": persisted_selection,
        "results": [item.to_dict() for item in ranked_jobs],
    }


@app.post("/recommend/candidates-for-job")
def recommend_candidates_for_job(request: RecommendCandidatesRequest) -> dict[str, Any]:
    artifacts = service.artifacts

    data = deepcopy(artifacts.data)
    insertion_summary: dict[str, Any] | None = None

    if request.job_id is not None:
        job_idx = _resolve_job_idx(request.job_id, artifacts.mappings)
    else:
        assert request.job is not None
        query_job_id = request.job.job_id
        is_existing_job = bool(query_job_id and query_job_id in artifacts.mappings.job_id_to_idx)
        if not query_job_id:
            query_job_id = f"__query_job_{time.time_ns()}"

        raw_job = JobInput(
            job_id=query_job_id,
            description=request.job.description,
            skills=request.job.skills,
            contract=request.job.contract,
            experience=request.job.experience,
            salary=request.job.salary,
            category=request.job.category,
            company=request.job.company,
            timestamp=request.job.timestamp,
        )

        normalized_job = normalize_job_input(raw_job, graph_context=artifacts.mappings)
        try:
            if is_existing_job:
                data, job_idx, job_summary = update_job_in_graph(
                    data,
                    normalized_job,
                    mappings=artifacts.mappings,
                    text_encoder=artifacts.text_encoder,
                    copy_data=True,
                )
            else:
                data, job_idx, job_summary = add_job_to_graph(
                    data,
                    normalized_job,
                    mappings=artifacts.mappings,
                    text_encoder=artifacts.text_encoder,
                    copy_data=True,
                )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        insertion_summary = {
            "normalization": normalized_job.to_dict(),
            "job_insertion": job_summary.__dict__,
        }

    ranking_timestamp = int(time.time())
    allowed_candidate_indices: list[int] | None = None
    if request.candidate_ids is not None:
        resolved_candidate_indices: list[int] = []
        unknown_candidate_ids: list[str] = []
        for candidate_id in request.candidate_ids:
            candidate_idx = artifacts.mappings.candidate_id_to_idx.get(candidate_id)
            if candidate_idx is None:
                unknown_candidate_ids.append(candidate_id)
            else:
                resolved_candidate_indices.append(int(candidate_idx))

        if unknown_candidate_ids:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "Some candidate_ids were not found.",
                    "unknown_candidate_ids": unknown_candidate_ids,
                },
            )
        allowed_candidate_indices = resolved_candidate_indices

    try:
        data, candidature_idx, candidature_summary = add_temporary_job_query_candidature(
            data,
            job_idx=int(job_idx),
            timestamp=ranking_timestamp,
            mappings=artifacts.mappings,
            copy_data=False,
        )

        model = extend_model_for_new_nodes(artifacts.model, data)

        ranked_candidates = rank_candidates_for_job(
            model=model,
            data=data,
            candidature_idx=candidature_idx,
            allowed_candidate_indices=allowed_candidate_indices,
            config=artifacts.config,
            mappings=artifacts.mappings,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to rank candidates: {exc}") from exc

    persisted_selection: dict[str, Any] | None = None
    if request.persist_selection_rank is not None:
        selected = next((item for item in ranked_candidates if item.rank == request.persist_selection_rank), None)
        if selected is None:
            raise HTTPException(status_code=400, detail="persist_selection_rank was not found in ranked results")
        try:
            data = attach_candidate_to_candidature(
                data,
                candidate_idx=int(selected.metadata["candidate_index"]),
                candidature_idx=candidature_idx,
                copy_data=False,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        persisted_selection = {
            "rank": int(selected.rank),
            "candidate_id": selected.candidate_id,
            "candidate_index": int(selected.metadata["candidate_index"]),
        }

    if request.persist_graph:
        save_runtime_state(data=data, mappings=artifacts.mappings, config=artifacts.config)

    return {
        "request_ts": int(time.time()),
        "top_k": request.top_k,
        "job": {
            "job_index": int(job_idx),
            "job_id": artifacts.mappings.job_idx_to_id.get(int(job_idx), request.job_id),
            "query_candidature_index": int(candidature_idx),
            "query_candidature_insertion": candidature_summary.__dict__,
            "insertion": insertion_summary,
        },
        "persisted_selection": persisted_selection,
        "results": [item.to_dict() for item in ranked_candidates],
    }


@app.post("/interactions")
def add_interaction(request: InteractionRequest) -> dict[str, Any]:
    artifacts = service.artifacts

    if request.candidate_idx is not None:
        candidate_idx = int(request.candidate_idx)
    else:
        assert request.candidate_id is not None
        candidate_idx = _resolve_candidate_idx(request.candidate_id, artifacts.mappings)

    if request.job_idx is not None:
        job_idx = int(request.job_idx)
    else:
        assert request.job_id is not None
        job_idx = _resolve_job_idx(request.job_id, artifacts.mappings)

    candidate_count = int(artifacts.data["candidate"].node_id.numel())
    job_count = int(artifacts.data["job"].node_id.numel())
    if candidate_idx < 0 or candidate_idx >= candidate_count:
        raise HTTPException(
            status_code=400,
            detail=f"candidate_idx {candidate_idx} is out of bounds (size={candidate_count}).",
        )
    if job_idx < 0 or job_idx >= job_count:
        raise HTTPException(
            status_code=400,
            detail=f"job_idx {job_idx} is out of bounds (size={job_count}).",
        )

    timestamp = request.timestamp or int(time.time())
    log_interaction(
        candidate_idx=candidate_idx,
        job_idx=job_idx,
        score=request.score,
        ts=timestamp,
    )

    threshold = int(os.getenv("ONLINE_RETRAIN_THRESHOLD", "3"))
    retrain_due = should_retrain(threshold=threshold)

    return {
        "status": "ok",
        "logged": True,
        "retrain_due": retrain_due,
        "threshold": threshold,
    }
