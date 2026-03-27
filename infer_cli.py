from __future__ import annotations

from datetime import datetime
from pathlib import Path

from candidate_input import (
    normalize_candidate_input,
    normalize_job_input,
    prompt_candidate_input,
    prompt_job_input,
)
from graph_runtime import (
    add_candidate_to_graph,
    add_job_to_graph,
    add_temporary_candidature,
    extend_model_for_new_nodes,
    load_runtime_artifacts,
    save_runtime_state,
)
from ranker import rank_jobs_for_candidature

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "required_data"


def _prompt_mode() -> str:
    print("\nChoose action")
    print("  1. Add candidate and rank jobs")
    print("  2. Add job to persistent graph")
    raw_value = input("Select option [1/2]: ").strip()
    return raw_value or "1"


def _print_candidate_summary(normalized_candidate) -> None:
    timestamp_iso = datetime.fromtimestamp(normalized_candidate.timestamp).isoformat(sep=" ")
    print("\nCandidate summary")
    print(f"  Recommendation time: {timestamp_iso}")
    print(f"  Time bucket: {normalized_candidate.yearmonth}")
    print(f"  Matched skills: {normalized_candidate.skills}")
    print(f"  Matched contract: {normalized_candidate.contract}")
    print(f"  Matched origin: {normalized_candidate.origin}")
    print(f"  Matched experience: {normalized_candidate.experience}")
    print(f"  Matched salary category: {normalized_candidate.salary_category}")

    unknown_skills = normalized_candidate.extra.get("unknown_skills", [])
    if unknown_skills:
        print(f"  Unmapped skills: {unknown_skills}")
    if normalized_candidate.extra.get("unmapped_contract") is not None:
        print(f"  Unmapped contract: {normalized_candidate.extra['unmapped_contract']}")
    if normalized_candidate.extra.get("unmapped_origin") is not None:
        print(f"  Unmapped origin: {normalized_candidate.extra['unmapped_origin']}")
    if normalized_candidate.extra.get("unmapped_experience") is not None:
        print(f"  Unmapped experience: {normalized_candidate.extra['unmapped_experience']}")


def _print_job_summary(normalized_job) -> None:
    timestamp_iso = datetime.fromtimestamp(normalized_job.timestamp).isoformat(sep=" ")
    print("\nJob summary")
    print(f"  Job ID: {normalized_job.job_id}")
    print(f"  Timestamp: {timestamp_iso}")
    print(f"  Time bucket: {normalized_job.yearmonth}")
    print(f"  Matched skills: {normalized_job.skills}")
    print(f"  Matched contract: {normalized_job.contract}")
    print(f"  Matched experience: {normalized_job.experience}")
    print(f"  Matched salary category: {normalized_job.salary_category}")
    print(f"  Matched category: {normalized_job.category}")
    print(f"  Matched company: {normalized_job.company}")

    unknown_skills = normalized_job.extra.get("unknown_skills", [])
    if unknown_skills:
        print(f"  Unmapped skills: {unknown_skills}")
    if normalized_job.extra.get("unmapped_contract") is not None:
        print(f"  Unmapped contract: {normalized_job.extra['unmapped_contract']}")
    if normalized_job.extra.get("unmapped_experience") is not None:
        print(f"  Unmapped experience: {normalized_job.extra['unmapped_experience']}")
    if normalized_job.extra.get("unmapped_category") is not None:
        print(f"  Unmapped category: {normalized_job.extra['unmapped_category']}")
    if normalized_job.extra.get("unmapped_company") is not None:
        print(f"  Unmapped company: {normalized_job.extra['unmapped_company']}")


def _print_candidate_insertion_summary(candidate_summary, candidature_summary) -> None:
    print("\nGraph insertion summary")
    print(f"  Candidate node index: {candidate_summary.candidate_idx}")
    print(f"  Candidate -> time index: {candidate_summary.candidate_time_idx}")
    print(f"  Candidate -> skill indices: {candidate_summary.matched_skill_indices}")
    print(f"  Candidate -> contract index: {candidate_summary.contract_idx}")
    print(f"  Candidate -> origin index: {candidate_summary.origin_idx}")
    print(f"  Candidate -> experience index: {candidate_summary.experience_idx}")
    print(f"  Candidate -> salary index: {candidate_summary.salary_idx}")
    print(f"  New time node created: {candidate_summary.time_node_created or candidature_summary.time_node_created}")
    print(f"  Candidature node index: {candidature_summary.candidature_idx}")
    print(f"  Candidature -> time index: {candidature_summary.candidature_time_idx}")


def _print_job_insertion_summary(job_summary) -> None:
    print("\nGraph insertion summary")
    print(f"  Job node index: {job_summary.job_idx}")
    print(f"  Job -> time index: {job_summary.candidate_time_idx}")
    print(f"  Job -> skill indices: {job_summary.matched_skill_indices}")
    print(f"  Job -> contract index: {job_summary.contract_idx}")
    print(f"  Job -> experience index: {job_summary.experience_idx}")
    print(f"  Job -> salary index: {job_summary.salary_idx}")
    print(f"  Job -> category index: {job_summary.category_idx}")
    print(f"  Job -> company index: {job_summary.company_idx}")
    print(f"  New time node created: {job_summary.time_node_created}")


def _run_candidate_flow(artifacts) -> None:
    candidate = prompt_candidate_input()
    normalized_candidate = normalize_candidate_input(
        candidate,
        graph_context=artifacts.mappings,
    )
    _print_candidate_summary(normalized_candidate)

    data, candidate_idx, candidate_summary = add_candidate_to_graph(
        artifacts.data,
        normalized_candidate,
        mappings=artifacts.mappings,
        text_encoder=artifacts.text_encoder,
        copy_data=False,
    )
    data, candidature_idx, candidature_summary = add_temporary_candidature(
        data,
        candidate_idx=candidate_idx,
        timestamp=normalized_candidate.timestamp,
        mappings=artifacts.mappings,
        copy_data=False,
    )
    _print_candidate_insertion_summary(candidate_summary, candidature_summary)

    model = extend_model_for_new_nodes(artifacts.model, data)
    save_runtime_state(data=data, mappings=artifacts.mappings, config=artifacts.config)

    ranked_jobs = rank_jobs_for_candidature(
        model=model,
        data=data,
        candidature_idx=candidature_idx,
        config=artifacts.config,
        mappings=artifacts.mappings,
        top_k=10,
    )

    if not ranked_jobs:
        print("No ranked jobs were produced.")
        return

    print("\nTop job recommendations")
    for job in ranked_jobs:
        print(f"{job.rank}. {job.job_id} -> {job.score:.4f}")


def _run_job_flow(artifacts) -> None:
    job = prompt_job_input()
    normalized_job = normalize_job_input(
        job,
        graph_context=artifacts.mappings,
    )
    _print_job_summary(normalized_job)

    data, _, job_summary = add_job_to_graph(
        artifacts.data,
        normalized_job,
        mappings=artifacts.mappings,
        text_encoder=artifacts.text_encoder,
        copy_data=False,
    )
    _print_job_insertion_summary(job_summary)

    _ = extend_model_for_new_nodes(artifacts.model, data)
    save_runtime_state(data=data, mappings=artifacts.mappings, config=artifacts.config)
    print("Job persisted to graph snapshot.")


def main() -> None:
    """
    Terminal entrypoint for persistent graph mutation and candidate ranking.
    """
    artifacts = load_runtime_artifacts(
        data_dir=DEFAULT_DATA_DIR,
        model_path=None,
        config=None,
    )

    mode = _prompt_mode()
    if mode == "2":
        _run_job_flow(artifacts)
        return

    _run_candidate_flow(artifacts)


if __name__ == "__main__":
    main()
