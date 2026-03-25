from __future__ import annotations

from datetime import datetime
from pathlib import Path

from candidate_input import normalize_candidate_input, prompt_candidate_input
from graph_runtime import (
    add_candidate_to_graph,
    add_temporary_candidature,
    extend_model_for_new_nodes,
    load_runtime_artifacts,
)
from ranker import rank_jobs_for_candidature

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "required_data"


def _print_inference_summary(normalized_candidate) -> None:
    timestamp_iso = datetime.fromtimestamp(normalized_candidate.timestamp).isoformat(sep=" ")
    print("\nInference summary")
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


def _print_graph_insertion_summary(candidate_summary, candidature_summary) -> None:
    print("\nGraph insertion summary")
    print(f"  Candidate node index: {candidate_summary.candidate_idx}")
    print(f"  Candidate -> time index: {candidate_summary.candidate_time_idx}")
    print(f"  Candidate -> skill indices: {candidate_summary.matched_skill_indices}")
    print(f"  Candidate -> contract index: {candidate_summary.contract_idx}")
    print(f"  Candidate -> origin index: {candidate_summary.origin_idx}")
    print(f"  Candidate -> experience index: {candidate_summary.experience_idx}")
    print(f"  Candidate -> salary index: {candidate_summary.salary_idx}")
    print(f"  Candidature node index: {candidature_summary.candidature_idx}")
    print(f"  Candidate -> candidature created: {candidature_summary.candidature_idx is not None}")
    print(f"  Candidature -> time index: {candidature_summary.candidature_time_idx}")


def main() -> None:
    """
    Terminal entrypoint for inference.

    Flow:
    1. Load graph + trained model artifacts
    2. Prompt for candidate details
    3. Normalize user input into graph-compatible values
    4. Add a temporary candidate node and its attribute edges
    5. Add a temporary candidature node
    6. Run ranking against jobs
    7. Print top-k job recommendations
    """
    artifacts = load_runtime_artifacts(
        data_dir=DEFAULT_DATA_DIR,
        model_path=None,
        config=None,
    )

    candidate = prompt_candidate_input()
    normalized_candidate = normalize_candidate_input(
        candidate,
        graph_context=artifacts.mappings,
    )
    _print_inference_summary(normalized_candidate)

    data, candidate_idx, candidate_summary = add_candidate_to_graph(
        artifacts.data,
        normalized_candidate,
        mappings=artifacts.mappings,
        text_encoder=artifacts.text_encoder,
    )
    data, candidature_idx, candidature_summary = add_temporary_candidature(
        data,
        candidate_idx=candidate_idx,
        timestamp=normalized_candidate.timestamp,
        mappings=artifacts.mappings,
    )
    _print_graph_insertion_summary(candidate_summary, candidature_summary)

    model = extend_model_for_new_nodes(artifacts.model, data)
    ranked_jobs = rank_jobs_for_candidature(
        model=model,
        data=data,
        candidature_idx=candidature_idx,
        config=artifacts.config,
        mappings=artifacts.mappings,
        top_k=10,
    )

    if not ranked_jobs:
        print("No ranked jobs yet. The inference template is in place.")
        return

    for job in ranked_jobs:
        print(f"{job.rank}. {job.job_id} -> {job.score:.4f}")


if __name__ == "__main__":
    main()
