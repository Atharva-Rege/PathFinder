# event_processor.py
from pathlib import Path

from graph_runtime import (
    add_candidate_to_graph,
    add_temporary_candidature,
    extend_model_for_new_nodes,
    load_runtime_artifacts,
)
from job_runtime import add_job_to_graph
from graph_persistence import save_graph, save_mappings
from inference_types import NormalizedCandidateInput
from ranker import rank_jobs_for_candidature

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
GRAPH_PATH = OUTPUT_DIR / "graph.pt"
MAPPINGS_PATH = OUTPUT_DIR / "mappings.pt"

def process_candidate_event(payload: dict):
    artifacts = load_runtime_artifacts(data_dir=str(OUTPUT_DIR.parent / "required_data"))

    # normalize externally (you already have normalize_candidate_input)
    from candidate_input import normalize_candidate_input
    normalized: NormalizedCandidateInput = normalize_candidate_input(
        payload["candidate"],
        graph_context=artifacts.mappings,
    )

    data, c_idx, _ = add_candidate_to_graph(
        artifacts.data,
        normalized,
        mappings=artifacts.mappings,
        text_encoder=artifacts.text_encoder,
    )
    data, cand_idx, _ = add_temporary_candidature(
        data,
        candidate_idx=c_idx,
        timestamp=normalized.timestamp,
        mappings=artifacts.mappings,
    )

    model = extend_model_for_new_nodes(artifacts.model, data)

    # recency filter handled in ranker call via config
    ranked = rank_jobs_for_candidature(
        model=model,
        data=data,
        candidature_idx=cand_idx,
        config=artifacts.config,
        mappings=artifacts.mappings,
        top_k=10,
    )

    # persist
    save_graph(data, GRAPH_PATH)
    save_mappings(artifacts.mappings, MAPPINGS_PATH)

    return ranked


def process_job_event(payload: dict):
    artifacts = load_runtime_artifacts(data_dir=str(OUTPUT_DIR.parent / "required_data"))

    data, j_idx = add_job_to_graph(
        artifacts.data,
        payload["job"],
        mappings=artifacts.mappings,
        text_encoder=artifacts.text_encoder,
    )

    # persist
    save_graph(data, GRAPH_PATH)
    save_mappings(artifacts.mappings, MAPPINGS_PATH)

    return {"job_idx": j_idx}