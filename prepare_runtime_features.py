from pathlib import Path

import numpy as np
import pandas as pd

from graph_builder import DEFAULT_DATA_DIR, DEFAULT_SAVE_DIR, load_data, load_nodes
from text_encoder_runtime import build_text_encoder


def main() -> None:
    data_dir = DEFAULT_DATA_DIR
    save_dir = DEFAULT_SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    (
        candidate,
        candidate_skills,
        job_candidates,
        job_skills,
        candidate_pre,
        _job,
        job_pre,
        _skill,
        _hierarchy,
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

    _ = (
        unique_skill_id,
        unique_candidature_id,
        unique_contract_id,
        unique_exp_id,
        unique_origin_id,
        unique_salary_id,
        unique_category_id,
        unique_company_id,
        unique_concept_id,
        unique_time_id,
    )

    model = build_text_encoder("sentence-transformers/all-MiniLM-L6-v2")

    job_sentences = unique_job_id["jobID"].fillna("").astype(str).tolist()
    print(f"Encoding {len(job_sentences)} job sentences...")
    job_features = model.encode(job_sentences, show_progress_bar=True)

    candidate_cv = pd.DataFrame({"name": unique_user_id["nameID"]})
    candidate_cv = candidate_cv.merge(candidate, on="name", how="left")
    candidate_sentences = candidate_cv["description"].fillna("").astype(str).tolist()
    print(f"Encoding {len(candidate_sentences)} candidate sentences...")
    candidate_features = model.encode(candidate_sentences, show_progress_bar=True)

    np.save(save_dir / "job_features.npy", job_features)
    np.save(save_dir / "candidate_features.npy", candidate_features)

    print(f"Saved {(save_dir / 'job_features.npy')} with shape {job_features.shape}")
    print(f"Saved {(save_dir / 'candidate_features.npy')} with shape {candidate_features.shape}")


if __name__ == "__main__":
    main()
