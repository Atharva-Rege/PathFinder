from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from graph_builder import clean_salary
from inference_types import CandidateInput, GraphMappings, NormalizedCandidateInput

from inference_types import JobInput, NormalizedJobInput, GraphMappings

def _prompt_optional(prompt_text: str) -> str | None:
    value = input(prompt_text).strip()
    return value or None


def _parse_skills(raw_value: str) -> list[str]:
    if not raw_value.strip():
        return []
    return [skill.strip() for skill in raw_value.split(",") if skill.strip()]


def _parse_salary(raw_value: str | None) -> float | int | None:
    if raw_value is None or not raw_value.strip():
        return None

    sanitized = raw_value.strip().lower().replace(",", "").replace("eur", "").replace("euro", "").replace("€", "")
    if sanitized.endswith("k"):
        sanitized = sanitized[:-1] + "000"

    try:
        numeric_value = float(sanitized)
    except ValueError:
        return None

    return int(numeric_value) if numeric_value.is_integer() else numeric_value


def _parse_timestamp(raw_value: str | None) -> int | None:
    if raw_value is None or not raw_value.strip():
        return int(time.time())

    stripped = raw_value.strip()
    if stripped.isdigit():
        return int(stripped)

    supported_formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y",
        "%d/%m/%Y %H:%M:%S",
    ]
    for date_format in supported_formats:
        try:
            return int(datetime.strptime(stripped, date_format).timestamp())
        except ValueError:
            continue

    return int(time.time())


def _derive_yearmonth(
    timestamp: int,
    time_to_idx: dict[int, int],
    time_base_yearmonth: int | None,
) -> int | None:
    dt_value = datetime.fromtimestamp(timestamp)
    absolute_yearmonth = dt_value.year * 12 + dt_value.month - 1

    if not time_to_idx or time_base_yearmonth is None:
        return None

    derived = absolute_yearmonth - time_base_yearmonth
    return derived if derived in time_to_idx else None


def _normalize_label(value: str | None, valid_values: dict[str, int]) -> str | None:
    if value is None:
        return None

    stripped = value.strip()
    if not stripped:
        return None

    if stripped in valid_values:
        return stripped

    lowered_lookup = {key.lower(): key for key in valid_values}
    return lowered_lookup.get(stripped.lower())


def prompt_candidate_input() -> CandidateInput:
    """
    Collect candidate details from the terminal in a simple prompt-driven flow.
    """
    skills_raw = input("Candidate skills (comma separated): ").strip()
    salary_raw = _prompt_optional("Current salary: ")
    timestamp_raw = _prompt_optional("Timestamp or date (blank = now): ")

    return CandidateInput(
        description=input("Candidate description / CV text: ").strip(),
        skills=_parse_skills(skills_raw),
        contract=_prompt_optional("Contract: "),
        origin=_prompt_optional("Origin: "),
        experience=_prompt_optional("Experience: "),
        salary_current=_parse_salary(salary_raw),
        timestamp=_parse_timestamp(timestamp_raw),
    )


def normalize_candidate_input(
    candidate: CandidateInput,
    *,
    graph_context: GraphMappings | dict[str, Any] | None = None,
) -> NormalizedCandidateInput:
    """
    Convert terminal input into values compatible with the trained graph.
    """
    mappings = graph_context if isinstance(graph_context, GraphMappings) else GraphMappings()
    timestamp = candidate.timestamp or int(time.time())

    normalized_skills: list[str] = []
    unknown_skills: list[str] = []
    for skill in candidate.skills:
        normalized_skill = _normalize_label(skill, mappings.skill_to_idx)
        if normalized_skill is None:
            unknown_skills.append(skill)
        else:
            normalized_skills.append(normalized_skill)

    contract = _normalize_label(candidate.contract, mappings.contract_to_idx)
    origin = _normalize_label(candidate.origin, mappings.origin_to_idx)
    experience = _normalize_label(candidate.experience, mappings.experience_to_idx)
    salary_category = clean_salary(candidate.salary_current) if candidate.salary_current is not None else None
    if salary_category not in mappings.salary_to_idx:
        salary_category = None

    yearmonth = _derive_yearmonth(
        timestamp,
        mappings.time_to_idx,
        mappings.time_base_yearmonth,
    )

    return NormalizedCandidateInput(
        description=candidate.description,
        skills=normalized_skills,
        contract=contract,
        origin=origin,
        experience=experience,
        salary_category=salary_category,
        timestamp=timestamp,
        yearmonth=yearmonth,
        extra={
            "raw_input": candidate.to_dict(),
            "unknown_skills": unknown_skills,
            "unmapped_contract": candidate.contract if candidate.contract and contract is None else None,
            "unmapped_origin": candidate.origin if candidate.origin and origin is None else None,
            "unmapped_experience": candidate.experience if candidate.experience and experience is None else None,
        },
    )

from inference_types import JobInput, NormalizedJobInput
from graph_builder import clean_salary


def prompt_job_input() -> JobInput:
    """
    Collect job details from terminal.
    """
    skills_raw = input("Job skills (comma separated): ").strip()
    salary_raw = _prompt_optional("Salary: ")
    timestamp_raw = _prompt_optional("Timestamp or date (blank = now): ")

    return JobInput(
        job_id=input("Job ID: ").strip(),
        description=input("Job description: ").strip(),
        skills=_parse_skills(skills_raw),
        contract=_prompt_optional("Contract: "),
        experience=_prompt_optional("Experience: "),
        salary=_parse_salary(salary_raw),
        category=_prompt_optional("Category: "),
        company=_prompt_optional("Company: "),
        timestamp=_parse_timestamp(timestamp_raw),
    )


def normalize_job_input(
    job: JobInput,
    *,
    graph_context: GraphMappings | dict[str, Any] | None = None,
) -> NormalizedJobInput:
    """
    Convert job input into graph-compatible values.
    """
    mappings = graph_context if isinstance(graph_context, GraphMappings) else GraphMappings()
    timestamp = job.timestamp or int(time.time())

    normalized_skills = []
    for skill in job.skills:
        normalized = _normalize_label(skill, mappings.skill_to_idx)
        if normalized:
            normalized_skills.append(normalized)

    contract = _normalize_label(job.contract, mappings.contract_to_idx)
    experience = _normalize_label(job.experience, mappings.experience_to_idx)

    salary_category = clean_salary(job.salary) if job.salary is not None else None
    if salary_category not in mappings.salary_to_idx:
        salary_category = None

    category = _normalize_label(job.category, mappings.category_to_idx)
    company = _normalize_label(job.company, mappings.company_to_idx)

    yearmonth = _derive_yearmonth(
        timestamp,
        mappings.time_to_idx,
        mappings.time_base_yearmonth,
    )

    return NormalizedJobInput(
        job_id=job.job_id,
        description=job.description,
        skills=normalized_skills,
        contract=contract,
        experience=experience,
        salary_category=salary_category,
        category=category,
        company=company,
        timestamp=timestamp,
        yearmonth=yearmonth,
        extra={"raw_input": job.to_dict()},
    )


