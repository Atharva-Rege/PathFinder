from dataclasses import dataclass, field
from typing import Any

from torch_geometric.data import HeteroData


@dataclass
class CandidateInput:
    """Raw candidate payload collected from the terminal."""

    candidate_id: str | None = None
    description: str = ""
    skills: list[str] = field(default_factory=list)
    contract: str | None = None
    origin: str | None = None
    experience: str | None = None
    salary_current: float | int | None = None
    timestamp: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "description": self.description,
            "skills": list(self.skills),
            "contract": self.contract,
            "origin": self.origin,
            "experience": self.experience,
            "salary_current": self.salary_current,
            "timestamp": self.timestamp,
        }


@dataclass
class JobInput:
    """Raw job payload collected from the terminal."""

    job_id: str
    description: str = ""
    skills: list[str] = field(default_factory=list)
    contract: str | None = None
    experience: str | None = None
    salary: float | int | None = None
    category: str | None = None
    company: str | None = None
    timestamp: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "description": self.description,
            "skills": list(self.skills),
            "contract": self.contract,
            "experience": self.experience,
            "salary": self.salary,
            "category": self.category,
            "company": self.company,
            "timestamp": self.timestamp,
        }


@dataclass
class NormalizedCandidateInput:
    """Candidate payload after mapping into graph-compatible values."""

    candidate_id: str | None
    description: str
    skills: list[str]
    contract: str | None
    origin: str | None
    experience: str | None
    salary_category: str | None
    timestamp: int
    yearmonth: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "description": self.description,
            "skills": list(self.skills),
            "contract": self.contract,
            "origin": self.origin,
            "experience": self.experience,
            "salary_category": self.salary_category,
            "timestamp": self.timestamp,
            "yearmonth": self.yearmonth,
            "extra": dict(self.extra),
        }


@dataclass
class NormalizedJobInput:
    """Job payload after mapping into graph-compatible values."""

    job_id: str
    description: str
    skills: list[str]
    contract: str | None
    experience: str | None
    salary_category: str | None
    category: str | None
    company: str | None
    timestamp: int
    yearmonth: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "description": self.description,
            "skills": list(self.skills),
            "contract": self.contract,
            "experience": self.experience,
            "salary_category": self.salary_category,
            "category": self.category,
            "company": self.company,
            "timestamp": self.timestamp,
            "yearmonth": self.yearmonth,
            "extra": dict(self.extra),
        }


@dataclass
class GraphMappings:
    """
    Lookup tables needed to connect a new candidate to existing graph nodes.

    Each mapping should translate a normalized value into the corresponding
    node index already present in the graph.
    """

    skill_to_idx: dict[str, int] = field(default_factory=dict)
    contract_to_idx: dict[str, int] = field(default_factory=dict)
    origin_to_idx: dict[str, int] = field(default_factory=dict)
    experience_to_idx: dict[str, int] = field(default_factory=dict)
    salary_to_idx: dict[str, int] = field(default_factory=dict)
    category_to_idx: dict[str, int] = field(default_factory=dict)
    company_to_idx: dict[str, int] = field(default_factory=dict)
    time_to_idx: dict[int, int] = field(default_factory=dict)
    time_base_yearmonth: int | None = None
    candidate_idx_to_id: dict[int, Any] = field(default_factory=dict)
    candidate_id_to_idx: dict[Any, int] = field(default_factory=dict)
    job_idx_to_id: dict[int, Any] = field(default_factory=dict)
    job_id_to_idx: dict[Any, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, dict[Any, int] | dict[int, Any]]:
        return {
            "skill_to_idx": dict(self.skill_to_idx),
            "contract_to_idx": dict(self.contract_to_idx),
            "origin_to_idx": dict(self.origin_to_idx),
            "experience_to_idx": dict(self.experience_to_idx),
            "salary_to_idx": dict(self.salary_to_idx),
            "category_to_idx": dict(self.category_to_idx),
            "company_to_idx": dict(self.company_to_idx),
            "time_to_idx": dict(self.time_to_idx),
            "time_base_yearmonth": self.time_base_yearmonth,
            "candidate_idx_to_id": dict(self.candidate_idx_to_id),
            "candidate_id_to_idx": dict(self.candidate_id_to_idx),
            "job_idx_to_id": dict(self.job_idx_to_id),
            "job_id_to_idx": dict(self.job_id_to_idx),
        }


@dataclass
class RuntimeArtifacts:
    """Objects loaded once and reused across terminal inference calls."""

    data: HeteroData
    model: Any
    mappings: GraphMappings
    text_encoder: Any | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphInsertionSummary:
    """Debug summary of the nodes and edges added during inference."""

    candidate_idx: int
    candidature_idx: int | None = None
    candidate_time_idx: int | None = None
    candidature_time_idx: int | None = None
    matched_skill_indices: list[int] = field(default_factory=list)
    contract_idx: int | None = None
    origin_idx: int | None = None
    experience_idx: int | None = None
    salary_idx: int | None = None
    category_idx: int | None = None
    company_idx: int | None = None
    job_idx: int | None = None
    time_node_created: bool = False


@dataclass
class RankedJob:
    """Single ranked inference result."""

    job_id: Any
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "score": self.score,
            "rank": self.rank,
            "metadata": dict(self.metadata),
        }


@dataclass
class RankedCandidate:
    """Single ranked candidate inference result."""

    candidate_id: Any
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "score": self.score,
            "rank": self.rank,
            "metadata": dict(self.metadata),
        }
