from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ChunkCategory(str, Enum):
    workflows = "workflows"
    dashboards_reports = "dashboards_reports"
    data_entry = "data_entry"
    automation = "automation"
    compliance_governance = "compliance_governance"
    integrations_user_facing = "integrations_user_facing"
    administration_config = "administration_config"
    uncategorized = "uncategorized"


class ModuleType(str, Enum):
    workflow = "workflow"
    dashboard = "dashboard"
    form = "form"
    automation_surface = "automation_surface"
    compliance_surface = "compliance_surface"
    config_surface = "config_surface"
    reporting_module = "reporting_module"


class Frequency(str, Enum):
    continuous = "continuous"
    daily = "daily"
    weekly = "weekly"
    periodic = "periodic"
    event_driven = "event_driven"


class AccessSensitivity(str, Enum):
    public = "public"
    internal = "internal"
    restricted = "restricted"
    highly_restricted = "highly_restricted"


class RoleCategory(str, Enum):
    clinical = "clinical"
    administrative = "administrative"
    analytical = "analytical"
    compliance_governance = "compliance_governance"
    operational = "operational"
    training_change_mgmt = "training_change_mgmt"
    supervisory = "supervisory"


class InteractionPattern(str, Enum):
    primary_daily_user = "primary_daily_user"
    periodic_reviewer = "periodic_reviewer"
    exception_handler = "exception_handler"
    oversight_approver = "oversight_approver"
    configuration_owner = "configuration_owner"


class SenioritySignal(str, Enum):
    entry_level = "entry_level"
    experienced = "experienced"
    senior_specialist = "senior_specialist"
    leadership = "leadership"


class TransformationType(str, Enum):
    existing_unchanged = "existing_unchanged"
    existing_augmented = "existing_augmented"
    existing_consolidated = "existing_consolidated"
    newly_created = "newly_created"


class RetrievalChannel(str, Enum):
    title = "title"
    skills = "skills"
    dual = "dual"


class DataSensitivity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    regulated = "regulated"


# ---------------------------------------------------------------------------
# Phase 1 models
# ---------------------------------------------------------------------------

class ContentChunk(BaseModel):
    id: str
    source_url: str
    category: ChunkCategory
    heading_path: list[str] = Field(default_factory=list)
    text: str
    tables: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 2 models
# ---------------------------------------------------------------------------

class UserFacingModule(BaseModel):
    name: str
    type: ModuleType
    description: str
    user_actions: list[str] = Field(default_factory=list)
    decisions_required: list[str] = Field(default_factory=list)
    domain_knowledge_needed: list[str] = Field(default_factory=list)
    frequency: Frequency = Frequency.daily
    access_sensitivity: AccessSensitivity = AccessSensitivity.internal
    source_chunks: list[str] = Field(default_factory=list)


class OrganizationalObligations(BaseModel):
    compliance_regimes: list[str] = Field(default_factory=list)
    data_governance_needs: list[str] = Field(default_factory=list)
    training_needs: list[str] = Field(default_factory=list)
    change_management_needs: list[str] = Field(default_factory=list)
    oversight_needs: list[str] = Field(default_factory=list)
    vendor_liaison_needs: list[str] = Field(default_factory=list)


class WorkflowEdge(BaseModel):
    from_module: str
    to_module: str
    handoff_description: str = ""


class WorkflowInteractionMap(BaseModel):
    edges: list[WorkflowEdge] = Field(default_factory=list)


class SystemRepresentation(BaseModel):
    modules: list[UserFacingModule] = Field(default_factory=list)
    obligations: OrganizationalObligations = Field(
        default_factory=OrganizationalObligations
    )
    workflow_map: WorkflowInteractionMap = Field(
        default_factory=WorkflowInteractionMap
    )


# ---------------------------------------------------------------------------
# Phase 3 models
# ---------------------------------------------------------------------------

class RoleTransformation(BaseModel):
    transformation_type: TransformationType
    rationale: str = ""


class RoleNeed(BaseModel):
    id: str
    description: str
    category: RoleCategory
    interaction_pattern: InteractionPattern
    domain_expertise: list[str] = Field(default_factory=list)
    system_skills: list[str] = Field(default_factory=list)
    seniority_signal: SenioritySignal = SenioritySignal.experienced
    derived_job_titles: list[str] = Field(default_factory=list)
    derived_skill_keywords: list[str] = Field(default_factory=list)
    source_module: str = ""
    source_obligations: list[str] = Field(default_factory=list)
    transformation: RoleTransformation = Field(
        default_factory=lambda: RoleTransformation(
            transformation_type=TransformationType.existing_unchanged
        )
    )


# ---------------------------------------------------------------------------
# Phase 4 models
# ---------------------------------------------------------------------------

class VacancyRecord(BaseModel):
    identifier: str
    title: str
    enriched_job_title: str
    description: str = ""
    enriched_skills: str = ""
    enriched_tasks: str = ""
    enriched_industry: str = ""
    enriched_contract_type: str = ""
    country: str = ""
    locality: str = ""


class RetrievalHit(BaseModel):
    vacancy_id: str
    vacancy_title: str
    enriched_job_title: str
    cosine_score: float
    channel: RetrievalChannel
    query_used: str = ""


class ScoringDetail(BaseModel):
    vacancy_id: str
    enriched_job_title: str
    role_need_id: str
    task_score: float
    domain_score: float
    seniority_score: float
    composite_score: float
    rationale: str = ""


class RetrievalResult(BaseModel):
    role_need_id: str
    hits: list[RetrievalHit] = Field(default_factory=list)
    scored: list[ScoringDetail] = Field(default_factory=list)


class ClusterInfo(BaseModel):
    cluster_id: int
    canonical_title: str
    member_titles: list[str] = Field(default_factory=list)
    member_vacancy_ids: list[str] = Field(default_factory=list)
    centroid_distance: float = 0.0


class RecommendedRole(BaseModel):
    canonical_title: str
    alternative_titles: list[str] = Field(default_factory=list)
    mapped_role_needs: list[str] = Field(default_factory=list)
    representative_vacancy_ids: list[str] = Field(default_factory=list)
    category: RoleCategory = RoleCategory.operational
    interaction_pattern: InteractionPattern = InteractionPattern.primary_daily_user
    seniority: SenioritySignal = SenioritySignal.experienced
    confidence: float = 0.0
    retrieval_channel: RetrievalChannel = RetrievalChannel.title
    transformation: RoleTransformation = Field(
        default_factory=lambda: RoleTransformation(
            transformation_type=TransformationType.existing_unchanged
        )
    )
    justification: str = ""


# ---------------------------------------------------------------------------
# Phase 6 – Final output
# ---------------------------------------------------------------------------

class RoleRosterMetadata(BaseModel):
    system_name: str = ""
    generated_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    total_roles: int = 0
    coverage_pct: float = 0.0
    uncovered_needs: list[str] = Field(default_factory=list)


class RoleRoster(BaseModel):
    metadata: RoleRosterMetadata = Field(default_factory=RoleRosterMetadata)
    roles: list[RecommendedRole] = Field(default_factory=list)

    by_function: dict[str, list[RecommendedRole]] = Field(default_factory=dict)
    by_interaction_pattern: dict[str, list[RecommendedRole]] = Field(
        default_factory=dict
    )
    by_transformation: dict[str, list[RecommendedRole]] = Field(
        default_factory=dict
    )


class IntermediateArtifacts(BaseModel):
    content_chunks: list[ContentChunk] = Field(default_factory=list)
    system_representation: SystemRepresentation = Field(
        default_factory=SystemRepresentation
    )
    role_needs: list[RoleNeed] = Field(default_factory=list)
    retrieval_results: list[RetrievalResult] = Field(default_factory=list)
    scoring_breakdowns: list[ScoringDetail] = Field(default_factory=list)
    clustering_output: list[ClusterInfo] = Field(default_factory=list)


class PipelineOutput(BaseModel):
    roster: RoleRoster = Field(default_factory=RoleRoster)
    intermediate: IntermediateArtifacts = Field(
        default_factory=IntermediateArtifacts
    )
