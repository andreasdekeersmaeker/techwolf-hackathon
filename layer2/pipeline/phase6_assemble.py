"""Phase 6: Generate justification narratives and assemble the final RoleRoster."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime

from models.schemas import (
    ClusterInfo,
    ContentChunk,
    IntermediateArtifacts,
    PipelineOutput,
    RecommendedRole,
    RetrievalResult,
    RoleNeed,
    RoleRoster,
    RoleRosterMetadata,
    ScoringDetail,
    SystemRepresentation,
)
from pipeline.phase5_reason import CoverageReport
from services.claude_client import ask_claude_json

log = logging.getLogger(__name__)


def run(
    roles: list[RecommendedRole],
    role_needs: list[RoleNeed],
    coverage: CoverageReport,
    system_rep: SystemRepresentation,
    chunks: list[ContentChunk],
    retrieval_results: list[RetrievalResult],
    scoring_breakdowns: list[ScoringDetail],
    clustering_output: list[ClusterInfo],
) -> PipelineOutput:
    """Phase 6: Assemble the final output with justifications."""
    log.info("Phase 6: Assembling final output")

    need_by_id = {n.id: n for n in role_needs}

    # Step 6.1 — Generate justification narratives
    roles_with_justifications = _generate_justifications(roles, need_by_id, system_rep)
    log.info("Generated justifications for %d roles", len(roles_with_justifications))

    # Step 6.2 — Build grouped views
    by_function: dict[str, list[RecommendedRole]] = defaultdict(list)
    by_interaction: dict[str, list[RecommendedRole]] = defaultdict(list)
    by_transformation: dict[str, list[RecommendedRole]] = defaultdict(list)

    for role in roles_with_justifications:
        by_function[role.category.value].append(role)
        by_interaction[role.interaction_pattern.value].append(role)
        by_transformation[role.transformation.transformation_type.value].append(role)

    # Detect system name from modules or chunks
    system_name = ""
    if system_rep.modules:
        # Ask LLM for a concise system name
        module_names = [m.name for m in system_rep.modules[:10]]
        result = ask_claude_json(
            system_prompt="You name software systems based on their modules. Return JSON: {\"name\": \"...\"}",
            user_prompt=f"Modules: {module_names}. What is a concise name for this system?",
        )
        if isinstance(result, dict):
            system_name = result.get("name", "Unnamed System")

    roster = RoleRoster(
        metadata=RoleRosterMetadata(
            system_name=system_name or "Unnamed System",
            generated_at=datetime.utcnow().isoformat(),
            total_roles=len(roles_with_justifications),
            coverage_pct=coverage.coverage_pct,
            uncovered_needs=[n.description[:120] for n in coverage.uncovered_needs],
        ),
        roles=roles_with_justifications,
        by_function=dict(by_function),
        by_interaction_pattern=dict(by_interaction),
        by_transformation=dict(by_transformation),
    )

    intermediate = IntermediateArtifacts(
        content_chunks=chunks,
        system_representation=system_rep,
        role_needs=role_needs,
        retrieval_results=retrieval_results,
        scoring_breakdowns=scoring_breakdowns,
        clustering_output=clustering_output,
    )

    output = PipelineOutput(roster=roster, intermediate=intermediate)
    log.info("Phase 6 complete. %d roles assembled.", roster.metadata.total_roles)
    return output


# ---------------------------------------------------------------------------
# Justification generation
# ---------------------------------------------------------------------------

_JUSTIFICATION_SYSTEM = """You generate concise justification narratives for recommended roles.
Each role is needed by an organization USING (not building) a software system.

For each role, write 2-3 sentences explaining:
1. WHY this role is needed (what system modules/workflows require it)
2. WHAT the person in this role does with the system
3. HOW this role relates to legacy roles (if applicable)

Be specific — reference actual module names and capabilities.
Return JSON: {"justifications": [{"title": "...", "justification": "..."}]}"""


def _generate_justifications(
    roles: list[RecommendedRole],
    need_by_id: dict[str, RoleNeed],
    system_rep: SystemRepresentation,
) -> list[RecommendedRole]:
    """Generate justification narratives for each role."""
    if not roles:
        return roles

    # Build context about modules
    module_summaries = {
        m.name: {
            "type": m.type.value,
            "description": m.description[:150],
            "user_actions": m.user_actions[:3],
        }
        for m in system_rep.modules
    }

    # Batch roles for efficiency
    batch_size = 10
    justified_roles: list[RecommendedRole] = []

    for batch_start in range(0, len(roles), batch_size):
        batch = roles[batch_start : batch_start + batch_size]

        role_infos = []
        for role in batch:
            # Gather source module info
            source_modules = set()
            for need_id in role.mapped_role_needs:
                need = need_by_id.get(need_id)
                if need and need.source_module:
                    source_modules.add(need.source_module)

            role_infos.append({
                "title": role.canonical_title,
                "category": role.category.value,
                "seniority": role.seniority.value,
                "source_modules": list(source_modules),
                "transformation_type": role.transformation.transformation_type.value,
                "transformation_rationale": role.transformation.rationale,
                "needs_count": len(role.mapped_role_needs),
            })

        result = ask_claude_json(
            system_prompt=_JUSTIFICATION_SYSTEM,
            user_prompt=(
                f"System modules for context:\n{json.dumps(module_summaries, indent=2)}\n\n"
                f"Roles to justify:\n{json.dumps(role_infos, indent=2)}"
            ),
        )

        justification_map: dict[str, str] = {}
        if isinstance(result, dict) and "justifications" in result:
            for j in result["justifications"]:
                justification_map[j.get("title", "")] = j.get("justification", "")
        elif isinstance(result, list):
            for j in result:
                justification_map[j.get("title", "")] = j.get("justification", "")

        for role in batch:
            justification = justification_map.get(role.canonical_title, "")
            if not justification:
                # Fallback: generate a basic justification
                needs = [need_by_id[nid] for nid in role.mapped_role_needs if nid in need_by_id]
                modules = {n.source_module for n in needs if n.source_module}
                justification = (
                    f"This {role.seniority.value.replace('_', ' ')} role is needed to "
                    f"{needs[0].description[:100] if needs else 'operate system functions'}. "
                    f"It interacts with: {', '.join(modules) if modules else 'multiple system modules'}."
                )
            justified_roles.append(role.model_copy(update={"justification": justification}))

    return justified_roles
