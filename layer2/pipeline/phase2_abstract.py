"""Phase 2: Extract user-facing modules, organizational obligations, and workflow map."""

from __future__ import annotations

import json
import logging

from models.schemas import (
    AccessSensitivity,
    ChunkCategory,
    ContentChunk,
    Frequency,
    ModuleType,
    OrganizationalObligations,
    SystemRepresentation,
    UserFacingModule,
    WorkflowEdge,
    WorkflowInteractionMap,
)
from services.claude_client import ask_claude_json

log = logging.getLogger(__name__)


def run(chunks: list[ContentChunk]) -> SystemRepresentation:
    """Phase 2: Build user-facing system representation from content chunks."""
    log.info("Phase 2: Extracting system representation from %d chunks", len(chunks))

    modules = _extract_modules(chunks)
    log.info("Extracted %d user-facing modules", len(modules))

    obligations = _extract_obligations(chunks)
    log.info("Extracted organizational obligations")

    workflow_map = _build_workflow_map(modules)
    log.info("Built workflow map with %d edges", len(workflow_map.edges))

    return SystemRepresentation(
        modules=modules,
        obligations=obligations,
        workflow_map=workflow_map,
    )


# ---------------------------------------------------------------------------
# Module extraction
# ---------------------------------------------------------------------------

_MODULE_EXTRACTION_SYSTEM = """You are an expert business analyst examining a software system's documentation.
Your task is to identify USER-FACING MODULES — the parts of the system that human users interact with.

This system is already built and hosted. You are identifying what users DO with it, not how it is built.

For each module, extract:
- name: a clear descriptive name
- type: one of: workflow, dashboard, form, automation_surface, compliance_surface, config_surface, reporting_module
- description: what this module does from the user's perspective
- user_actions: specific actions a human performs here (e.g., "Reviews flagged records", "Approves referrals")
- decisions_required: judgments the system cannot make alone that require human expertise
- domain_knowledge_needed: specialized knowledge required (e.g., "ICD-10 coding", "GDPR data subject rights")
- frequency: how often users interact — continuous, daily, weekly, periodic, or event_driven
- access_sensitivity: public, internal, restricted, or highly_restricted

Return a JSON array of module objects. If the content doesn't describe any user-facing functionality, return an empty array."""


def _extract_modules(chunks: list[ContentChunk]) -> list[UserFacingModule]:
    """Extract user-facing modules from content chunks using LLM."""
    # Group chunks by category for more focused extraction
    category_groups: dict[ChunkCategory, list[ContentChunk]] = {}
    for chunk in chunks:
        category_groups.setdefault(chunk.category, []).append(chunk)

    all_modules: list[UserFacingModule] = []
    seen_names: set[str] = set()

    for category, cat_chunks in category_groups.items():
        # Combine chunk texts for context
        combined_text = ""
        chunk_ids = []
        for chunk in cat_chunks:
            section = f"\n\n## {' > '.join(chunk.heading_path) if chunk.heading_path else 'Section'}\n{chunk.text}"
            if len(combined_text) + len(section) > 12000:
                break
            combined_text += section
            chunk_ids.append(chunk.id)

        if not combined_text.strip():
            continue

        result = ask_claude_json(
            system_prompt=_MODULE_EXTRACTION_SYSTEM,
            user_prompt=(
                f"Content category: {category.value}\n\n"
                f"Website content:\n{combined_text}\n\n"
                "Extract all user-facing modules from this content."
            ),
        )

        if not isinstance(result, list):
            continue

        for item in result:
            name = item.get("name", "")
            if not name or name.lower() in seen_names:
                continue
            seen_names.add(name.lower())

            try:
                module = UserFacingModule(
                    name=name,
                    type=ModuleType(item.get("type", "workflow")),
                    description=item.get("description", ""),
                    user_actions=item.get("user_actions", []),
                    decisions_required=item.get("decisions_required", []),
                    domain_knowledge_needed=item.get("domain_knowledge_needed", []),
                    frequency=Frequency(item.get("frequency", "daily")),
                    access_sensitivity=AccessSensitivity(
                        item.get("access_sensitivity", "internal")
                    ),
                    source_chunks=chunk_ids,
                )
                all_modules.append(module)
            except (ValueError, KeyError) as e:
                log.warning("Skipping malformed module %s: %s", name, e)

    return all_modules


# ---------------------------------------------------------------------------
# Obligations extraction
# ---------------------------------------------------------------------------

_OBLIGATIONS_SYSTEM = """You are an expert in organizational governance examining a software system's documentation.
The system is already built, hosted, and maintained by a vendor. Your task is to identify what OBLIGATIONS
the adopting organization must fulfill when using this system.

These are NOT technical obligations (the vendor handles those). These are:
- Compliance obligations (GDPR, HIPAA, SOC2, industry regulations)
- Data governance (classifying data, responding to access requests, retention policies)
- Training needs (what users need to learn to use the system effectively)
- Change management (transitioning from legacy processes, retiring old workflows)
- Oversight needs (monitoring automated decisions, quality assurance of outputs)
- Vendor liaison (coordinating with the vendor on configuration, feature requests)

Return a JSON object with these exact keys:
compliance_regimes, data_governance_needs, training_needs, change_management_needs, oversight_needs, vendor_liaison_needs
Each value is an array of strings."""


def _extract_obligations(chunks: list[ContentChunk]) -> OrganizationalObligations:
    """Extract organizational obligations from all chunks."""
    # Focus on compliance/governance chunks but include others for context
    priority_cats = {
        ChunkCategory.compliance_governance,
        ChunkCategory.administration_config,
        ChunkCategory.automation,
    }

    text_parts = []
    for chunk in chunks:
        if chunk.category in priority_cats:
            text_parts.insert(0, chunk.text[:1000])
        else:
            text_parts.append(chunk.text[:500])

    combined = "\n\n---\n\n".join(text_parts)[:15000]

    result = ask_claude_json(
        system_prompt=_OBLIGATIONS_SYSTEM,
        user_prompt=f"System documentation content:\n\n{combined}",
    )

    if not isinstance(result, dict):
        return OrganizationalObligations()

    return OrganizationalObligations(
        compliance_regimes=result.get("compliance_regimes", []),
        data_governance_needs=result.get("data_governance_needs", []),
        training_needs=result.get("training_needs", []),
        change_management_needs=result.get("change_management_needs", []),
        oversight_needs=result.get("oversight_needs", []),
        vendor_liaison_needs=result.get("vendor_liaison_needs", []),
    )


# ---------------------------------------------------------------------------
# Workflow map
# ---------------------------------------------------------------------------

_WORKFLOW_MAP_SYSTEM = """You are mapping how user-facing modules in a software system connect to each other
from a WORKFLOW perspective. You are given a list of modules. Identify which modules feed into others —
where one module's output becomes another module's input, or where users move from one module to the next
as part of a business process.

Return a JSON array of edge objects, each with:
- from_module: name of the source module
- to_module: name of the destination module
- handoff_description: brief description of what passes between them (e.g., "Approved intake records flow to...")

Only include edges where there is a clear workflow connection. Do not force connections."""


def _build_workflow_map(modules: list[UserFacingModule]) -> WorkflowInteractionMap:
    """Use LLM to identify workflow connections between modules."""
    if len(modules) < 2:
        return WorkflowInteractionMap()

    module_summaries = []
    for m in modules:
        module_summaries.append({
            "name": m.name,
            "type": m.type.value,
            "description": m.description,
            "user_actions": m.user_actions[:5],
        })

    result = ask_claude_json(
        system_prompt=_WORKFLOW_MAP_SYSTEM,
        user_prompt=f"Modules:\n{json.dumps(module_summaries, indent=2)}",
    )

    edges = []
    module_names = {m.name for m in modules}
    if isinstance(result, list):
        for item in result:
            from_m = item.get("from_module", "")
            to_m = item.get("to_module", "")
            if from_m in module_names and to_m in module_names:
                edges.append(WorkflowEdge(
                    from_module=from_m,
                    to_module=to_m,
                    handoff_description=item.get("handoff_description", ""),
                ))

    return WorkflowInteractionMap(edges=edges)


# ---------------------------------------------------------------------------
# User-facing summary for checkpoint
# ---------------------------------------------------------------------------

def format_summary(rep: SystemRepresentation) -> str:
    """Format the system representation for user review at Checkpoint 1."""
    lines = ["=" * 60, "SYSTEM INTERACTION MODEL — Review", "=" * 60, ""]

    lines.append(f"User-Facing Modules ({len(rep.modules)}):")
    lines.append("-" * 40)
    for m in rep.modules:
        lines.append(f"  [{m.type.value.upper()}] {m.name}")
        lines.append(f"    {m.description[:120]}")
        if m.user_actions:
            lines.append(f"    Actions: {', '.join(m.user_actions[:3])}")
        if m.decisions_required:
            lines.append(f"    Decisions: {', '.join(m.decisions_required[:2])}")
        lines.append("")

    oblig = rep.obligations
    lines.append("Organizational Obligations:")
    lines.append("-" * 40)
    if oblig.compliance_regimes:
        lines.append(f"  Compliance: {', '.join(oblig.compliance_regimes)}")
    if oblig.data_governance_needs:
        lines.append(f"  Data governance: {', '.join(oblig.data_governance_needs[:3])}")
    if oblig.training_needs:
        lines.append(f"  Training: {', '.join(oblig.training_needs[:3])}")
    if oblig.oversight_needs:
        lines.append(f"  Oversight: {', '.join(oblig.oversight_needs[:3])}")
    lines.append("")

    lines.append(f"Workflow Connections ({len(rep.workflow_map.edges)}):")
    lines.append("-" * 40)
    for edge in rep.workflow_map.edges:
        lines.append(f"  {edge.from_module} → {edge.to_module}")
        if edge.handoff_description:
            lines.append(f"    ({edge.handoff_description})")

    return "\n".join(lines)
