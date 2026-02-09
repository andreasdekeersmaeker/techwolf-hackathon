"""Phase 3: Derive operational role needs from user-facing modules and obligations."""

from __future__ import annotations

import hashlib
import json
import logging

from models.schemas import (
    InteractionPattern,
    OrganizationalObligations,
    RoleCategory,
    RoleNeed,
    RoleTransformation,
    SenioritySignal,
    SystemRepresentation,
    TransformationType,
    UserFacingModule,
)
from services.claude_client import ask_claude_json

log = logging.getLogger(__name__)


def run(system_rep: SystemRepresentation) -> list[RoleNeed]:
    """Phase 3: Derive role needs from the system representation."""
    log.info("Phase 3: Deriving role needs")

    # Step 3.1 — Role needs from modules
    module_needs = _derive_needs_from_modules(system_rep.modules)
    log.info("Derived %d role needs from modules", len(module_needs))

    # Step 3.2 — Cross-cutting needs from obligations
    obligation_needs = _derive_needs_from_obligations(system_rep.obligations)
    log.info("Derived %d cross-cutting role needs from obligations", len(obligation_needs))

    all_needs = module_needs + obligation_needs

    # Step 3.3 — Deduplicate and merge
    deduplicated = _deduplicate_needs(all_needs)
    log.info("After deduplication: %d role needs", len(deduplicated))

    return deduplicated


# ---------------------------------------------------------------------------
# Module-based role needs
# ---------------------------------------------------------------------------

_MODULE_NEEDS_SYSTEM = """You are an organizational design expert. Given a user-facing module of a software system,
identify what HUMAN ROLES are needed to operate it.

This system is already built, hosted, and maintained externally. You are identifying roles that USE the system —
business, clinical, administrative, analytical, or operational roles.

DO NOT suggest: software engineers, data engineers, DevOps, QA engineers, product managers, IT administrators,
or any role responsible for building/maintaining the platform.

For each role need, provide:
- description: what this person does with/through this module (1-2 sentences)
- category: one of: clinical, administrative, analytical, compliance_governance, operational, training_change_mgmt, supervisory
- interaction_pattern: one of: primary_daily_user, periodic_reviewer, exception_handler, oversight_approver, configuration_owner
- domain_expertise: list of domain knowledge areas required (e.g., "Medical coding", "Financial reporting")
- system_skills: skills for using the system (e.g., "Dashboard interpretation", "Workflow management")
- seniority_signal: one of: entry_level, experienced, senior_specialist, leadership
  Use entry_level for routine data entry; experienced for domain judgment;
  senior_specialist for high-stakes decisions or regulated actions; leadership for oversight/accountability
- derived_job_titles: 2-5 plausible job titles, each MAX 10 WORDS (this is critical — titles must be short)
- derived_skill_keywords: up to 20 skill keywords, comma-separable (e.g., "patient intake, medical records, triage, EHR navigation")
- transformation_type: one of: existing_unchanged, existing_augmented, existing_consolidated, newly_created
- transformation_rationale: brief explanation of how this role relates to legacy roles

Return a JSON array of role need objects."""


def _derive_needs_from_modules(modules: list[UserFacingModule]) -> list[RoleNeed]:
    """Derive role needs from each user-facing module via LLM."""
    needs: list[RoleNeed] = []

    for module in modules:
        module_desc = json.dumps({
            "name": module.name,
            "type": module.type.value,
            "description": module.description,
            "user_actions": module.user_actions,
            "decisions_required": module.decisions_required,
            "domain_knowledge_needed": module.domain_knowledge_needed,
            "frequency": module.frequency.value,
            "access_sensitivity": module.access_sensitivity.value,
        }, indent=2)

        result = ask_claude_json(
            system_prompt=_MODULE_NEEDS_SYSTEM,
            user_prompt=f"Module:\n{module_desc}",
        )

        if not isinstance(result, list):
            continue

        for i, item in enumerate(result):
            need_id = hashlib.md5(
                f"{module.name}|{i}|{item.get('description', '')}".encode()
            ).hexdigest()[:10]

            try:
                # Enforce title length constraint
                titles = item.get("derived_job_titles", [])
                titles = [t for t in titles if len(t.split()) <= 10][:5]

                # Enforce skill keywords constraint
                skills = item.get("derived_skill_keywords", [])
                if isinstance(skills, str):
                    skills = [s.strip() for s in skills.split(",")]
                skills = skills[:20]

                transformation = RoleTransformation(
                    transformation_type=TransformationType(
                        item.get("transformation_type", "existing_unchanged")
                    ),
                    rationale=item.get("transformation_rationale", ""),
                )

                need = RoleNeed(
                    id=need_id,
                    description=item.get("description", ""),
                    category=RoleCategory(item.get("category", "operational")),
                    interaction_pattern=InteractionPattern(
                        item.get("interaction_pattern", "primary_daily_user")
                    ),
                    domain_expertise=item.get("domain_expertise", []),
                    system_skills=item.get("system_skills", []),
                    seniority_signal=SenioritySignal(
                        item.get("seniority_signal", "experienced")
                    ),
                    derived_job_titles=titles,
                    derived_skill_keywords=skills,
                    source_module=module.name,
                    transformation=transformation,
                )
                needs.append(need)
            except (ValueError, KeyError) as e:
                log.warning("Skipping malformed role need: %s", e)

    return needs


# ---------------------------------------------------------------------------
# Obligation-based cross-cutting role needs
# ---------------------------------------------------------------------------

_OBLIGATION_NEEDS_SYSTEM = """You are an organizational design expert. Given organizational obligations
that arise from adopting a software system, identify cross-cutting HUMAN ROLES needed to fulfill them.

These are roles that exist BECAUSE the system exists — they may be:
- Compliance officers ensuring the organization meets regulatory requirements when using the system
- Training coordinators helping staff adopt the system
- Change managers transitioning from legacy processes
- Quality/audit analysts monitoring automated outputs
- Vendor relationship managers coordinating with the system vendor
- Data governance roles classifying and managing data within the system

DO NOT suggest technical roles (engineers, DevOps, DBAs, etc.).

For each role, provide the same fields as before:
description, category, interaction_pattern, domain_expertise, system_skills, seniority_signal,
derived_job_titles (MAX 10 words each, 2-5 titles),
derived_skill_keywords (up to 20 terms),
transformation_type, transformation_rationale

Return a JSON array."""


def _derive_needs_from_obligations(obligations: OrganizationalObligations) -> list[RoleNeed]:
    """Derive cross-cutting role needs from organizational obligations."""
    oblig_dict = obligations.model_dump()

    # Only proceed if there are actual obligations
    has_content = any(bool(v) for v in oblig_dict.values())
    if not has_content:
        return []

    result = ask_claude_json(
        system_prompt=_OBLIGATION_NEEDS_SYSTEM,
        user_prompt=f"Organizational obligations:\n{json.dumps(oblig_dict, indent=2)}",
    )

    needs: list[RoleNeed] = []
    if not isinstance(result, list):
        return needs

    for i, item in enumerate(result):
        need_id = hashlib.md5(
            f"obligation|{i}|{item.get('description', '')}".encode()
        ).hexdigest()[:10]

        try:
            titles = item.get("derived_job_titles", [])
            titles = [t for t in titles if len(t.split()) <= 10][:5]

            skills = item.get("derived_skill_keywords", [])
            if isinstance(skills, str):
                skills = [s.strip() for s in skills.split(",")]
            skills = skills[:20]

            # Determine source obligations
            source_obligs = []
            desc_lower = item.get("description", "").lower()
            for key, values in oblig_dict.items():
                if values and any(v.lower() in desc_lower for v in values):
                    source_obligs.append(key)

            transformation = RoleTransformation(
                transformation_type=TransformationType(
                    item.get("transformation_type", "newly_created")
                ),
                rationale=item.get("transformation_rationale", ""),
            )

            need = RoleNeed(
                id=need_id,
                description=item.get("description", ""),
                category=RoleCategory(item.get("category", "compliance_governance")),
                interaction_pattern=InteractionPattern(
                    item.get("interaction_pattern", "periodic_reviewer")
                ),
                domain_expertise=item.get("domain_expertise", []),
                system_skills=item.get("system_skills", []),
                seniority_signal=SenioritySignal(
                    item.get("seniority_signal", "senior_specialist")
                ),
                derived_job_titles=titles,
                derived_skill_keywords=skills,
                source_obligations=source_obligs or ["organizational_obligations"],
                transformation=transformation,
            )
            needs.append(need)
        except (ValueError, KeyError) as e:
            log.warning("Skipping malformed obligation role need: %s", e)

    return needs


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

_DEDUP_SYSTEM = """You are deduplicating a list of role needs. Some may be semantically redundant —
describing the same type of person from different angles.

Group the role needs by their IDs. Return a JSON array of groups, where each group is:
{
  "keep_id": "the ID of the most comprehensive/representative need to keep",
  "merge_ids": ["IDs to merge into the kept one"],
  "merged_titles": ["union of all derived_job_titles from merged needs"],
  "merged_skills": ["union of all derived_skill_keywords from merged needs"],
  "highest_seniority": "the highest seniority_signal among all needs in the group"
}

If a role need is unique (not redundant with any other), still include it as a group of one.
Only merge needs that truly describe the same role. Different seniority levels of the same
functional role should be kept separate."""


def _deduplicate_needs(needs: list[RoleNeed]) -> list[RoleNeed]:
    """Deduplicate semantically redundant role needs using LLM."""
    if len(needs) <= 3:
        return needs

    need_summaries = []
    for n in needs:
        need_summaries.append({
            "id": n.id,
            "description": n.description[:200],
            "category": n.category.value,
            "seniority": n.seniority_signal.value,
            "titles": n.derived_job_titles,
        })

    result = ask_claude_json(
        system_prompt=_DEDUP_SYSTEM,
        user_prompt=f"Role needs to deduplicate:\n{json.dumps(need_summaries, indent=2)}",
    )

    if not isinstance(result, list):
        return needs

    need_by_id = {n.id: n for n in needs}
    merged_ids: set[str] = set()
    deduplicated: list[RoleNeed] = []

    for group in result:
        keep_id = group.get("keep_id", "")
        merge_ids = group.get("merge_ids", [])

        if keep_id not in need_by_id:
            continue

        kept = need_by_id[keep_id]

        # Merge titles and skills from merged needs
        all_titles = list(kept.derived_job_titles)
        all_skills = list(kept.derived_skill_keywords)
        all_modules = [kept.source_module] if kept.source_module else []

        for mid in merge_ids:
            if mid in need_by_id and mid != keep_id:
                merged = need_by_id[mid]
                all_titles.extend(merged.derived_job_titles)
                all_skills.extend(merged.derived_skill_keywords)
                if merged.source_module:
                    all_modules.append(merged.source_module)
                merged_ids.add(mid)

        # Deduplicate titles/skills preserving order
        seen_titles: set[str] = set()
        unique_titles = []
        for t in all_titles:
            tl = t.lower()
            if tl not in seen_titles:
                seen_titles.add(tl)
                unique_titles.append(t)

        seen_skills: set[str] = set()
        unique_skills = []
        for s in all_skills:
            sl = s.lower()
            if sl not in seen_skills:
                seen_skills.add(sl)
                unique_skills.append(s)

        # Apply highest seniority
        highest = group.get("highest_seniority", kept.seniority_signal.value)
        try:
            seniority = SenioritySignal(highest)
        except ValueError:
            seniority = kept.seniority_signal

        updated = kept.model_copy(update={
            "derived_job_titles": unique_titles[:5],
            "derived_skill_keywords": unique_skills[:20],
            "seniority_signal": seniority,
        })
        deduplicated.append(updated)

    # Add any needs that weren't in any group
    for n in needs:
        if n.id not in merged_ids and n.id not in {d.id for d in deduplicated}:
            deduplicated.append(n)

    return deduplicated
