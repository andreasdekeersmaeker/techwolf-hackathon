"""Phase 5: Coverage analysis, gap detection, and follow-up question generation."""

from __future__ import annotations

import logging

from models.schemas import RecommendedRole, RetrievalResult, RoleNeed

log = logging.getLogger(__name__)


class CoverageReport:
    """Result of coverage analysis."""

    def __init__(self) -> None:
        self.covered_needs: list[str] = []
        self.uncovered_needs: list[RoleNeed] = []
        self.low_confidence_roles: list[RecommendedRole] = []
        self.follow_up_questions: list[str] = []
        self.coverage_pct: float = 0.0

    @property
    def has_gaps(self) -> bool:
        return bool(self.uncovered_needs) or bool(self.follow_up_questions)


def run(
    role_needs: list[RoleNeed],
    roles: list[RecommendedRole],
    retrieval_results: list[RetrievalResult],
) -> CoverageReport:
    """Phase 5: Analyze coverage and generate follow-up questions if needed."""
    log.info("Phase 5: Analyzing coverage")

    report = CoverageReport()

    # Check coverage: every role need should map to at least one recommended role
    covered_need_ids: set[str] = set()
    for role in roles:
        covered_need_ids.update(role.mapped_role_needs)

    for need in role_needs:
        if need.id in covered_need_ids:
            report.covered_needs.append(need.id)
        else:
            report.uncovered_needs.append(need)

    total = len(role_needs)
    report.coverage_pct = (len(report.covered_needs) / total * 100) if total > 0 else 100.0

    log.info(
        "Coverage: %d/%d needs covered (%.1f%%)",
        len(report.covered_needs), total, report.coverage_pct,
    )

    # Identify low-confidence roles
    for role in roles:
        if role.confidence < 3.5:
            report.low_confidence_roles.append(role)

    if report.low_confidence_roles:
        log.info("%d roles with low confidence (< 3.5)", len(report.low_confidence_roles))

    # Generate follow-up questions based on specific conditions
    _generate_follow_up_questions(report, role_needs, roles)

    if report.follow_up_questions:
        log.info("Generated %d follow-up questions", len(report.follow_up_questions))
    else:
        log.info("No follow-up questions needed — proceeding autonomously")

    return report


def _generate_follow_up_questions(
    report: CoverageReport,
    role_needs: list[RoleNeed],
    roles: list[RecommendedRole],
) -> None:
    """Generate targeted follow-up questions based on gaps and ambiguities."""

    # Condition 1: Uncovered needs
    if report.uncovered_needs:
        need_descriptions = [n.description[:100] for n in report.uncovered_needs[:5]]
        report.follow_up_questions.append(
            "I couldn't find matching roles for these operational needs:\n"
            + "\n".join(f"  - {d}" for d in need_descriptions)
            + "\nCan you describe who handles these in your current organization?"
        )

    # Condition 2: Domain ambiguity — needs that touch multiple domains
    domain_counts: dict[str, int] = {}
    for need in role_needs:
        for exp in need.domain_expertise:
            domain_counts[exp] = domain_counts.get(exp, 0) + 1

    ambiguous_domains = [d for d, count in domain_counts.items() if count >= 3]
    if len(ambiguous_domains) > 3:
        report.follow_up_questions.append(
            f"Multiple modules require overlapping domain expertise in: "
            f"{', '.join(ambiguous_domains[:5])}. "
            f"Should these be handled by generalist roles or domain specialists?"
        )

    # Condition 3: Consolidation opportunity — many needs mapping to similar roles
    from collections import Counter
    title_freq = Counter()
    for role in roles:
        title_freq[role.canonical_title] += len(role.mapped_role_needs)

    heavily_loaded = [t for t, count in title_freq.items() if count >= 5]
    if heavily_loaded:
        report.follow_up_questions.append(
            f"The following roles cover many responsibilities: "
            f"{', '.join(heavily_loaded[:3])}. "
            f"Should these be split into separate positions or kept as consolidated roles?"
        )


def format_report(report: CoverageReport) -> str:
    """Format coverage report for user display."""
    lines = ["=" * 60, "COVERAGE ANALYSIS", "=" * 60, ""]
    lines.append(f"Coverage: {report.coverage_pct:.1f}%")
    lines.append(f"Covered needs: {len(report.covered_needs)}")
    lines.append(f"Uncovered needs: {len(report.uncovered_needs)}")
    lines.append(f"Low-confidence roles: {len(report.low_confidence_roles)}")
    lines.append("")

    if report.uncovered_needs:
        lines.append("Uncovered Needs:")
        for need in report.uncovered_needs:
            lines.append(f"  - {need.description[:120]}")
        lines.append("")

    if report.low_confidence_roles:
        lines.append("Low-Confidence Roles:")
        for role in report.low_confidence_roles:
            lines.append(f"  - {role.canonical_title} (confidence: {role.confidence:.2f})")
        lines.append("")

    if report.follow_up_questions:
        lines.append("Follow-Up Questions:")
        for i, q in enumerate(report.follow_up_questions, 1):
            lines.append(f"\n  Q{i}: {q}")

    return "\n".join(lines)
