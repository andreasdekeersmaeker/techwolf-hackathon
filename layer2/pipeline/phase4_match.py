"""Phase 4: Dual-channel JobBERT retrieval, exclusion filter, LLM re-ranking, clustering."""

from __future__ import annotations

import json
import logging
from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from config import CLUSTER_DISTANCE_THRESHOLD, RERANK_THRESHOLD, RETRIEVAL_TOP_K
from models.schemas import (
    ClusterInfo,
    RecommendedRole,
    RetrievalChannel,
    RetrievalHit,
    RetrievalResult,
    RoleCategory,
    RoleNeed,
    RoleTransformation,
    ScoringDetail,
    TransformationType,
)
from services.claude_client import ask_claude_json
from services.jobbert_service import embed_skill_lists, embed_titles
from services.vacancy_store import VacancyStore, is_excluded_role

log = logging.getLogger(__name__)


def run(
    role_needs: list[RoleNeed],
    store: VacancyStore,
) -> tuple[list[RecommendedRole], list[RetrievalResult], list[ScoringDetail], list[ClusterInfo]]:
    """Phase 4: Match role needs to vacancies and cluster into recommended roles."""
    log.info("Phase 4: Matching %d role needs to vacancies", len(role_needs))

    # Step 4.1 + 4.2: Dual-channel retrieval
    retrieval_results = _dual_channel_retrieval(role_needs, store)
    log.info("Retrieval complete for %d needs", len(retrieval_results))

    # Step 4.3: Apply exclusion filter
    for rr in retrieval_results:
        rr.hits = _apply_exclusion_filter(rr.hits, store)
    log.info("Exclusion filter applied")

    # Step 4.4: LLM re-ranking
    all_scoring: list[ScoringDetail] = []
    for rr in retrieval_results:
        need = next((n for n in role_needs if n.id == rr.role_need_id), None)
        if not need:
            continue
        scored = _rerank(need, rr.hits, store)
        rr.scored = scored
        all_scoring.extend(scored)
    log.info("Re-ranking complete: %d scored pairs", len(all_scoring))

    # Step 4.5: Cluster into roles
    roles, clusters = _cluster_into_roles(role_needs, retrieval_results, store)
    log.info("Clustering complete: %d recommended roles", len(roles))

    return roles, retrieval_results, all_scoring, clusters


# ---------------------------------------------------------------------------
# Step 4.1 + 4.2: Dual-channel retrieval
# ---------------------------------------------------------------------------

def _dual_channel_retrieval(
    role_needs: list[RoleNeed],
    store: VacancyStore,
) -> list[RetrievalResult]:
    """Run title-to-title and skills-to-title retrieval for each role need."""
    results: list[RetrievalResult] = []

    for need in role_needs:
        all_hits: dict[str, RetrievalHit] = {}  # vacancy_id -> best hit

        # Channel A: Title-to-Title
        if need.derived_job_titles:
            title_embeddings = embed_titles(need.derived_job_titles)
            search_results = store.search(title_embeddings, top_k=RETRIEVAL_TOP_K)

            for q_idx, hits in enumerate(search_results):
                query_title = need.derived_job_titles[q_idx]
                for rec_idx, score in hits:
                    rec = store.get_record(rec_idx)
                    vid = rec["identifier"]
                    hit = RetrievalHit(
                        vacancy_id=vid,
                        vacancy_title=rec.get("title", ""),
                        enriched_job_title=rec.get("enriched_job_title", ""),
                        cosine_score=score,
                        channel=RetrievalChannel.title,
                        query_used=query_title,
                    )
                    if vid not in all_hits or score > all_hits[vid].cosine_score:
                        all_hits[vid] = hit

        # Channel B: Skills-to-Title
        if need.derived_skill_keywords:
            skill_string = ", ".join(need.derived_skill_keywords[:20])
            skill_embeddings = embed_skill_lists([skill_string])
            search_results = store.search(skill_embeddings, top_k=RETRIEVAL_TOP_K)

            for rec_idx, score in search_results[0]:
                rec = store.get_record(rec_idx)
                vid = rec["identifier"]
                hit = RetrievalHit(
                    vacancy_id=vid,
                    vacancy_title=rec.get("title", ""),
                    enriched_job_title=rec.get("enriched_job_title", ""),
                    cosine_score=score,
                    channel=RetrievalChannel.skills,
                    query_used=skill_string[:100],
                )
                if vid in all_hits:
                    # Dual match — keep higher score, mark as dual
                    existing = all_hits[vid]
                    best_score = max(existing.cosine_score, score)
                    all_hits[vid] = existing.model_copy(update={
                        "cosine_score": best_score,
                        "channel": RetrievalChannel.dual,
                    })
                elif vid not in all_hits:
                    all_hits[vid] = hit

        results.append(RetrievalResult(
            role_need_id=need.id,
            hits=list(all_hits.values()),
        ))

    return results


# ---------------------------------------------------------------------------
# Step 4.3: Exclusion filter
# ---------------------------------------------------------------------------

def _apply_exclusion_filter(
    hits: list[RetrievalHit], store: VacancyStore
) -> list[RetrievalHit]:
    """Remove hits matching excluded role categories."""
    filtered = []
    for hit in hits:
        rec = store.get_record_by_id(hit.vacancy_id)
        title = hit.enriched_job_title or (rec.get("enriched_job_title", "") if rec else "")
        desc = rec.get("description", "") if rec else ""
        if not is_excluded_role(title, desc):
            filtered.append(hit)
    return filtered


# ---------------------------------------------------------------------------
# Step 4.4: LLM re-ranking
# ---------------------------------------------------------------------------

_RERANK_SYSTEM = """You are evaluating how well a job vacancy matches an operational role need.
The role need describes what a human user of a software system must do.
The vacancy is from a real job database.

Score the match on three axes (0-5 each):
- task_score: Do the vacancy's tasks align with what the role need requires?
- domain_score: Do the vacancy's skills/description match the required domain expertise?
- seniority_score: Does the vacancy's apparent seniority match the required level?

Also provide a brief rationale (1 sentence).

Return JSON: {"task_score": N, "domain_score": N, "seniority_score": N, "rationale": "..."}"""


def _rerank(
    need: RoleNeed,
    hits: list[RetrievalHit],
    store: VacancyStore,
) -> list[ScoringDetail]:
    """Re-rank hits using LLM scoring. Batches for efficiency."""
    if not hits:
        return []

    scored: list[ScoringDetail] = []

    # Process in batches of 5 for efficiency
    batch_size = 5
    for batch_start in range(0, len(hits), batch_size):
        batch = hits[batch_start : batch_start + batch_size]

        vacancy_infos = []
        for hit in batch:
            rec = store.get_record_by_id(hit.vacancy_id)
            if rec:
                vacancy_infos.append({
                    "vacancy_id": hit.vacancy_id,
                    "enriched_job_title": rec.get("enriched_job_title", ""),
                    "enriched_skills": rec.get("enriched_skills", ""),
                    "enriched_tasks": rec.get("enriched_tasks", ""),
                    "description_preview": rec.get("description", "")[:300],
                })

        if not vacancy_infos:
            continue

        need_desc = {
            "description": need.description,
            "category": need.category.value,
            "domain_expertise": need.domain_expertise,
            "seniority": need.seniority_signal.value,
        }

        result = ask_claude_json(
            system_prompt=_RERANK_SYSTEM,
            user_prompt=(
                f"Role need:\n{json.dumps(need_desc, indent=2)}\n\n"
                f"Score each of these vacancies against the role need. "
                f"Return a JSON array of scoring objects, one per vacancy.\n\n"
                f"Vacancies:\n{json.dumps(vacancy_infos, indent=2)}"
            ),
        )

        if isinstance(result, dict):
            result = [result]

        if isinstance(result, list):
            for i, score_data in enumerate(result):
                if i >= len(vacancy_infos):
                    break
                vi = vacancy_infos[i]
                task_s = float(score_data.get("task_score", 0))
                domain_s = float(score_data.get("domain_score", 0))
                seniority_s = float(score_data.get("seniority_score", 0))
                composite = 0.40 * task_s + 0.40 * domain_s + 0.20 * seniority_s

                if composite >= RERANK_THRESHOLD:
                    scored.append(ScoringDetail(
                        vacancy_id=vi["vacancy_id"],
                        enriched_job_title=vi["enriched_job_title"],
                        role_need_id=need.id,
                        task_score=task_s,
                        domain_score=domain_s,
                        seniority_score=seniority_s,
                        composite_score=composite,
                        rationale=score_data.get("rationale", ""),
                    ))

    return scored


# ---------------------------------------------------------------------------
# Step 4.5: Clustering
# ---------------------------------------------------------------------------

def _cluster_into_roles(
    role_needs: list[RoleNeed],
    retrieval_results: list[RetrievalResult],
    store: VacancyStore,
) -> tuple[list[RecommendedRole], list[ClusterInfo]]:
    """Cluster surviving vacancies into distinct recommended roles."""
    need_by_id = {n.id: n for n in role_needs}

    # Collect all surviving vacancy titles with their associations
    vacancy_need_map: dict[str, list[str]] = defaultdict(list)  # enriched_title -> [need_ids]
    vacancy_ids_map: dict[str, list[str]] = defaultdict(list)   # enriched_title -> [vacancy_ids]
    vacancy_scores: dict[str, list[float]] = defaultdict(list)  # enriched_title -> [scores]
    vacancy_channels: dict[str, set[str]] = defaultdict(set)    # enriched_title -> channels

    for rr in retrieval_results:
        for sd in rr.scored:
            title = sd.enriched_job_title
            vacancy_need_map[title].append(sd.role_need_id)
            vacancy_ids_map[title].append(sd.vacancy_id)
            vacancy_scores[title].append(sd.composite_score)
            # Find the channel from hits
            for hit in rr.hits:
                if hit.vacancy_id == sd.vacancy_id:
                    vacancy_channels[title].add(hit.channel.value)

    unique_titles = list(vacancy_need_map.keys())

    if not unique_titles:
        log.warning("No vacancies survived re-ranking. Returning empty results.")
        return [], []

    if len(unique_titles) == 1:
        # Single title — no clustering needed
        title = unique_titles[0]
        need_ids = list(set(vacancy_need_map[title]))
        channels = vacancy_channels[title]
        channel = (
            RetrievalChannel.dual if "dual" in channels
            else RetrievalChannel.skills if "skills" in channels
            else RetrievalChannel.title
        )
        avg_score = np.mean(vacancy_scores[title])

        # Determine category and seniority from most common need
        categories = [need_by_id[nid].category for nid in need_ids if nid in need_by_id]
        seniorities = [need_by_id[nid].seniority_signal for nid in need_ids if nid in need_by_id]
        transformations = [need_by_id[nid].transformation for nid in need_ids if nid in need_by_id]

        role = RecommendedRole(
            canonical_title=title,
            alternative_titles=[],
            mapped_role_needs=need_ids,
            representative_vacancy_ids=list(set(vacancy_ids_map[title]))[:5],
            category=_most_common(categories, RoleCategory.operational),
            seniority=_most_common(seniorities, SenioritySignal.experienced),
            confidence=float(avg_score),
            retrieval_channel=channel,
            transformation=transformations[0] if transformations else RoleTransformation(
                transformation_type=TransformationType.existing_unchanged
            ),
        )
        cluster = ClusterInfo(
            cluster_id=0,
            canonical_title=title,
            member_titles=[title],
            member_vacancy_ids=list(set(vacancy_ids_map[title]))[:10],
        )
        return [role], [cluster]

    # Embed unique titles for clustering
    title_embeddings = embed_titles(unique_titles)

    # Agglomerative clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=CLUSTER_DISTANCE_THRESHOLD,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(title_embeddings)

    # Group by cluster
    cluster_groups: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_groups[label].append(idx)

    roles: list[RecommendedRole] = []
    clusters: list[ClusterInfo] = []

    for cluster_id, title_indices in cluster_groups.items():
        cluster_titles = [unique_titles[i] for i in title_indices]
        cluster_embeddings = title_embeddings[title_indices]

        # Find centroid and closest title
        centroid = cluster_embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        distances = 1 - cluster_embeddings @ centroid
        canonical_idx = title_indices[int(np.argmin(distances))]
        canonical_title = unique_titles[canonical_idx]

        # Aggregate data from all titles in cluster
        all_need_ids: set[str] = set()
        all_vacancy_ids: set[str] = set()
        all_scores: list[float] = []
        all_channels: set[str] = set()

        for title in cluster_titles:
            all_need_ids.update(vacancy_need_map.get(title, []))
            all_vacancy_ids.update(vacancy_ids_map.get(title, []))
            all_scores.extend(vacancy_scores.get(title, []))
            all_channels.update(vacancy_channels.get(title, set()))

        channels = all_channels
        channel = (
            RetrievalChannel.dual if "dual" in channels
            else RetrievalChannel.skills if "skills" in channels
            else RetrievalChannel.title
        )

        # Determine category, seniority, interaction pattern from needs
        need_ids_list = list(all_need_ids)
        categories = [need_by_id[nid].category for nid in need_ids_list if nid in need_by_id]
        seniorities = [need_by_id[nid].seniority_signal for nid in need_ids_list if nid in need_by_id]
        patterns = [need_by_id[nid].interaction_pattern for nid in need_ids_list if nid in need_by_id]
        transformations = [need_by_id[nid].transformation for nid in need_ids_list if nid in need_by_id]

        role = RecommendedRole(
            canonical_title=canonical_title,
            alternative_titles=[t for t in cluster_titles if t != canonical_title],
            mapped_role_needs=need_ids_list,
            representative_vacancy_ids=list(all_vacancy_ids)[:5],
            category=_most_common(categories, RoleCategory.operational),
            interaction_pattern=_most_common(patterns, None),
            seniority=_most_common(seniorities, SenioritySignal.experienced),
            confidence=float(np.mean(all_scores)) if all_scores else 0.0,
            retrieval_channel=channel,
            transformation=transformations[0] if transformations else RoleTransformation(
                transformation_type=TransformationType.existing_unchanged
            ),
        )
        roles.append(role)

        cluster_info = ClusterInfo(
            cluster_id=cluster_id,
            canonical_title=canonical_title,
            member_titles=cluster_titles,
            member_vacancy_ids=list(all_vacancy_ids)[:10],
            centroid_distance=float(np.min(distances)),
        )
        clusters.append(cluster_info)

    # Sort by confidence descending
    roles.sort(key=lambda r: r.confidence, reverse=True)

    return roles, clusters


def _most_common(items: list, default):
    """Return the most common item in a list."""
    if not items:
        return default
    from collections import Counter
    counter = Counter(items)
    return counter.most_common(1)[0][0]
