"""Phase 1: Crawl the Layer 1 website and extract structured content chunks."""

from __future__ import annotations

import hashlib
import logging
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag

from config import LAYER1_URL
from models.schemas import ChunkCategory, ContentChunk
from services.claude_client import ask_claude_json

log = logging.getLogger(__name__)


def run(base_url: str = LAYER1_URL) -> list[ContentChunk]:
    """Crawl the Layer 1 site and return ContentChunks."""
    log.info("Phase 1: Crawling %s", base_url)

    pages = _crawl_site(base_url)
    log.info("Crawled %d pages", len(pages))

    chunks: list[ContentChunk] = []
    for url, html in pages.items():
        page_chunks = _extract_chunks_from_page(url, html)
        chunks.extend(page_chunks)

    log.info("Extracted %d raw chunks", len(chunks))

    # Classify uncategorized chunks with LLM
    uncat = [c for c in chunks if c.category == ChunkCategory.uncategorized]
    if uncat:
        log.info("Classifying %d uncategorized chunks with LLM", len(uncat))
        chunks = _reclassify_chunks(chunks)

    log.info("Phase 1 complete: %d content chunks", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Crawling
# ---------------------------------------------------------------------------

def _crawl_site(base_url: str) -> dict[str, str]:
    """Crawl all internal pages starting from base_url. Returns {url: html}."""
    visited: dict[str, str] = {}
    to_visit = [base_url]
    base_parsed = urlparse(base_url)
    base_prefix = base_url.rstrip("/")

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        while to_visit:
            url = to_visit.pop(0)
            if url in visited:
                continue

            try:
                resp = client.get(url)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                log.warning("Failed to fetch %s: %s", url, e)
                continue

            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type:
                continue

            visited[url] = resp.text

            # Extract internal links
            soup = BeautifulSoup(resp.text, "lxml")
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                absolute = urljoin(url, href).split("#")[0].split("?")[0]
                parsed = urlparse(absolute)
                if (
                    parsed.netloc == base_parsed.netloc
                    and absolute.startswith(base_prefix)
                    and absolute not in visited
                    and absolute not in to_visit
                ):
                    to_visit.append(absolute)

    return visited


# ---------------------------------------------------------------------------
# Chunk extraction
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: dict[ChunkCategory, list[str]] = {
    ChunkCategory.workflows: [
        "workflow", "process", "step", "journey", "approval", "pipeline",
        "sequence", "procedure", "flow",
    ],
    ChunkCategory.dashboards_reports: [
        "dashboard", "report", "analytics", "kpi", "chart", "metric",
        "visualization", "overview", "summary", "monitor",
    ],
    ChunkCategory.data_entry: [
        "form", "input", "entry", "register", "submit", "capture",
        "record", "fill", "create new",
    ],
    ChunkCategory.automation: [
        "automat", "trigger", "rule", "notification", "alert", "schedule",
        "cron", "event-driven", "batch",
    ],
    ChunkCategory.compliance_governance: [
        "compliance", "audit", "gdpr", "hipaa", "regulatory", "consent",
        "privacy", "access control", "permission", "governance", "policy",
    ],
    ChunkCategory.integrations_user_facing: [
        "integration", "import", "export", "sync", "connect", "api",
        "external system", "third-party", "interop",
    ],
    ChunkCategory.administration_config: [
        "configuration", "settings", "admin", "manage users", "template",
        "customize", "preferences", "setup", "threshold",
    ],
}


def _classify_chunk(heading_path: list[str], text: str) -> ChunkCategory:
    """Heuristic classification based on keywords in heading and text."""
    combined = " ".join(heading_path).lower() + " " + text[:500].lower()
    scores: dict[ChunkCategory, int] = {}
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in keywords if kw in combined)

    if not scores or max(scores.values()) == 0:
        return ChunkCategory.uncategorized

    return max(scores, key=lambda c: scores[c])


def _extract_chunks_from_page(url: str, html: str) -> list[ContentChunk]:
    """Parse a single HTML page into ContentChunks."""
    soup = BeautifulSoup(html, "lxml")

    # Remove script, style, nav, footer
    for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    body = soup.find("body")
    if not body:
        return []

    chunks: list[ContentChunk] = []
    # Walk through heading-delimited sections
    sections = _split_by_headings(body)
    for heading_path, content_elements in sections:
        text = _elements_to_text(content_elements)
        if len(text.strip()) < 20:
            continue

        tables = _extract_tables(content_elements)
        category = _classify_chunk(heading_path, text)
        chunk_id = hashlib.md5(
            (url + "|" + "|".join(heading_path) + "|" + text[:100]).encode()
        ).hexdigest()[:12]

        chunks.append(ContentChunk(
            id=chunk_id,
            source_url=url,
            category=category,
            heading_path=heading_path,
            text=text,
            tables=tables,
        ))

    return chunks


def _split_by_headings(
    root: Tag,
) -> list[tuple[list[str], list[Tag]]]:
    """Split a container into sections by heading tags.

    Returns list of (heading_path, [content_elements]).
    """
    heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}
    sections: list[tuple[list[str], list[Tag]]] = []

    current_path: list[str] = []
    current_elements: list[Tag] = []

    for child in root.children:
        if not isinstance(child, Tag):
            continue

        if child.name in heading_tags:
            # Save previous section
            if current_elements:
                sections.append((list(current_path), current_elements))
                current_elements = []

            level = int(child.name[1])
            heading_text = child.get_text(strip=True)
            # Adjust path to current level
            current_path = current_path[: level - 1]
            current_path.append(heading_text)
        else:
            current_elements.append(child)

    # Last section
    if current_elements:
        sections.append((list(current_path), current_elements))

    # If no headings found, treat entire body as one chunk
    if not sections:
        sections.append(([], [root]))

    return sections


def _elements_to_text(elements: list[Tag]) -> str:
    """Extract clean text from a list of elements."""
    parts = []
    for el in elements:
        text = el.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _extract_tables(elements: list[Tag]) -> list[dict]:
    """Extract table data as list of dicts."""
    tables = []
    for el in elements:
        for table_tag in (el.find_all("table") if hasattr(el, "find_all") else []):
            rows = table_tag.find_all("tr")
            if not rows:
                continue
            headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
            table_data = []
            for row in rows[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if cells:
                    table_data.append(dict(zip(headers, cells)))
            if table_data:
                tables.append({"headers": headers, "rows": table_data})
    return tables


# ---------------------------------------------------------------------------
# LLM reclassification for uncategorized chunks
# ---------------------------------------------------------------------------

def _reclassify_chunks(chunks: list[ContentChunk]) -> list[ContentChunk]:
    """Use Claude to reclassify uncategorized chunks."""
    categories = [c.value for c in ChunkCategory if c != ChunkCategory.uncategorized]
    uncat_chunks = [(i, c) for i, c in enumerate(chunks) if c.category == ChunkCategory.uncategorized]

    if not uncat_chunks:
        return chunks

    # Batch them — send up to 20 at a time
    batch_size = 20
    for batch_start in range(0, len(uncat_chunks), batch_size):
        batch = uncat_chunks[batch_start : batch_start + batch_size]
        items = []
        for _, chunk in batch:
            items.append({
                "id": chunk.id,
                "heading_path": chunk.heading_path,
                "text_preview": chunk.text[:300],
            })

        result = ask_claude_json(
            system_prompt=(
                "You classify website content sections into categories. "
                "Each section comes from a software system's documentation website. "
                "Classify each into exactly one of these categories:\n"
                + "\n".join(f"- {c}" for c in categories)
            ),
            user_prompt=(
                "Classify each of the following sections. Return a JSON array of objects "
                'with "id" and "category" fields.\n\n'
                + json.dumps(items, indent=2)
            ),
        )

        if isinstance(result, list):
            id_to_cat = {item["id"]: item.get("category", "uncategorized") for item in result}
            for idx, chunk in batch:
                new_cat = id_to_cat.get(chunk.id, "uncategorized")
                try:
                    chunks[idx] = chunk.model_copy(update={"category": ChunkCategory(new_cat)})
                except ValueError:
                    pass  # keep uncategorized

    return chunks


# Need json import for the reclassify function
import json
