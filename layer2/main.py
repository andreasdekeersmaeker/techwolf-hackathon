#!/usr/bin/env python3
"""Layer 2 — Agentic Role Recommendation System.

Main orchestrator that runs the full pipeline and serves the results website.

Usage:
    # One-time: preprocess vacancies (build JobBERT index)
    python main.py preprocess [--max-records N]

    # Run the full pipeline and serve the website
    python main.py run [--skip-checkpoint] [--save PATH]

    # Serve previously saved results
    python main.py serve --load PATH
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the layer2 directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, LAYER1_URL, LAYER2_PORT
from models.schemas import PipelineOutput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("layer2")


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Preprocess vacancies: embed titles with JobBERT and build FAISS index."""
    from services.vacancy_store import VacancyStore

    store = VacancyStore()
    if store.is_indexed and not args.force:
        log.info("Index already exists. Use --force to rebuild.")
        return
    store.preprocess(max_records=args.max_records)


def cmd_run(args: argparse.Namespace) -> None:
    """Run the full pipeline and serve results."""
    from services.vacancy_store import VacancyStore

    store = VacancyStore()
    if not store.is_indexed:
        log.error(
            "Vacancy index not found. Run 'python main.py preprocess' first."
        )
        sys.exit(1)

    store.load_index()

    # ---- Phase 1: Ingest ----
    from pipeline import phase1_ingest
    chunks = phase1_ingest.run(args.layer1_url)

    # ---- Phase 2: Abstract ----
    from pipeline import phase2_abstract
    system_rep = phase2_abstract.run(chunks)

    # ---- Checkpoint 1: User validation ----
    if not args.skip_checkpoint:
        summary = phase2_abstract.format_summary(system_rep)
        print("\n" + summary + "\n")
        answer = input(
            "Is this system representation correct? [Y/n/edit] > "
        ).strip().lower()
        if answer in ("n", "no"):
            print("Please re-run with adjusted Layer 1 content. Exiting.")
            sys.exit(0)
        elif answer == "edit":
            print(
                "Manual editing of the system representation is not yet supported.\n"
                "Proceeding with current representation."
            )

    # ---- Phase 3: Needs ----
    from pipeline import phase3_needs
    role_needs = phase3_needs.run(system_rep)

    # ---- Phase 4: Match ----
    from pipeline import phase4_match
    roles, retrieval_results, scoring_breakdowns, clustering_output = (
        phase4_match.run(role_needs, store)
    )

    # ---- Phase 5: Reason ----
    from pipeline import phase5_reason
    coverage = phase5_reason.run(role_needs, roles, retrieval_results)

    # ---- Checkpoint 2: Conditional follow-up ----
    if coverage.has_gaps and not args.skip_checkpoint:
        report_text = phase5_reason.format_report(coverage)
        print("\n" + report_text + "\n")

        if coverage.follow_up_questions:
            print("The agent has follow-up questions:\n")
            for i, q in enumerate(coverage.follow_up_questions, 1):
                print(f"  Q{i}: {q}\n")

            answer = input("Provide answers or press Enter to skip > ").strip()
            if answer:
                log.info("User provided feedback: %s", answer[:200])
                # In a full implementation, this would feed back into Phase 3-4
                # For now, we proceed with current results
                log.info("Proceeding with current results (feedback noted).")

    # ---- Phase 6: Assemble ----
    from pipeline import phase6_assemble
    output = phase6_assemble.run(
        roles=roles,
        role_needs=role_needs,
        coverage=coverage,
        system_rep=system_rep,
        chunks=chunks,
        retrieval_results=retrieval_results,
        scoring_breakdowns=scoring_breakdowns,
        clustering_output=clustering_output,
    )

    # Save results
    save_path = Path(args.save) if args.save else DATA_DIR / "pipeline_output.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(
        output.model_dump_json(indent=2), encoding="utf-8"
    )
    log.info("Pipeline output saved to %s", save_path)

    # Serve website
    _serve(output, args.port)


def cmd_serve(args: argparse.Namespace) -> None:
    """Serve previously saved results."""
    load_path = Path(args.load)
    if not load_path.exists():
        log.error("File not found: %s", load_path)
        sys.exit(1)

    log.info("Loading pipeline output from %s", load_path)
    raw = load_path.read_text(encoding="utf-8")
    output = PipelineOutput.model_validate_json(raw)
    log.info(
        "Loaded: %d roles, %d chunks, %d needs",
        output.roster.metadata.total_roles,
        len(output.intermediate.content_chunks),
        len(output.intermediate.role_needs),
    )
    _serve(output, args.port)


def _serve(output: PipelineOutput, port: int) -> None:
    """Start the FastAPI server."""
    import uvicorn
    from server import app, set_pipeline_output

    set_pipeline_output(output)
    log.info("Starting server at http://localhost:%d/jobrecommendations", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer 2 — Agentic Role Recommendation System"
    )
    sub = parser.add_subparsers(dest="command")

    # preprocess
    p_pre = sub.add_parser("preprocess", help="Build JobBERT vacancy index")
    p_pre.add_argument("--max-records", type=int, default=None,
                       help="Limit number of vacancy records to process")
    p_pre.add_argument("--force", action="store_true",
                       help="Rebuild index even if it exists")

    # run
    p_run = sub.add_parser("run", help="Run full pipeline and serve results")
    p_run.add_argument("--layer1-url", default=LAYER1_URL,
                       help="Layer 1 website URL")
    p_run.add_argument("--skip-checkpoint", action="store_true",
                       help="Skip interactive checkpoints")
    p_run.add_argument("--save", default=None,
                       help="Path to save pipeline output JSON")
    p_run.add_argument("--port", type=int, default=LAYER2_PORT,
                       help="Server port (default: 8001)")

    # serve
    p_serve = sub.add_parser("serve", help="Serve saved results")
    p_serve.add_argument("--load", required=True,
                         help="Path to pipeline output JSON")
    p_serve.add_argument("--port", type=int, default=LAYER2_PORT,
                         help="Server port (default: 8001)")

    args = parser.parse_args()

    if args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
