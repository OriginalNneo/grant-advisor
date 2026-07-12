"""
Text-based interface. Examples:

    python -m app.cli analyze path/to/project.pdf
    python -m app.cli analyze path/to/project.pdf --model gemma4:e4b
    python -m app.cli chat path/to/project.pdf      # analyze, then drop into an interactive follow-up chat
    python -m app.cli add-grant path/to/grant.pdf   # extract a grant PDF into grants_db/grants.json
    python -m app.cli describe "a community arts festival for 200 attendees, need ~$10k"
    python -m app.cli vet-proposal path/to/proposal.pdf   # grant-agnostic completeness check
    python -m app.cli vet-grant-fit path/to/proposal.pdf [--grant-id ID]   # check fit against one grant's criteria
    python -m app.cli categories "a community arts festival for 200 attendees"   # rank matching grant categories
"""
from __future__ import annotations

import argparse
import sys

from . import agent, category_match, grant_fit, grant_ingest, ollama_client, proposal_vet


def cmd_analyze(args: argparse.Namespace) -> None:
    print(f"Extracting and analyzing {args.pdf} ...\n", file=sys.stderr)
    try:
        result = agent.analyze_pdf(args.pdf, model=args.model)
    except ollama_client.OllamaUnavailable as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("=== Shortlisted matches (local semantic search) ===")
    for g in result["shortlist"]:
        print(f"  {g['score']:.2f}  {g['name']}  ({g['id']})")
    print("\n=== Recommendation ===")
    print(result["summary"])

    if args.chat:
        _interactive_loop(result["conversation"], args.model)


def cmd_chat(args: argparse.Namespace) -> None:
    try:
        result = agent.analyze_pdf(args.pdf, model=args.model)
    except ollama_client.OllamaUnavailable as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("=== Recommendation ===")
    print(result["summary"])
    _interactive_loop(result["conversation"], args.model)


def cmd_describe(args: argparse.Namespace) -> None:
    text = " ".join(args.text)
    try:
        result = agent.analyze_text(text, model=args.model)
    except ollama_client.OllamaUnavailable as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("=== Shortlisted matches (local semantic search) ===")
    for g in result["shortlist"]:
        print(f"  {g['score']:.2f}  {g['name']}  ({g['id']})")
    print("\n=== Recommendation ===")
    print(result["summary"])

    if args.chat:
        _interactive_loop(result["conversation"], args.model)


def cmd_add_grant(args: argparse.Namespace) -> None:
    print(f"Extracting grant record from {args.pdf} ...\n", file=sys.stderr)
    try:
        grant = grant_ingest.ingest_grant_pdf(args.pdf, model=args.model)
    except ollama_client.OllamaUnavailable as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Added '{grant['id']}': {grant['name']} (funder: {grant['funder']})")
    print(f"  focus_areas: {', '.join(grant['focus_areas'])}")
    print(f"  eligibility: {grant['eligibility']}")
    print("  Verify amounts/deadlines/eligibility on the funder's own site before relying on this.")


# --- vet-proposal: grant-agnostic completeness check (advisory-only; see app/proposal_vet.py) ---

def cmd_vet_proposal(args: argparse.Namespace) -> None:
    print(f"Extracting and vetting {args.pdf} for completeness ...\n", file=sys.stderr)
    try:
        result = proposal_vet.vet_proposal_completeness(args.pdf, model=args.model)
    except ollama_client.OllamaUnavailable as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"=== Completeness check: {result['source_pdf']} ({result['page_count']} pages) ===\n")
    for c in result["criteria"]:
        print(f"[{c['status'].upper():7}] {c['label']}")
        if c["evidence"]:
            print(f"    evidence:   {c['evidence']}")
        if c["suggestion"]:
            print(f"    suggestion: {c['suggestion']}")
        print()
    print("=== Overall ===")
    print(result["overall_summary"])
    print(
        "\nNote: this is a grant-agnostic completeness check only (it does not check fit against "
        "any specific grant's criteria), and it is advisory -- review each item yourself; nothing "
        "here is auto-rewritten proposal text."
    )

# --- end vet-proposal block ---


# --- vet-grant-fit: fit check against one grant's specific criteria (advisory-only; see app/grant_fit.py) ---

def cmd_vet_grant_fit(args: argparse.Namespace) -> None:
    print(f"Extracting and checking {args.pdf} against grant criteria ...\n", file=sys.stderr)
    try:
        result = grant_fit.vet_grant_fit(args.pdf, grant_id=args.grant_id, model=args.model)
    except ollama_client.OllamaUnavailable as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    grant = result["grant"]
    print(f"=== Grant fit check: {result['source_pdf']} ({result['page_count']} pages) ===")
    print(f"Matched grant: {grant['name']} ({grant['id']}), funder: {grant['funder']} [{result['matched_via']}]\n")
    if result.get("note"):
        print(f"Note: {result['note']}\n")

    for c in result["criteria"]:
        print(f"[{c['status'].upper():7}] {c['criterion']}")
        if c["evidence"]:
            print(f"    evidence:   {c['evidence']}")
        if c["suggestion"]:
            print(f"    suggestion: {c['suggestion']}")
        print()

    print("=== Deficit summary (not_met before partial) ===")
    if result["deficit_summary"]:
        for d in result["deficit_summary"]:
            print(f"[{d['status'].upper():7}] {d['criterion']}")
            print(f"    suggestion: {d['suggestion']}")
    else:
        print("No unmet or partially-met criteria found.")
    print(
        "\nNote: this checks fit against one specific grant's criteria only, and it is advisory -- review "
        "each item yourself; nothing here is auto-rewritten proposal text."
    )

# --- end vet-grant-fit block ---


# --- categories: rank the grants db's focus_areas categories by correlation with a text description ---

def cmd_categories(args: argparse.Namespace) -> None:
    text = " ".join(args.text)
    results = category_match.score_categories(text, top_k=args.top_k)
    if not results:
        print("No focus_areas categories found in the grants database (or empty text).")
        return
    print(f"=== Top {len(results)} matching categories ===")
    for r in results:
        print(f"  {r['score']:.3f}  {r['category']}")

# --- end categories block ---


def _interactive_loop(conversation: list[dict], model: str | None) -> None:
    print("\nFollow-up chat. Type 'exit' to quit.\n")
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message or message.lower() in {"exit", "quit"}:
            break
        try:
            result = agent.continue_chat(conversation, message, model=model)
        except ollama_client.OllamaUnavailable as e:
            print(f"Error: {e}")
            continue
        conversation = result["conversation"]
        print(f"\nadvisor> {result['summary']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="grant-advisor", description="Local, Ollama-powered grant advisor.")
    parser.add_argument("--model", default=None, help="Override the Ollama chat model (default: $OLLAMA_CHAT_MODEL or llama3.1)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="Analyze a PDF and print grant recommendations.")
    p_analyze.add_argument("pdf", help="Path to the project/organization PDF.")
    p_analyze.add_argument("--chat", action="store_true", help="Drop into interactive follow-up chat after analyzing.")
    p_analyze.set_defaults(func=cmd_analyze)

    p_chat = sub.add_parser("chat", help="Analyze a PDF then go straight into interactive follow-up chat.")
    p_chat.add_argument("pdf", help="Path to the project/organization PDF.")
    p_chat.set_defaults(func=cmd_chat)

    p_add_grant = sub.add_parser("add-grant", help="Extract a grant program PDF and add it to grants_db/grants.json.")
    p_add_grant.add_argument("pdf", help="Path to the grant program's PDF (guidelines/factsheet).")
    p_add_grant.set_defaults(func=cmd_add_grant)

    p_describe = sub.add_parser("describe", help="Describe a project/event in plain text (no PDF) and get grant recommendations.")
    p_describe.add_argument("text", nargs="+", help="Free-text description, e.g. what event you want to organize.")
    p_describe.add_argument("--chat", action="store_true", help="Drop into interactive follow-up chat after analyzing.")
    p_describe.set_defaults(func=cmd_describe)

    # --- vet-proposal: grant-agnostic completeness check ---
    p_vet_proposal = sub.add_parser(
        "vet-proposal",
        help="Grant-agnostic completeness check on a proposal PDF (advisory-only; no grant matching).",
    )
    p_vet_proposal.add_argument("pdf", help="Path to the event/funding proposal PDF.")
    p_vet_proposal.set_defaults(func=cmd_vet_proposal)
    # --- end vet-proposal block ---

    # --- vet-grant-fit: fit check against one grant's specific criteria ---
    p_vet_grant_fit = sub.add_parser(
        "vet-grant-fit",
        help="Check a proposal PDF against one grant's specific funding criteria (advisory-only).",
    )
    p_vet_grant_fit.add_argument("pdf", help="Path to the event/funding proposal PDF.")
    p_vet_grant_fit.add_argument(
        "--grant-id",
        default=None,
        help="Grant id to check against (see 'python -m app.cli' output or grants_db/grants.json). "
        "If omitted, the best-fitting grant is auto-selected via semantic search.",
    )
    p_vet_grant_fit.set_defaults(func=cmd_vet_grant_fit)
    # --- end vet-grant-fit block ---

    # --- categories: rank matching grant categories ---
    p_categories = sub.add_parser(
        "categories",
        help="Rank the grants database's focus_areas categories (e.g. STEM, Sustainability) by correlation with a text description.",
    )
    p_categories.add_argument("text", nargs="+", help="Free-text description of the project/proposal.")
    p_categories.add_argument("--top-k", type=int, default=3, dest="top_k", help="Number of top categories to show (default: 3).")
    p_categories.set_defaults(func=cmd_categories)
    # --- end categories block ---

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
