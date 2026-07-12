"""
Grant-specific fit checker: does THIS proposal satisfy THIS grant's stated
funding criteria?

This is deliberately the counterpart to app/proposal_vet.py, not a
replacement for it: proposal_vet.py asks a grant-agnostic completeness
question ("does this proposal have a budget, a timeline, etc.?"); this module
asks a grant-specific fit question ("does this proposal meet grant X's
501(c)(3) requirement, budget cap, sustainability-plan requirement, etc.?").
Both are advisory-only -- neither ever drafts or rewrites proposal content.

Grants in grants_db/grants.json carry an optional `criteria` field: a list of
discrete, individually checkable requirement strings (see grant_ingest.py /
csv_to_json.py). Because the number and wording of criteria varies per grant,
the JSON schema we hand to Ollama's `format` param is built dynamically per
call -- one fixed, indexed, required property per criterion (criterion_0,
criterion_1, ...) so the model can't drop, merge, reorder, or invent items,
the same anti-hallucination trick proposal_vet.py uses with its static
rubric, just applied to a variable-length list. We key results by index
rather than by criterion text to avoid any length/collision/invalid-character
issues with using arbitrary requirement text as a JSON property name.

If a grant hasn't been ingested with structured `criteria` yet (all of the
current grants.json sample entries), we fall back to checking the proposal
against its free-text `eligibility` blurb as a single item, and say so
explicitly in the returned result -- we never fabricate criteria that aren't
in the record.
"""
from __future__ import annotations

from pathlib import Path

from . import chunking, grants_matcher, ollama_client, pdf_extract

# Proposals can be 20+ pages; see ollama_client.extract_json's num_ctx
# override. This doesn't touch the global OLLAMA_NUM_CTX default used by
# every other call in the project.
GRANT_FIT_NUM_CTX = 65536

_STATUS_VALUES = ["met", "partial", "not_met", "unclear"]

# Ranking used to merge per-segment results when a proposal is long enough to be
# chunked: a requirement satisfied in any segment is satisfied by the proposal,
# so the strongest status wins. "unclear" outranks "not_met" because ambiguous
# relevant text is more informative than nothing found. See
# chunking.merge_objects.
_STATUS_PRIORITY = {"met": 3, "partial": 2, "unclear": 1, "not_met": 0}

GRANT_FIT_PROMPT_TEMPLATE = """You are checking a funding proposal against ONE SPECIFIC grant program's stated \
requirements -- this is NOT a generic completeness check, it's a point-by-point fit check against this particular \
funder's rules.

Grant: {grant_name} (funder: {funder})

For each of the following {n} requirement(s), decide whether the proposal satisfies it:

{criteria_list}

For each requirement, give:
- status: "met" (clearly satisfied), "partial" (partially addressed, ambiguous, or only implied), "not_met" (not \
addressed, or the proposal contradicts the requirement), or "unclear" (the proposal doesn't give enough information \
to judge either way).
- evidence: if there is ANY relevant text in the proposal (status "met", "partial", or "unclear"), evidence MUST \
include a short excerpt copied verbatim, character-for-character, from the proposal text -- wrapped in double \
quotes, with a page/section reference outside the quotes if you can identify one (e.g. 'p.4, Organization section: \
"..."'). Do not paraphrase or summarize the words inside the quotes, and do not skip words or sentences from the \
middle of the quoted span -- a reviewer needs to find this exact text in the source document. You may add a brief \
reasoning sentence alongside the quote to explain why it does or doesn't satisfy the requirement. Leave evidence \
empty only if status is "not_met" and there is nothing relevant in the text at all.
- suggestion: plain-language, advisory suggestion telling the proposal's human author what to add, fix, or clarify \
to satisfy this requirement, and why it matters for this funder. Do NOT draft or rewrite proposal text, section \
content, or example wording for them to copy in -- only describe what they should go address themselves. This tool \
is advisory-only and must never generate replacement proposal content.

--- PROPOSAL TEXT ---
{text}
{tables}
--- END PROPOSAL TEXT ---
"""


def _build_schema(criteria_texts: list[str]) -> dict:
    """Build a JSON schema with one fixed, required, indexed property per
    criterion (criterion_0, criterion_1, ...) so the model can't drop,
    reorder, merge, or invent items -- mirrors proposal_vet.py's static-rubric
    trick, adapted for a per-grant variable-length criteria list.
    """
    properties = {}
    for i in range(len(criteria_texts)):
        key = f"criterion_{i}"
        properties[key] = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": _STATUS_VALUES,
                    "description": "Whether this specific requirement is met, partially met, not met, or unclear from the proposal text.",
                },
                "evidence": {
                    "type": "string",
                    "description": (
                        "If status is 'met', 'partial', or 'unclear': must contain a verbatim excerpt copied "
                        "character-for-character from the proposal text, wrapped in double quotes -- never "
                        "paraphrase what's inside the quotes -- plus an optional page/section reference "
                        "outside the quotes, and optionally a brief reasoning sentence. Empty string only if "
                        "status is 'not_met' with nothing relevant in the text."
                    ),
                },
                "suggestion": {
                    "type": "string",
                    "description": (
                        "A plain-language, advisory suggestion telling the proposal's human author what to "
                        "add, fix, or clarify to satisfy this requirement. Do NOT draft or rewrite proposal "
                        "text -- describe what's missing/weak, never write the replacement content yourself."
                    ),
                },
            },
            "required": ["status", "evidence", "suggestion"],
        }
    return {"type": "object", "properties": properties, "required": list(properties.keys())}


def _format_criteria_list(criteria_texts: list[str]) -> str:
    return "\n".join(f"{i + 1}. {text}" for i, text in enumerate(criteria_texts))


def _build_prompt(grant: dict, criteria_texts: list[str], text: str, tables_section: str) -> str:
    return GRANT_FIT_PROMPT_TEMPLATE.format(
        grant_name=grant.get("name", ""),
        funder=grant.get("funder", ""),
        n=len(criteria_texts),
        criteria_list=_format_criteria_list(criteria_texts),
        text=text,
        tables=tables_section,
    )


def _run_grant_fit(
    grant: dict, criteria_texts: list[str], schema: dict, text: str, tables_section: str, model: str | None
) -> dict:
    """Check the proposal text against the grant's criteria.

    Short proposals (every one seen so far) go through as a single call, exactly
    as before. A proposal too long for the context window is split into
    overlapping segments, checked segment by segment against the same criteria,
    and merged back into one result by strongest status per criterion.
    """
    if not chunking.needs_chunking(text, GRANT_FIT_NUM_CTX):
        return ollama_client.extract_json(
            _build_prompt(grant, criteria_texts, text, tables_section), schema, model=model, num_ctx=GRANT_FIT_NUM_CTX
        )

    segments = chunking.split_text(text, chunking.char_budget(GRANT_FIT_NUM_CTX))
    per_segment = [
        ollama_client.extract_json(
            _build_prompt(grant, criteria_texts, segment, tables_section), schema, model=model, num_ctx=GRANT_FIT_NUM_CTX
        )
        for segment in segments
    ]
    keys = [f"criterion_{i}" for i in range(len(criteria_texts))]
    return chunking.merge_objects(per_segment, keys, _STATUS_PRIORITY)


def _select_grant(proposal_text: str, grant_id: str | None) -> tuple[dict, str]:
    """Return (grant_record, matched_via) or raise ValueError if none found."""
    if grant_id:
        grants = grants_matcher.load_grants()
        match = next((g for g in grants if g.get("id") == grant_id), None)
        if match is None:
            raise ValueError(f"No grant with id '{grant_id}' found in the grants database.")
        return match, "explicit_grant_id"

    shortlist = grants_matcher.shortlist_grants(proposal_text, top_k=3)
    if not shortlist:
        raise ValueError(
            "No grants available to match against -- the grants database appears to be empty, and no "
            "--grant-id was provided."
        )
    return shortlist[0].grant, "auto_shortlist"


def _compose_deficit_summary(criteria: list[dict]) -> list[dict]:
    """Unmet/partial criteria in priority order: not_met before partial."""
    not_met = [c for c in criteria if c["status"] == "not_met"]
    partial = [c for c in criteria if c["status"] == "partial"]
    return [
        {"criterion": c["criterion"], "status": c["status"], "suggestion": c["suggestion"]}
        for c in (not_met + partial)
    ]


def vet_grant_fit(pdf_path: str | Path, grant_id: str | None = None, model: str | None = None) -> dict:
    """Check a proposal PDF against one grant's specific funding criteria.

    If grant_id is omitted, the best-fitting grant is auto-selected via
    grants_matcher.shortlist_grants(). Extracts the full proposal text (no
    truncation -- see GRANT_FIT_NUM_CTX) and asks the local model to judge
    the proposal against each of the grant's `criteria` items via a
    schema-constrained single-shot call. If the grant has no structured
    `criteria` yet, falls back to checking against its free-text
    `eligibility` blurb as a single item and flags this in the result.

    Returns:
        {
            "source_pdf": str,
            "page_count": int,
            "grant": {"id": ..., "name": ..., "funder": ...},
            "matched_via": "explicit_grant_id" | "auto_shortlist",
            "structured_criteria": bool,
            "criteria": [
                {"criterion": str, "status": "met"|"partial"|"not_met"|"unclear",
                 "evidence": ..., "suggestion": ...},
                ...
            ],
            "deficit_summary": [ {"criterion", "status", "suggestion"}, ... ],  # not_met before partial
            "note": str,  # present only when falling back to eligibility
            "full_text": str,  # source proposal text, for clause look-up in a UI
        }

    Never auto-rewrites proposal content -- suggestions are advisory prose
    only, enforced both by the prompt instructions and the schema's
    description fields. Raises ValueError if no grant can be resolved or the
    resolved grant has nothing to check against (no criteria and no
    eligibility text).
    """
    extracted = pdf_extract.extract_pdf(pdf_path)
    tables_block = pdf_extract.summarize_tables(extracted.tables)
    tables_section = f"\n[Extracted tables]\n{tables_block}\n" if tables_block else ""

    grant, matched_via = _select_grant(extracted.text, grant_id)

    structured_criteria = bool(grant.get("criteria"))
    if structured_criteria:
        criteria_texts = list(grant["criteria"])
        note = ""
    else:
        eligibility = grant.get("eligibility", "").strip()
        if not eligibility:
            raise ValueError(
                f"Grant '{grant.get('id')}' has neither structured criteria nor an eligibility "
                "description to check the proposal against."
            )
        criteria_texts = [eligibility]
        note = (
            f"Grant '{grant.get('id')}' hasn't been ingested with structured criteria yet -- checked "
            "against its free-text eligibility description instead. Re-ingest this grant (or add a "
            "'criteria' column) for a more precise, point-by-point check."
        )

    schema = _build_schema(criteria_texts)
    raw = _run_grant_fit(grant, criteria_texts, schema, extracted.text, tables_section, model)

    criteria = []
    for i, criterion_text in enumerate(criteria_texts):
        item = raw.get(f"criterion_{i}", {}) or {}
        status = item.get("status", "unclear")
        if status not in _STATUS_VALUES:
            status = "unclear"
        criteria.append({
            "criterion": criterion_text,
            "status": status,
            "evidence": item.get("evidence", ""),
            "suggestion": item.get("suggestion", ""),
        })

    result = {
        "source_pdf": str(pdf_path),
        "page_count": extracted.page_count,
        "grant": {
            "id": grant.get("id", ""),
            "name": grant.get("name", ""),
            "funder": grant.get("funder", ""),
        },
        "matched_via": matched_via,
        "structured_criteria": structured_criteria,
        "criteria": criteria,
        "deficit_summary": _compose_deficit_summary(criteria),
        "full_text": extracted.text,
        # Raw pdfplumber tables (list of rows of cells), so a UI can render actual
        # tables in the document view instead of the jumbled inline cell text that
        # full_text necessarily contains (page.extract_text() flattens tables too).
        "tables": extracted.tables,
    }
    if note:
        result["note"] = note
    return result


if __name__ == "__main__":
    import sys

    pdf_arg = sys.argv[1]
    grant_id_arg = sys.argv[2] if len(sys.argv) > 2 else None
    result = vet_grant_fit(pdf_arg, grant_id=grant_id_arg)
    print(f"Matched grant: {result['grant']['name']} ({result['grant']['id']}) via {result['matched_via']}")
    if result.get("note"):
        print(f"Note: {result['note']}")
    for c in result["criteria"]:
        print(f"[{c['status'].upper():7}] {c['criterion']}")
        print(f"    evidence: {c['evidence']}")
        print(f"    suggestion: {c['suggestion']}")
    print("\n=== Deficit summary ===")
    for d in result["deficit_summary"]:
        print(f"[{d['status'].upper():7}] {d['criterion']} -- {d['suggestion']}")
