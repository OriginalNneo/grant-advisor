"""
Grant-agnostic completeness check for event/funding proposals.

This is deliberately NOT grant-matching: it doesn't check a proposal against
any specific funder's criteria (a parallel workstream owns that). Instead it
asks a much narrower question -- does this proposal contain the concepts any
reasonable funding proposal should have (a clear purpose, a real budget
breakdown, a timeline, etc.)? Committees submit ~16 of these per cycle, often
20+ pages each, and a human reviewer shouldn't have to read all of that just
to notice "there's no budget justification" or "no success metrics."

Advisory-only by design: this module never rewrites, drafts, or auto-fills
proposal content. It only reports which concepts are present/weak/missing,
with a quoted-or-paraphrased evidence pointer so a reviewer can spot-check
the (small, local) model's judgment quickly, and a plain-language suggestion
describing what the human author should go address themselves.
"""
from __future__ import annotations

from pathlib import Path

from . import chunking, ollama_client, pdf_extract

# A larger context window than the ollama_client default (32768), since a
# 20+ page proposal's full extracted text plus the rubric prompt and JSON
# schema can run well past that. See ollama_client.extract_json's num_ctx
# override -- this doesn't touch the global OLLAMA_NUM_CTX default used by
# every other call in the project.
VET_NUM_CTX = 65536

# The completeness rubric. Each entry is (key, label, description) -- key is
# the JSON-schema property name, label is the human-readable name printed by
# the CLI/API, description is what we tell the model to look for. Keep this
# list grant-agnostic: it's about what ANY reasonable funding proposal should
# contain, not what a specific grant program requires.
RUBRIC: list[tuple[str, str, str]] = [
    (
        "purpose_objectives",
        "Purpose / objectives",
        "A clear statement of what the project is and what it aims to achieve -- "
        "not just a title, but concrete objectives.",
    ),
    (
        "target_audience",
        "Target audience / beneficiaries",
        "Who this project is for -- the audience, participants, or community it "
        "benefits, ideally with some indication of scale (how many people).",
    ),
    (
        "budget_breakdown",
        "Budget breakdown with justification",
        "An itemized budget (not just one lump-sum total) that shows what the money "
        "is spent on and why, e.g. a line-item table or category-by-category costs.",
    ),
    (
        "timeline_schedule",
        "Timeline / schedule",
        "A schedule of key dates or phases -- when things happen between now and "
        "project completion.",
    ),
    (
        "outcomes_metrics",
        "Expected outcomes / success metrics",
        "How success will be measured -- concrete expected outcomes, deliverables, "
        "or metrics, not just vague aspirations.",
    ),
    (
        "sustainability_risk",
        "Sustainability / risk considerations",
        "What happens after the funded period, and/or what could go wrong and how "
        "it would be handled -- contingency planning, continuation plans, risks.",
    ),
    (
        "accountability",
        "Organizer / committee accountability",
        "Who is responsible for running and reporting on the project -- named "
        "organizers, roles, or a committee structure, and how they're accountable.",
    ),
]

_STATUS_VALUES = ["present", "weak", "missing"]

# Ranking used to merge per-segment results when a proposal is long enough to be
# chunked: a concept present in any segment is present in the proposal, so the
# strongest status wins. See chunking.merge_objects.
_STATUS_PRIORITY = {"present": 2, "weak": 1, "missing": 0}


def _build_schema() -> dict:
    """Build the JSON schema passed to Ollama's `format` param.

    One fixed, named property per rubric item (rather than a freeform array)
    so the model can't drop, reorder, or invent criteria -- every key in
    RUBRIC is required in the output.
    """
    properties = {}
    for key, label, description in RUBRIC:
        properties[key] = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": _STATUS_VALUES,
                    "description": f"Whether '{label}' is present, weak, or missing in the proposal.",
                },
                "evidence": {
                    "type": "string",
                    "description": (
                        "Must contain a verbatim excerpt copied character-for-character from the "
                        "proposal text above, wrapped in double quotes -- never paraphrase what's "
                        "inside the quotes. You may add a short lead-in or page/section reference "
                        "outside the quotes (e.g. 'p.3, Budget section: \"...\"'), but the quoted "
                        "part itself must be an exact copy so a reviewer can Ctrl+F it in the "
                        "source document. Empty string if status is 'missing'."
                    ),
                },
                "suggestion": {
                    "type": "string",
                    "description": (
                        "A plain-language, advisory suggestion telling the proposal's human "
                        "author what to add or strengthen and why. Do NOT draft or rewrite "
                        "proposal text -- describe what's missing/weak, never write the "
                        "replacement content yourself."
                    ),
                },
            },
            "required": ["status", "evidence", "suggestion"],
        }
    return {"type": "object", "properties": properties, "required": [key for key, _, _ in RUBRIC]}


COMPLETENESS_SCHEMA = _build_schema()

PROMPT_TEMPLATE = """You are a grant-agnostic completeness reviewer for a committee that submits event/funding \
proposals. Your ONLY job is to check whether the proposal below covers a set of concepts that any reasonable \
funding proposal should contain -- you are NOT evaluating it against any specific grant program's rules, and you \
are NOT scoring its quality beyond marking each concept present/weak/missing.

For each of the following {n} criteria, decide if the proposal text covers it:

{criteria_list}

For each criterion, give:
- status: "present" (clearly covered), "weak" (mentioned but thin/vague/incomplete), or "missing" (not addressed at all).
- evidence: MUST include a short excerpt copied verbatim, character-for-character, from the proposal text above, \
wrapped in double quotes -- do not paraphrase or summarize the words inside the quotes, and do not skip words or \
sentences from the middle of the quoted span. You may add a brief lead-in or page/section reference outside the \
quotes, but the quoted part itself must be an exact copy so a reviewer can find it in the source document. Leave \
empty only if status is "missing".
- suggestion: plain-language advice to the proposal's human author about what to add or improve, and why it matters \
for a funding reviewer. Do NOT write or draft replacement proposal text, section content, or example wording for \
them to copy in -- only describe what they should go address themselves.

--- PROPOSAL TEXT ---
{text}
{tables}
--- END PROPOSAL TEXT ---
"""


def _format_criteria_list() -> str:
    return "\n".join(f"{i + 1}. {label}: {desc}" for i, (_, label, desc) in enumerate(RUBRIC))


def _build_prompt(text: str, tables_section: str) -> str:
    return PROMPT_TEMPLATE.format(
        n=len(RUBRIC),
        criteria_list=_format_criteria_list(),
        text=text,
        tables=tables_section,
    )


def _run_completeness(text: str, tables_section: str, model: str | None) -> dict:
    """Run the completeness checklist over the proposal text.

    Short proposals (every one seen so far) go through as a single call, exactly
    as before. A proposal long enough to overflow the context window is split
    into overlapping segments, checked segment by segment against the same fixed
    rubric, and merged back into one result by strongest status per criterion.
    """
    if not chunking.needs_chunking(text, VET_NUM_CTX):
        return ollama_client.extract_json(
            _build_prompt(text, tables_section), COMPLETENESS_SCHEMA, model=model, num_ctx=VET_NUM_CTX
        )

    segments = chunking.split_text(text, chunking.char_budget(VET_NUM_CTX))
    per_segment = [
        ollama_client.extract_json(
            _build_prompt(segment, tables_section), COMPLETENESS_SCHEMA, model=model, num_ctx=VET_NUM_CTX
        )
        for segment in segments
    ]
    return chunking.merge_objects(per_segment, [key for key, _, _ in RUBRIC], _STATUS_PRIORITY)


def _compose_overall_summary(criteria: list[dict]) -> str:
    """Compose the overall summary ourselves from per-criterion statuses,
    rather than asking the model for a free-text summary.

    Rationale: the per-criterion statuses are already schema-constrained and
    the reliable signal; a model-authored overall summary on a small local
    model is one more place for it to hallucinate or (worse) drift into
    writing proposal-ish prose. A deterministic composition from the
    structured statuses is both more trustworthy and fully advisory.
    """
    missing = [c["label"] for c in criteria if c["status"] == "missing"]
    weak = [c["label"] for c in criteria if c["status"] == "weak"]
    present_count = sum(1 for c in criteria if c["status"] == "present")
    total = len(criteria)

    parts = [f"{present_count}/{total} completeness criteria are clearly present."]
    if missing:
        parts.append(f"Missing: {', '.join(missing)}.")
    if weak:
        parts.append(f"Weak/thin: {', '.join(weak)}.")
    if not missing and not weak:
        parts.append("No grant-agnostic gaps found -- still worth a human read-through before submission.")
    return " ".join(parts)


def vet_proposal_completeness(pdf_path: str | Path, model: str | None = None) -> dict:
    """Run the grant-agnostic completeness check over a proposal PDF.

    Extracts the full PDF text (no truncation -- see VET_NUM_CTX), asks the
    local model to judge each rubric criterion via a schema-constrained
    single-shot call, and returns a structured, advisory-only result. A
    proposal too long to fit the context window even at VET_NUM_CTX falls back
    to segment-by-segment checking (see _run_completeness / app/chunking.py):

        {
            "source_pdf": str,
            "page_count": int,
            "criteria": [
                {"name": key, "label": ..., "status": "present"|"weak"|"missing",
                 "evidence": ..., "suggestion": ...},
                ...
            ],
            "overall_summary": str,
            "full_text": str,  # source document text, for clause look-up in a UI
        }

    Never auto-rewrites proposal content -- suggestions are advisory prose
    only, enforced both by the prompt instructions and the schema's
    description fields.
    """
    extracted = pdf_extract.extract_pdf(pdf_path)
    tables_block = pdf_extract.summarize_tables(extracted.tables)
    tables_section = f"\n[Extracted tables]\n{tables_block}\n" if tables_block else ""

    raw = _run_completeness(extracted.text, tables_section, model)

    criteria = []
    for key, label, _description in RUBRIC:
        item = raw.get(key, {}) or {}
        status = item.get("status", "missing")
        if status not in _STATUS_VALUES:
            status = "missing"
        criteria.append({
            "name": key,
            "label": label,
            "status": status,
            "evidence": item.get("evidence", ""),
            "suggestion": item.get("suggestion", ""),
        })

    return {
        "source_pdf": str(pdf_path),
        "page_count": extracted.page_count,
        "criteria": criteria,
        "overall_summary": _compose_overall_summary(criteria),
        # Full extracted text, so a UI can let a reviewer jump from an
        # `evidence` quote to its location in the source document ("click to
        # view clause") without re-extracting or re-uploading the PDF.
        "full_text": extracted.text,
        # Raw pdfplumber tables (list of rows of cells), so a UI can render actual
        # tables in the document view instead of the jumbled inline cell text that
        # full_text necessarily contains (page.extract_text() flattens tables too).
        "tables": extracted.tables,
    }


if __name__ == "__main__":
    import sys

    result = vet_proposal_completeness(sys.argv[1])
    for c in result["criteria"]:
        print(f"[{c['status'].upper():7}] {c['label']}")
        print(f"    evidence: {c['evidence']}")
        print(f"    suggestion: {c['suggestion']}")
    print("\n" + result["overall_summary"])
