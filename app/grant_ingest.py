"""
Turn an uploaded grant PDF (a funder's program guidelines/factsheet) into a
structured record for grants_db/grants.json.

This is the inverse of the project-matching pipeline: instead of matching a
project against an existing grants database, this *builds* that database by
having the local LLM read a grant document and extract it into the same
schema csv_to_json.py produces.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from . import grant_markdown, grants_matcher, ollama_client, pdf_extract

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Drop grant PDFs directly into this folder to have them picked up by
# list_pending_pdfs()/the "convert pending grants" flow, instead of uploading
# one at a time through the web UI. Markdown exports (see grant_markdown.py)
# are written to the markdown/ subfolder, one file per grant, named by id.
GRANTS_INBOX_DIR = PROJECT_ROOT / "grants"
GRANTS_MARKDOWN_DIR = GRANTS_INBOX_DIR / "markdown"

# Once a PDF dropped in GRANTS_INBOX_DIR is ingested, it's moved here -- so
# "already converted?" is just "is this PDF still sitting in the inbox root,
# or has it been moved out?" rather than a separate tracking file to keep in
# sync. This also means browsing grants/ in Finder shows at a glance what's
# still pending vs. already handled.
GRANTS_CONVERTED_DIR = GRANTS_INBOX_DIR / "converted"

# Proposals/grant docs can run long; see proposal_vet.py/grant_fit.py's
# num_ctx override for the same reasoning. Page-citation in grant_markdown.py
# needs the *full* document, so (unlike this module's previous behavior)
# extraction is no longer truncated to a fixed character count.
GRANT_INGEST_NUM_CTX = 65536

# Same field set as csv_to_json.py / grants_template.csv, minus id/last_verified
# which we derive ourselves rather than trust the model on.
GRANT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "The grant program's official name."},
        "funder": {"type": "string", "description": "The organization offering the grant."},
        "focus_areas": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short topical tags describing what this grant funds, e.g. 'community health', 'digital inclusion'.",
        },
        "eligibility": {"type": "string", "description": "Who can apply -- entity type, location, sector restrictions, etc."},
        "criteria": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Discrete, individually checkable funding requirements a proposal must satisfy, e.g. "
                "'501(c)(3) status required', 'budget capped at $50,000', 'must include a sustainability "
                "plan'. Each entry should be a single, specific, verifiable requirement -- NOT a vague "
                "restatement of the eligibility blurb above. If the document doesn't spell out discrete "
                "requirements beyond general eligibility, leave this empty rather than inventing items."
            ),
        },
        "typical_amount": {"type": "string", "description": "Funding amount or range, as stated in the document, e.g. '$5,000 - $50,000'."},
        "geography": {"type": "string", "description": "Geographic eligibility, e.g. 'United States', 'Singapore', 'Global'."},
        "link": {"type": "string", "description": "URL to the grant's page, if present in the document, else empty string."},
        "notes": {"type": "string", "description": "Deadlines, application process notes, or other caveats worth flagging."},
    },
    "required": ["name", "funder", "focus_areas", "eligibility", "criteria", "typical_amount", "geography", "link", "notes"],
}

EXTRACTION_PROMPT_TEMPLATE = """You are extracting structured data from a grant program's own PDF documentation \
(guidelines, factsheet, or call for proposals). Read the text below and fill in the fields as accurately as \
possible based ONLY on what's stated in the document. If a field isn't mentioned, use an empty string (or empty \
list for focus_areas) rather than guessing.

focus_areas should be a list of short topical tags (2-6 words each) capturing what kinds of projects this grant \
funds -- these are used later to semantically match applicants' projects, so be specific and comprehensive rather \
than vague.

criteria should be a list of discrete, individually checkable funding requirements -- the kind of thing a reviewer \
could tick off one at a time against a specific proposal, e.g. "501(c)(3) status required", "budget capped at \
$50,000", "must include a sustainability plan", "match funding of at least 20% required". Do NOT just restate the \
eligibility field in list form -- only include items specific and concrete enough to check point-by-point. If the \
document doesn't state any requirements this specific, leave the list empty.

--- DOCUMENT TEXT ---
{text}
{tables}
--- END DOCUMENT ---
"""


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "grant"


def extract_grant_from_pdf(
    pdf_path: str | Path,
    model: str | None = None,
    extracted: pdf_extract.ExtractedPDF | None = None,
) -> dict:
    """Extract a grant PDF into a grants_db-schema record. Does not write to disk.

    Pass `extracted` if the caller already ran pdf_extract.extract_pdf() on
    this file (e.g. ingest_grant_pdf(), which also needs .pages for markdown
    citation) to avoid re-parsing the PDF a second time.
    """
    if extracted is None:
        extracted = pdf_extract.extract_pdf(pdf_path)
    tables_block = pdf_extract.summarize_tables(extracted.tables)
    tables_section = f"\n[Extracted tables]\n{tables_block}\n" if tables_block else ""

    prompt = EXTRACTION_PROMPT_TEMPLATE.format(text=extracted.text, tables=tables_section)
    fields = ollama_client.extract_json(prompt, GRANT_SCHEMA, model=model, num_ctx=GRANT_INGEST_NUM_CTX)

    if not fields.get("name"):
        raise ValueError(
            f"Couldn't extract a grant name from {Path(pdf_path).name} -- this may not be a grant "
            "program document, or the model failed to parse it."
        )

    # Deliberately NOT collision-suffixed: the id is a pure function of
    # funder+name, so re-ingesting the same grant (from any source -- web
    # upload, CLI, the grants/ inbox folder) always resolves to the same id
    # and correctly updates that one record via add_grant_to_db()'s
    # upsert-by-id behavior, instead of piling up "-2"/"-3" duplicates of the
    # same grant. Two genuinely different grants would only collide here if
    # they share both funder and name exactly, which is already how the rest
    # of the system treats "the same grant."
    grant_id = _slugify(f"{fields.get('funder', '')}-{fields['name']}")

    today = date.today().isoformat()
    auto_note = f"Auto-extracted from '{Path(pdf_path).name}' on {today}; verify amounts/deadlines/eligibility on the funder's own site."
    notes = fields.get("notes", "")
    fields["notes"] = f"{notes}\n\n{auto_note}".strip()

    return {
        "id": grant_id,
        "name": fields["name"],
        "funder": fields.get("funder", ""),
        "focus_areas": fields.get("focus_areas", []),
        "eligibility": fields.get("eligibility", ""),
        "criteria": fields.get("criteria", []),
        "typical_amount": fields.get("typical_amount", ""),
        "geography": fields.get("geography", ""),
        "link": fields.get("link", ""),
        "notes": fields["notes"],
        "last_verified": today,
        # Re-ingesting a grant (e.g. the funder published a refreshed version)
        # always resets archived to False -- if you're re-extracting it,
        # you're saying it's current again.
        "archived": False,
    }


def list_pending_pdfs() -> list[str]:
    """Filenames of PDFs sitting directly in GRANTS_INBOX_DIR (i.e. not yet
    moved into GRANTS_CONVERTED_DIR by a prior ingest_grant_pdf() call).
    Path.glob("*.pdf") only matches direct children, so this naturally
    excludes anything already inside markdown/ or converted/. Used to power
    the "we found N grant PDFs, convert now?" prompt shown when the web UI
    loads.
    """
    if not GRANTS_INBOX_DIR.exists():
        return []
    return sorted(p.name for p in GRANTS_INBOX_DIR.glob("*.pdf"))


def _unique_converted_destination(filename: str) -> Path:
    """Avoid clobbering a previously-converted file if two different PDFs
    happen to share a filename."""
    dest = GRANTS_CONVERTED_DIR / filename
    if not dest.exists():
        return dest
    stem, suffix = Path(filename).stem, Path(filename).suffix
    n = 2
    while (GRANTS_CONVERTED_DIR / f"{stem}-{n}{suffix}").exists():
        n += 1
    return GRANTS_CONVERTED_DIR / f"{stem}-{n}{suffix}"


def raw_text_path(grant_id: str) -> Path:
    """Path to a grant's full extracted document text, saved at ingestion
    time so app/grant_qa.py has something to ground Q&A answers in without
    needing the original PDF to still exist on disk.
    """
    return GRANTS_MARKDOWN_DIR / f"{grant_id}.raw.txt"


def pages_path(grant_id: str) -> Path:
    """Path to a grant's per-page extracted text (JSON list, 1:1 with the
    source PDF's physical pages), saved at ingestion time. Unlike
    raw_text_path()'s single joined string, this preserves page boundaries so
    a criterion/requirement string can later be located on a specific page --
    see grant_markdown.locate_source() -- for the "view in grant document"
    panel in the web UI, without needing the original PDF re-extracted.
    """
    return GRANTS_MARKDOWN_DIR / f"{grant_id}.pages.json"


def delete_markdown_exports(grant_ids: list[str]) -> None:
    """Remove the markdown + raw-text + per-page companion files for the
    given grant ids (called alongside grants_matcher.delete_grants() when
    permanently deleting a grant -- these are orphaned files otherwise, since
    nothing else references them once the grant record is gone).
    """
    for gid in grant_ids:
        (GRANTS_MARKDOWN_DIR / f"{gid}.md").unlink(missing_ok=True)
        raw_text_path(gid).unlink(missing_ok=True)
        pages_path(gid).unlink(missing_ok=True)


def ingest_grant_pdf(pdf_path: str | Path, model: str | None = None) -> dict:
    """Extract a grant PDF, add/update it in grants_db/grants.json, and write
    three companion files to GRANTS_MARKDOWN_DIR: a page-cited markdown
    export for human review, the full raw extracted text (`<id>.raw.txt`)
    that app/grant_qa.py grounds question-answering in, and the per-page text
    (`<id>.pages.json`) that the web UI's "view in grant document" panel uses
    to locate + highlight a specific criterion on its source page.

    If `pdf_path` lives directly in GRANTS_INBOX_DIR (i.e. it was dropped
    into grants/ rather than uploaded through the web UI), moves it into
    GRANTS_CONVERTED_DIR afterward so list_pending_pdfs() won't offer to
    convert it again -- and so browsing grants/ shows at a glance which PDFs
    are still pending vs. already handled. Either way, the grant record's
    `source_file` field is set to the PDF's final on-disk location (relative
    to the project root) so it can be served back for that same panel.
    """
    pdf_path = Path(pdf_path)
    extracted = pdf_extract.extract_pdf(pdf_path)
    grant = extract_grant_from_pdf(pdf_path, model=model, extracted=extracted)

    final_pdf_path = pdf_path
    if pdf_path.resolve().parent == GRANTS_INBOX_DIR.resolve():
        GRANTS_CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
        final_pdf_path = _unique_converted_destination(pdf_path.name)
        pdf_path.rename(final_pdf_path)

    try:
        grant["source_file"] = str(final_pdf_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        # Shouldn't normally happen (uploads/inbox are both under the project
        # root), but don't fail ingestion over it -- just skip the reference.
        grant["source_file"] = ""

    grant = grants_matcher.add_grant_to_db(grant)

    markdown = grant_markdown.render_grant_markdown(grant, extracted.pages)
    GRANTS_MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    (GRANTS_MARKDOWN_DIR / f"{grant['id']}.md").write_text(markdown)
    raw_text_path(grant["id"]).write_text(extracted.text)
    pages_path(grant["id"]).write_text(json.dumps(extracted.pages))

    return grant


if __name__ == "__main__":
    import sys

    result = ingest_grant_pdf(sys.argv[1])
    print(f"Added grant '{result['id']}': {result['name']}")
