"""
Render a grants_db-schema grant record as a human-readable markdown file,
with each fact labeled with the source PDF page it was found on where a
match can be located.

This is a companion export, not a data store: grants_db/grants.json remains
the operational database that grants_matcher.py searches/embeds against.
Markdown here is purely for a human to skim/spot-check an ingested grant
without opening the original PDF.

Page citation is a best-effort substring lookup against the PDF's per-page
text (see pdf_extract.ExtractedPDF.pages) -- exactly the same "find where
this text came from" problem the frontend's clause-viewer solves in
static/index.html, just run in Python against ingestion-time data instead of
in the browser against vetting evidence. Some fields (e.g. model-authored
`focus_areas` tags) are paraphrases that won't appear verbatim anywhere in
the source text; those are rendered without a page label rather than
guessing or fabricating one.
"""
from __future__ import annotations

import re


def _token_pattern(token: str) -> str:
    """A trailing colon/comma/period/semicolon is often a seam the model
    added or dropped when extracting a field, not a real content difference
    -- make it optional rather than requiring or stripping it. Mirrors
    static/index.html's tokenPattern() (same problem, same fix, ported)."""
    m = re.match(r"^(.*?)([:;,.]+)$", token)
    if m and m.group(1):
        return re.escape(m.group(1)) + r"(?:" + re.escape(m.group(2)) + r")?"
    return re.escape(token)


def _find_in_page(page_text: str, snippet: str) -> tuple[int, int] | None:
    """Whitespace-tolerant search for the longest verbatim run of `snippet`'s
    words -- anywhere within snippet, not just a leading prefix -- inside
    `page_text`. Mirrors static/index.html's findInText() -- same rationale:
    a single-shot LLM extraction sometimes reformats whitespace or drops a
    word even when asked to preserve the document's own wording. Criteria in
    particular are often model-synthesized sentences that wrap a real
    verbatim fragment (e.g. an amount) in explanatory prose the source
    document never states in that form -- e.g. "Maximum subsidy capped at
    $25 per participant for general projects" only has "$25 per participant"
    verbatim in the PDF, in the middle of the sentence, not at the start. A
    prefix-only search anchored at snippet's first word would never find
    that; trying every starting offset (not just 0) does.
    """
    tokens = snippet.split()
    if not tokens:
        return None
    min_words = min(3, len(tokens))
    for n in range(len(tokens), min_words - 1, -1):
        for start in range(0, len(tokens) - n + 1):
            pattern = r"\s+".join(_token_pattern(t) for t in tokens[start:start + n])
            try:
                m = re.search(pattern, page_text, re.IGNORECASE)
            except re.error:
                continue
            if m:
                return (m.start(), m.end() - m.start())
    return None


def locate_source(pages: list[str], needle: str) -> dict | None:
    """Find `needle` (e.g. a criterion/requirement string) within a grant's
    per-page source text. Returns the first page it's found on as
    {"page": 1-based int, "index": int, "length": int, "page_text": str}, or
    None if it can't be located on any page closely enough. Powers both
    _find_page()/_cite() below (ingestion-time markdown citation) and the web
    UI's live "view in grant document" panel (app/server.py's
    /api/grants/{id}/locate), which additionally needs the match's exact
    offset within the page (not just which page) to render a highlight.
    """
    needle = needle.strip()
    if not needle:
        return None
    for i, page_text in enumerate(pages):
        match = _find_in_page(page_text, needle)
        if match:
            return {"page": i + 1, "index": match[0], "length": match[1], "page_text": page_text}
    return None


def _find_page(pages: list[str], needle: str) -> int | None:
    """Return the 1-based page number of the first page containing `needle`
    (whitespace/punctuation-tolerant), or None if no page matches closely
    enough. See locate_source() above -- this is a thin wrapper over it for
    callers that only need the page number, not the match offset.
    """
    result = locate_source(pages, needle)
    return result["page"] if result else None


def _cite(pages: list[str], value: str) -> str:
    """Return `value`, suffixed with '(p.N)' if it can be located in `pages`."""
    if not value:
        return value
    page = _find_page(pages, value)
    return f"{value} (p.{page})" if page else value


def render_grant_markdown(grant: dict, pages: list[str]) -> str:
    """Render one grant record as markdown, citing source pages where found.

    `grant` is a grants_db-schema record (see CLAUDE.md's Grant schema
    section); `pages` is ExtractedPDF.pages from the same PDF the record was
    extracted from.
    """
    lines = [f"# {grant.get('name', '(untitled grant)')}", ""]

    lines.append(f"**Funder:** {_cite(pages, grant.get('funder', ''))}")
    if grant.get("typical_amount"):
        lines.append(f"**Typical amount:** {_cite(pages, grant['typical_amount'])}")
    if grant.get("geography"):
        lines.append(f"**Geography:** {_cite(pages, grant['geography'])}")
    if grant.get("link"):
        lines.append(f"**Link:** {grant['link']}")
    lines.append("")

    lines.append("## Focus areas")
    focus_areas = grant.get("focus_areas") or []
    if focus_areas:
        for fa in focus_areas:
            lines.append(f"- {_cite(pages, fa)}")
    else:
        lines.append("_None extracted._")
    lines.append("")

    lines.append("## Eligibility")
    lines.append(_cite(pages, grant.get("eligibility", "")) or "_None extracted._")
    lines.append("")

    lines.append("## Criteria")
    criteria = grant.get("criteria") or []
    if criteria:
        for c in criteria:
            lines.append(f"- {_cite(pages, c)}")
    else:
        lines.append("_No discrete criteria extracted; see Eligibility above._")
    lines.append("")

    if grant.get("notes"):
        lines.append("## Notes")
        lines.append(grant["notes"])
        lines.append("")

    lines.append(f"_Grant ID: `{grant.get('id', '')}` · Last verified: {grant.get('last_verified', '')}_")
    return "\n".join(lines) + "\n"
