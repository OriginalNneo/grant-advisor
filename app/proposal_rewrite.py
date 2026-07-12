"""
On-demand rewrite suggestions for a single flagged excerpt of a proposal.

Deliberately separate from proposal_vet.py/grant_fit.py: those two remain
advisory-only exactly as before (report status, evidence, suggestions --
never generate replacement text), per CLAUDE.md's advisory-only constraint on
their prompts/schemas. This module is a distinct, explicitly user-invoked
capability -- called only when a human clicks "suggest a rewrite" on one
specific flagged criterion in the web UI, never baked into the audit itself.

Every caller must present the result as a draft for the proposal's human
author to adapt, not text this tool is authorized to finalize -- it's a
single-shot small-model generation and can still invent details if not
constrained, hence the prompt's explicit placeholder instruction below.
"""
from __future__ import annotations

from . import ollama_client

# A single excerpt + one requirement -- doesn't need the 65536 window
# proposal_vet.py/grant_fit.py use for full-document extraction.
REWRITE_NUM_CTX = 8192

REWRITE_PROMPT_TEMPLATE = """You are helping an author improve ONE specific flagged section of their \
funding proposal draft. You are NOT auditing the whole proposal -- you're given one excerpt, the specific \
requirement it fails to fully satisfy, and advice on what's missing or weak.

Requirement: {criterion}

What's missing or weak: {suggestion}

--- ORIGINAL EXCERPT ---
{excerpt}
--- END ORIGINAL EXCERPT ---

Rewrite ONLY this excerpt so it satisfies the requirement above. Stay grounded in the facts, numbers, and \
details already present in the original excerpt -- do NOT invent new figures, names, dates, or claims that \
weren't already there or clearly implied. If satisfying the requirement genuinely needs information the \
author hasn't provided (e.g. a specific dollar amount, date, or name), leave a bracketed placeholder like \
[ADD SPECIFIC AMOUNT] instead of inventing one.

Respond with ONLY the rewritten excerpt -- no preamble, no explanation, no surrounding quotation marks.
"""


def suggest_rewrite(excerpt: str, criterion: str, suggestion: str, model: str | None = None) -> str:
    """Generate one rewritten version of a single flagged excerpt, grounded in
    the excerpt's own facts (bracketed placeholders instead of invented
    specifics where information is genuinely missing).

    This is a draft for the proposal's human author to adapt -- not proposal
    text this tool is authorized to finalize. Callers must present it as
    such and never auto-apply it anywhere.
    """
    if not excerpt.strip():
        raise ValueError("Can't suggest a rewrite without source excerpt text.")

    prompt = REWRITE_PROMPT_TEMPLATE.format(
        criterion=criterion or "(unspecified)",
        suggestion=suggestion or "Not specified.",
        excerpt=excerpt,
    )
    return ollama_client.chat_once(prompt, model=model, num_ctx=REWRITE_NUM_CTX)
