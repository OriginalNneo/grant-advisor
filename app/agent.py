"""
Top-level orchestration: PDF in -> shortlisted + LLM-reasoned grant
recommendations out. This module is the thing both the FastAPI server and
the CLI call into, so the "interface" is just a thin shell around this.
"""
from __future__ import annotations

from pathlib import Path

from . import grants_matcher, ollama_client, pdf_extract

SYSTEM_PROMPT = """You are a private, locally-run grant advisor. You run entirely on the \
user's own machine via Ollama, so you can be candid and specific without worrying about \
data leaving the device.

You have tools:
- search_grants(query, top_k): semantic search over the local grants database
- get_grant_details(grant_id): full record for one grant
- read_file / write_file / list_files: sandboxed access to the data/ working directory
- save_report(filename, content): save a markdown recommendation report to data/reports/

When recommending grants:
1. Use search_grants / get_grant_details to verify details rather than guessing.
2. Pick the 3-5 best-fitting grants, not every possible match.
3. For each, explain in 1-3 sentences WHY it fits this specific project, and flag any \
eligibility concerns or missing information.
4. Always note that amounts/deadlines/eligibility should be verified on the funder's own \
site since grant programs change between cycles.
5. When asked for a final recommendation, call save_report to write a markdown report, \
then give a concise summary in your reply (don't just dump the whole report back as text).
"""


def _shortlist_block(shortlist) -> str:
    if not shortlist:
        return "(no local matches found)"
    return "\n".join(
        f"- {m.grant['id']}: {m.grant['name']} (funder: {m.grant.get('funder', '')}, "
        f"relevance {m.score:.2f})"
        for m in shortlist
    )


def analyze_text(text: str, top_k_shortlist: int = 6, model: str | None = None) -> dict:
    """Run the full pipeline on a plain-text project/event description.

    This is the core both analyze_pdf and the text-only "what grant can we
    apply for" Q&A path (no PDF, just a description typed during workplan
    discussion) share.
    """
    shortlist = grants_matcher.shortlist_grants(text, top_k=top_k_shortlist)

    user_prompt = f"""Here is a description of a project/event/organization:

---
{text[:8000]}
---

A local semantic search over the grants database already shortlisted these candidates:
{_shortlist_block(shortlist)}

Decide which of these (or others you find via search_grants) are genuinely the best fit. \
Use get_grant_details to check eligibility details before recommending. Then save a report \
with save_report and summarize it for me."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    final_text, conversation = ollama_client.run_agent_loop(messages, model=model)

    return {
        "summary": final_text,
        "shortlist": [
            {"id": m.grant["id"], "name": m.grant["name"], "score": round(m.score, 4)}
            for m in shortlist
        ],
        "conversation": conversation,
        "extracted_chars": len(text),
    }


def analyze_pdf(pdf_path: str | Path, top_k_shortlist: int = 6, model: str | None = None) -> dict:
    """Run the full pipeline on an uploaded PDF and return a recommendation."""
    extracted = pdf_extract.extract_pdf(pdf_path)
    result = analyze_text(extracted.text, top_k_shortlist=top_k_shortlist, model=model)
    result["source_pdf"] = str(pdf_path)
    return result


def continue_chat(conversation: list[dict], user_message: str, model: str | None = None) -> dict:
    """Continue an existing conversation (e.g. a follow-up question) with full tool access."""
    updated = list(conversation) + [{"role": "user", "content": user_message}]
    final_text, full_conversation = ollama_client.run_agent_loop(updated, model=model)
    return {"summary": final_text, "conversation": full_conversation}


if __name__ == "__main__":
    import sys

    result = analyze_pdf(sys.argv[1])
    print(result["summary"])
