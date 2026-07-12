"""
Answer a free-text question about ONE specific grant, grounded in that
grant's full extracted document text (see grant_ingest.raw_text_path) rather
than the model's general knowledge or the (schema-shaped, so lossier)
markdown export.

This is a read-only, informational pipeline -- distinct from grant matching
(agent.py, matches a project against many grants) and proposal vetting
(proposal_vet.py/grant_fit.py, checks a proposal against a rubric/criteria).
It never touches a proposal; it only answers "what does this grant say about
X?" against the grant's own source document.
"""
from __future__ import annotations

from . import grant_ingest, grants_matcher, ollama_client

# Grant program PDFs can run long; see proposal_vet.py/grant_fit.py's num_ctx
# override for the same reasoning.
QA_NUM_CTX = 65536

QA_PROMPT_TEMPLATE = """You are answering a question about ONE specific grant program, using ONLY the \
document text below -- not outside knowledge about this or any other grant program. Answer directly and \
concisely in plain prose (2-5 sentences), quoting or paraphrasing the relevant part of the document. If the \
document doesn't state an answer, say so plainly rather than guessing.

Grant: {name} (funder: {funder})

Question: {question}

--- GRANT DOCUMENT TEXT ---
{text}
--- END GRANT DOCUMENT TEXT ---
"""


def answer_grant_question(grant_id: str, question: str, model: str | None = None) -> str:
    """Answer `question` about the grant with id `grant_id`, grounded in its
    stored raw extracted text. Raises ValueError if the grant, or its stored
    document text, can't be found.
    """
    grants = grants_matcher.load_grants()
    grant = next((g for g in grants if g.get("id") == grant_id), None)
    if grant is None:
        raise ValueError(f"No grant with id '{grant_id}' found in the grants database.")

    raw_path = grant_ingest.raw_text_path(grant_id)
    if not raw_path.exists():
        raise ValueError(
            f"No stored document text for grant '{grant_id}' -- it may have been ingested before "
            "Q&A support existed. Re-ingest the grant's PDF to enable this."
        )

    prompt = QA_PROMPT_TEMPLATE.format(
        name=grant.get("name", ""),
        funder=grant.get("funder", ""),
        question=question,
        text=raw_path.read_text(),
    )
    return ollama_client.chat_once(prompt, model=model, num_ctx=QA_NUM_CTX)
