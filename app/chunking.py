"""
Fallback chunking for documents that overflow the model's usable context window.

The vetting pipelines (proposal_vet, grant_fit) feed a proposal's full extracted
text to the local model in one schema-constrained call, after raising num_ctx to
65536. That fits every proposal handled so far (6 to 24 pages). This module is
the fallback for the case that was previously untested: a document long enough
that its text would not fit the context window even at the raised num_ctx.

When that happens, split_text() breaks the text into overlapping segments that
each fit the budget. The caller runs the SAME fixed checklist on each segment,
and merge_objects() combines the per-segment results back into one object with
the original schema's shape, so nothing downstream has to change.

Merge rule: a checklist item counts as satisfied if ANY segment satisfies it, so
the strongest status seen across segments wins. A criterion covered in section 5
is covered in the proposal even if section 1 never mentioned it. Ties prefer the
segment that carried a real, non-empty evidence excerpt.
"""
from __future__ import annotations

# Rough characters-per-token for English prose under the gemma tokenizer. Kept
# conservative (the real ratio is around 4) so the estimate reserves headroom
# rather than under-counting and overflowing.
_CHARS_PER_TOKEN = 4

# Share of the context window left for the prompt scaffold, the JSON schema, and
# the model's own output. The document text gets the rest.
_TEXT_BUDGET_FRACTION = 0.65

# Overlap between neighbouring segments, so a concept that straddles a boundary
# still appears whole in at least one segment.
_DEFAULT_OVERLAP_CHARS = 800


def char_budget(num_ctx: int) -> int:
    """Maximum document-text characters that comfortably fit one call at num_ctx."""
    return int(num_ctx * _TEXT_BUDGET_FRACTION * _CHARS_PER_TOKEN)


def needs_chunking(text: str, num_ctx: int, max_chars: int | None = None) -> bool:
    """True when `text` is too long to send in a single call at num_ctx.

    `max_chars` overrides the computed budget, mainly so tests can force the
    chunked path on a short string without needing a genuinely huge document.
    """
    budget = max_chars if max_chars is not None else char_budget(num_ctx)
    return len(text or "") > budget


def split_text(text: str, max_chars: int, overlap: int = _DEFAULT_OVERLAP_CHARS) -> list[str]:
    """Split `text` into overlapping segments of at most `max_chars` each.

    Prefers to break on blank-line paragraph boundaries; hard-splits any single
    paragraph longer than the per-segment size. Every segment after the first
    repeats the last `overlap` characters of the previous one so a concept that
    lands on a boundary stays intact in at least one segment.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    text = text or ""
    if len(text) <= max_chars:
        return [text] if text else []

    overlap = max(0, min(overlap, max_chars // 4))
    base = max(1, max_chars - overlap)  # leave room to prepend the overlap tail

    base_segments: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        piece = para if not current else current + "\n\n" + para
        if len(piece) <= base:
            current = piece
            continue
        if current:
            base_segments.append(current)
            current = ""
        if len(para) <= base:
            current = para
        else:
            for i in range(0, len(para), base):
                base_segments.append(para[i : i + base])
    if current:
        base_segments.append(current)

    if overlap and len(base_segments) > 1:
        overlapped = [base_segments[0]]
        for prev, seg in zip(base_segments, base_segments[1:]):
            overlapped.append(prev[-overlap:] + seg)
        return overlapped
    return base_segments


def merge_objects(objs: list[dict], keys: list[str], status_priority: dict[str, int]) -> dict:
    """Merge per-segment result objects into one, keeping the strongest status
    per key.

    Each object is `{key: {"status": ..., "evidence": ..., "suggestion": ...}}`.
    For each key, the entry with the highest-ranked status wins; a tie is broken
    in favour of the entry that carries a non-empty evidence excerpt. Statuses
    not present in `status_priority` rank below every known status, so a stray
    or empty entry never beats a real one. Returns an object in the same shape a
    single call would produce, so callers stay unchanged.
    """
    merged: dict = {}
    for key in keys:
        best_item: dict = {}
        best_score: tuple[int, int] | None = None
        for obj in objs:
            item = (obj or {}).get(key) or {}
            rank = status_priority.get(item.get("status"), -1)
            has_evidence = 1 if (item.get("evidence") or "").strip() else 0
            score = (rank, has_evidence)
            if best_score is None or score > best_score:
                best_score = score
                best_item = item
        merged[key] = best_item
    return merged
