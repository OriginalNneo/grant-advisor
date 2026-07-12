"""
Grant-category correlation.

shortlist_grants() (grants_matcher.py) finds specific matching grants. This
answers a related but different question for the UI: which of the *focus
area* categories already present across the local grants database does a
piece of proposal/project text most strongly correlate with (e.g. "STEM",
"Sustainability")? Surfaced as a top-N panel in the Find a Grant and
vet-proposal views.

Same embed-then-cosine-similarity strategy as grants_matcher.shortlist_grants,
reusing its low-level _embed/_cosine_sim helpers (both stateless and generic
over arbitrary text) rather than duplicating the Ollama-calling code. Falls
back to keyword overlap if Ollama is unreachable, same as grants_matcher.

Categories aren't grants, so they get their own embedding cache file rather
than overloading grants_matcher's .embed_cache.json, which is keyed by grant id.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from . import grants_matcher

CATEGORY_EMBED_CACHE_PATH = grants_matcher.GRANTS_DB_PATH.parent / ".category_embed_cache.json"


def _normalize_category(raw: str) -> str:
    """Casefold + collapse whitespace so "STEM" and "stem " dedupe to one bucket."""
    return re.sub(r"\s+", " ", raw.strip()).casefold()


def _distinct_categories(grants: list[dict]) -> dict[str, str]:
    """normalized category -> display label (first-seen original casing). Archived
    grants are excluded, matching shortlist_grants' auto-matching exclusion."""
    labels: dict[str, str] = {}
    for g in grants:
        if g.get("archived"):
            continue
        for area in g.get("focus_areas") or []:
            if not area or not area.strip():
                continue
            key = _normalize_category(area)
            labels.setdefault(key, area.strip())
    return labels


def _load_cache() -> dict[str, list[float]]:
    if CATEGORY_EMBED_CACHE_PATH.exists():
        try:
            return json.loads(CATEGORY_EMBED_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict[str, list[float]]) -> None:
    try:
        CATEGORY_EMBED_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    except Exception:
        pass


def _keyword_score(text: str, category_label: str) -> float:
    """Crude fallback: fraction of the category label's keywords present in text."""
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    label_terms = re.findall(r"[a-z0-9]+", category_label.lower())
    if not label_terms:
        return 0.0
    hits = sum(1 for term in label_terms if term in tokens)
    return hits / len(label_terms)


def _rank_labels(text: str, labels: dict[str, str], top_k: int) -> list[dict]:
    """Shared ranking core for score_categories/score_grant_categories: embed
    (or keyword-score) each label against text, sorted descending."""
    if not labels:
        return []

    cache = _load_cache()
    missing = [key for key in labels if key not in cache]
    embeddings_ok = True
    if missing:
        vectors = grants_matcher._embed([labels[key] for key in missing])
        if vectors is None:
            embeddings_ok = False
        else:
            for key, vec in zip(missing, vectors):
                cache[key] = vec
            _save_cache(cache)

    query_vec = grants_matcher._embed([text]) if embeddings_ok else None

    if query_vec is not None:
        q = np.array(query_vec[0])
        results = [
            {"category": label, "score": grants_matcher._cosine_sim(q, np.array(cache[key]))}
            for key, label in labels.items()
        ]
    else:
        results = [{"category": label, "score": _keyword_score(text, label)} for label in labels.values()]

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def score_categories(text: str, top_k: int = 3, db_path: Path = grants_matcher.GRANTS_DB_PATH) -> list[dict]:
    """Rank the grants database's distinct focus_areas categories by how
    strongly `text` correlates with each. Returns up to top_k
    {"category": str, "score": float} dicts, sorted descending.

    Returns [] if the database has no focus_areas tags at all, or text is empty.
    """
    if not text.strip():
        return []
    labels = _distinct_categories(grants_matcher.load_grants(db_path))
    return _rank_labels(text, labels, top_k)


def score_grant_categories(text: str, focus_areas: list[str], top_k: int = 3) -> list[dict]:
    """Rank a *single* grant's own focus_areas by how strongly `text`
    correlates with each -- unlike score_categories above, which ranks across
    every category in the whole database. Used by the Find-a-Grant "view
    document" flow to show which of THIS grant's categories the user's
    project best fits (i.e. the category they'd actually apply under), rather
    than whichever category happens to be the first one locatable in the
    source PDF. Shares the same category embedding cache (keyed by normalized
    label) since a category's embedding doesn't depend on which grant it
    belongs to.

    Returns [] if focus_areas is empty or text is blank.
    """
    if not text.strip():
        return []
    labels: dict[str, str] = {}
    for area in focus_areas or []:
        if area and area.strip():
            labels.setdefault(_normalize_category(area), area.strip())
    return _rank_labels(text, labels, top_k)


if __name__ == "__main__":
    import sys

    sample_text = " ".join(sys.argv[1:]) or "An after-school coding and robotics program for middle schoolers."
    for r in score_categories(sample_text):
        print(f"{r['score']:.3f}  {r['category']}")
