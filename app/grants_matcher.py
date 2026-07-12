"""
Local grant shortlisting.

Strategy: embed every grant's description once (cached to disk) using a
local Ollama embedding model, embed the incoming project text the same way,
and rank grants by cosine similarity. This runs entirely on the user's
machine -- no project data ever leaves it.

If Ollama isn't reachable (e.g. running tests without the service up), we
fall back to a crude keyword-overlap score so the rest of the pipeline can
still be exercised.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import ollama
except Exception:  # pragma: no cover - missing package, or a broken/incompatible install
    ollama = None

GRANTS_DB_PATH = Path(__file__).resolve().parent.parent / "grants_db" / "grants.json"
EMBED_CACHE_PATH = Path(__file__).resolve().parent.parent / "grants_db" / ".embed_cache.json"
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


@dataclass
class GrantMatch:
    grant: dict
    score: float


def _load_payload(db_path: Path) -> dict:
    if not Path(db_path).exists():
        return {"_meta": {}, "grants": []}
    return json.loads(Path(db_path).read_text(encoding="utf-8"))


def load_grants(db_path: Path = GRANTS_DB_PATH) -> list[dict]:
    grants = _load_payload(db_path).get("grants", [])
    for g in grants:
        g.setdefault("archived", False)  # entries ingested before this field existed
        # Path (relative to the project root) to the original grant PDF, if
        # still on disk -- empty for CSV-sourced grants or entries ingested
        # before this field existed. Powers the web UI's "view in grant
        # document" panel (app/server.py's /api/grants/{id}/source-file).
        g.setdefault("source_file", "")
    return grants


def add_grant_to_db(grant: dict, db_path: Path = GRANTS_DB_PATH) -> dict:
    """Insert or update a single grant record, preserving the file's _meta block.

    If a grant with the same id already exists it's overwritten in place (an
    update, e.g. re-ingesting a refreshed version of the same program) and its
    stale cached embedding is dropped so it gets re-embedded on next search.
    """
    if not grant.get("name"):
        raise ValueError("Grant record is missing a required 'name' field.")

    payload = _load_payload(db_path)
    grants = payload.setdefault("grants", [])

    existing_idx = next((i for i, g in enumerate(grants) if g["id"] == grant["id"]), None)
    if existing_idx is not None:
        grants[existing_idx] = grant
    else:
        grants.append(grant)

    Path(db_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    cache = _load_embed_cache()
    if grant["id"] in cache:
        del cache[grant["id"]]
        _save_embed_cache(cache)

    return grant


def set_archived(ids: list[str], archived: bool, db_path: Path = GRANTS_DB_PATH) -> list[dict]:
    """Bulk-set the `archived` flag ("outdated" in the UI) for the given grant
    ids. Returns the updated records for whichever ids were actually found.
    """
    payload = _load_payload(db_path)
    grants = payload.setdefault("grants", [])
    id_set = set(ids)
    updated = []
    for g in grants:
        if g["id"] in id_set:
            g["archived"] = archived
            updated.append(g)
    Path(db_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return updated


def delete_grants(ids: list[str], db_path: Path = GRANTS_DB_PATH) -> list[str]:
    """Permanently remove grants with the given ids and evict their cached
    embeddings. Returns the ids actually found and removed. Companion
    markdown/raw-text exports (grants/markdown/<id>.*) are a grant_ingest.py
    concern, not this module's -- callers should also call
    grant_ingest.delete_markdown_exports() for the same ids.
    """
    payload = _load_payload(db_path)
    grants = payload.setdefault("grants", [])
    id_set = set(ids)
    removed = [g["id"] for g in grants if g["id"] in id_set]
    payload["grants"] = [g for g in grants if g["id"] not in id_set]
    Path(db_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if removed:
        cache = _load_embed_cache()
        if any(gid in cache for gid in removed):
            for gid in removed:
                cache.pop(gid, None)
            _save_embed_cache(cache)

    return removed


def _grant_text(grant: dict) -> str:
    parts = [
        grant.get("name", ""),
        grant.get("funder", ""),
        "Focus areas: " + ", ".join(grant.get("focus_areas", [])),
        "Eligibility: " + grant.get("eligibility", ""),
        "Criteria: " + ", ".join(grant.get("criteria", [])),
        "Notes: " + grant.get("notes", ""),
    ]
    return "\n".join(p for p in parts if p)


def _embed(texts: list[str]) -> list[list[float]] | None:
    """Embed a batch of texts via Ollama. Returns None if Ollama is unreachable."""
    if ollama is None:
        return None
    try:
        vectors = []
        for t in texts:
            resp = ollama.embeddings(model=EMBED_MODEL, prompt=t)
            vectors.append(resp["embedding"])
        return vectors
    except Exception:
        return None


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


def _keyword_score(project_text: str, grant: dict) -> float:
    """Crude fallback: fraction of grant focus-area keywords present in the project text."""
    text_lower = project_text.lower()
    words = set(re.findall(r"[a-z0-9]+", text_lower))
    focus_terms = []
    for area in grant.get("focus_areas", []):
        focus_terms.extend(re.findall(r"[a-z0-9]+", area.lower()))
    if not focus_terms:
        return 0.0
    hits = sum(1 for term in focus_terms if term in words)
    return hits / len(focus_terms)


def _load_embed_cache() -> dict:
    if EMBED_CACHE_PATH.exists():
        try:
            return json.loads(EMBED_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_embed_cache(cache: dict) -> None:
    try:
        EMBED_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    except Exception:
        pass


def _grants_embeddings(grants: list[dict]) -> dict[str, list[float]] | None:
    """Get (and cache) embeddings for every grant, keyed by grant id."""
    cache = _load_embed_cache()
    missing = [g for g in grants if g["id"] not in cache]
    if missing:
        vectors = _embed([_grant_text(g) for g in missing])
        if vectors is None:
            return None  # Ollama unreachable -- caller should fall back
        for g, v in zip(missing, vectors):
            cache[g["id"]] = v
        _save_embed_cache(cache)
    return cache


def shortlist_grants(project_text: str, top_k: int = 5, db_path: Path = GRANTS_DB_PATH) -> list[GrantMatch]:
    """Rank grants by relevance to project_text, returning the top_k matches.

    Archived ("outdated") grants are excluded -- they're kept in the database
    for reference but shouldn't be auto-surfaced as a match. An archived
    grant can still be looked up and vetted against explicitly by id
    (grant_fit.py's _select_grant does that directly via load_grants(), not
    through this function).
    """
    grants = [g for g in load_grants(db_path) if not g.get("archived")]
    if not grants:
        return []

    embed_cache = _grants_embeddings(grants)

    if embed_cache is not None:
        query_vec = _embed([project_text])
        if query_vec is not None:
            q = np.array(query_vec[0])
            scored = [
                GrantMatch(grant=g, score=_cosine_sim(q, np.array(embed_cache[g["id"]])))
                for g in grants
            ]
            scored.sort(key=lambda m: m.score, reverse=True)
            return scored[:top_k]

    # Fallback path: no embeddings available, use keyword overlap.
    scored = [GrantMatch(grant=g, score=_keyword_score(project_text, g)) for g in grants]
    scored.sort(key=lambda m: m.score, reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    import sys

    sample_text = " ".join(sys.argv[1:]) or "An AI nonprofit building open source tools for community health."
    for m in shortlist_grants(sample_text, top_k=5):
        print(f"{m.score:.3f}  {m.grant['name']}")
