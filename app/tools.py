"""
Tools the local LLM is allowed to call.

Everything here is plain Python the model invokes through Ollama's
tool-calling interface (see ollama_client.py). File access is sandboxed to
the project's data/ directory on purpose -- the model can read/write its
own working files and reports, but can't wander the rest of your disk.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import grants_matcher

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = (PROJECT_ROOT / "data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "reports").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "uploads").mkdir(parents=True, exist_ok=True)

MAX_READ_CHARS = 20_000


class SandboxViolation(Exception):
    pass


def _resolve_in_sandbox(relative_path: str) -> Path:
    """Resolve a user/model-supplied relative path inside DATA_DIR, refusing escapes."""
    candidate = (DATA_DIR / relative_path).resolve()
    if DATA_DIR not in candidate.parents and candidate != DATA_DIR:
        raise SandboxViolation(
            f"Refusing to access '{relative_path}': outside the sandboxed data/ directory."
        )
    return candidate


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def list_files(subdir: str = "") -> str:
    """List files under data/ (optionally inside a subdirectory)."""
    try:
        target = _resolve_in_sandbox(subdir)
    except SandboxViolation as e:
        return json.dumps({"error": str(e)})
    if not target.exists():
        return json.dumps({"error": f"No such directory: {subdir}"})
    entries = [
        str(p.relative_to(DATA_DIR)) for p in sorted(target.rglob("*")) if p.is_file()
    ]
    return json.dumps({"files": entries})


def read_file(path: str) -> str:
    """Read a text file from inside data/. Truncates very large files."""
    try:
        target = _resolve_in_sandbox(path)
    except SandboxViolation as e:
        return json.dumps({"error": str(e)})
    if not target.exists() or not target.is_file():
        return json.dumps({"error": f"No such file: {path}"})
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return json.dumps({"error": f"Could not read {path}: {e}"})
    truncated = len(content) > MAX_READ_CHARS
    return json.dumps({
        "path": path,
        "truncated": truncated,
        "content": content[:MAX_READ_CHARS],
    })


def write_file(path: str, content: str) -> str:
    """Write a text file inside data/, creating parent directories as needed."""
    try:
        target = _resolve_in_sandbox(path)
    except SandboxViolation as e:
        return json.dumps({"error": str(e)})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return json.dumps({"status": "written", "path": path, "bytes": len(content.encode("utf-8"))})


def save_report(filename: str, content: str) -> str:
    """Save a recommendation report into data/reports/."""
    if not filename.endswith(".md"):
        filename = filename + ".md"
    return write_file(f"reports/{filename}", content)


def search_grants(query: str, top_k: int = 5) -> str:
    """Semantically search the local grants database for matches to a query."""
    matches = grants_matcher.shortlist_grants(query, top_k=top_k)
    return json.dumps({
        "matches": [
            {
                "id": m.grant["id"],
                "name": m.grant["name"],
                "funder": m.grant.get("funder", ""),
                "focus_areas": m.grant.get("focus_areas", []),
                "score": round(m.score, 4),
            }
            for m in matches
        ]
    })


def get_grant_details(grant_id: str) -> str:
    """Get the full record for a single grant by id."""
    grants = grants_matcher.load_grants()
    for g in grants:
        if g["id"] == grant_id:
            return json.dumps(g)
    return json.dumps({"error": f"No grant with id '{grant_id}'"})


DISPATCH = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "save_report": save_report,
    "search_grants": search_grants,
    "get_grant_details": get_grant_details,
}

# Ollama/OpenAI-style tool schemas passed to ollama.chat(tools=TOOLS)
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the sandboxed data/ working directory, optionally inside a subdirectory like 'reports' or 'uploads'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subdir": {"type": "string", "description": "Subdirectory under data/ to list. Empty string lists everything."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file's contents from the sandboxed data/ working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to data/, e.g. 'uploads/project.txt'."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file inside the sandboxed data/ working directory, creating folders as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to data/, e.g. 'notes/scratch.txt'."},
                    "content": {"type": "string", "description": "Text content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_report",
            "description": "Save a markdown grant-recommendation report into data/reports/.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Report filename, e.g. 'acme-nonprofit-grants.md'."},
                    "content": {"type": "string", "description": "Full markdown content of the report."},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_grants",
            "description": "Semantically search the local grants database for programs relevant to a topic, focus area, or project description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for, e.g. 'AI nonprofit serving rural communities'."},
                    "top_k": {"type": "integer", "description": "How many results to return (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_grant_details",
            "description": "Get full details for one grant by its id (ids come back from search_grants).",
            "parameters": {
                "type": "object",
                "properties": {
                    "grant_id": {"type": "string", "description": "The grant's id, e.g. 'nsf-sbir-sttr'."},
                },
                "required": ["grant_id"],
            },
        },
    },
]


def call_tool(name: str, arguments: dict) -> str:
    """Dispatch a model-requested tool call by name. Always returns a string."""
    fn = DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool '{name}'"})
    try:
        return fn(**arguments)
    except Exception as e:
        return json.dumps({"error": f"Tool '{name}' failed: {e}"})
