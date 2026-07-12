"""
Local-only FastAPI server. Run with:

    uvicorn app.server:app --reload --port 8000

Then open http://localhost:8000 in a browser for the bundled minimal UI, or
hit the JSON endpoints directly from your own application:

    POST /api/analyze       (multipart form, field "file" = the PDF)     -> recommendation + conversation_id
    POST /api/analyze-text  {"text": "..."}                              -> recommendation + conversation_id (no PDF)
    POST /api/chat          {"conversation_id": "...", "message": "..."} -> follow-up answer
    POST /api/grants/upload (multipart form, field "file" = a grant PDF) -> extracted grant record, added to the DB
    GET  /api/grants/pending                                              -> grant PDFs in grants/ not yet converted
    POST /api/grants/convert-pending                                      -> convert all pending grants/ PDFs
    GET  /api/grants/{grant_id}/markdown                                  -> a grant's page-cited markdown export
    POST /api/grants/{grant_id}/ask  {"question": "..."}                  -> answer grounded in that grant's document
    GET  /api/grants/{grant_id}/source-file                               -> the grant's original PDF, if still on disk
    GET  /api/grants/{grant_id}/page-image/{page_num}                     -> that page of the grant's PDF, rendered as a PNG
    GET  /api/grants/{grant_id}/page-text-layer/{page_num}                -> every word on that page with a normalized
                                                                              bounding box, for client-side highlighting
    POST /api/grants/{grant_id}/locate  {"text": "..."}                   -> locate a requirement in the grant's
                                                                              source pages (page + text offset)
    POST /api/grants/archive {"ids": [...], "archived": true}             -> bulk mark grants outdated/archived (or restore)
    POST /api/grants/delete  {"ids": [...]}                               -> bulk permanently delete grants
    POST /api/proposals/vet (multipart form, field "file" = a proposal PDF) -> grant-agnostic completeness check
    POST /api/proposals/vet-grant-fit (multipart form, field "file" = a proposal PDF,
                                        optional form field "grant_id")    -> fit check against one grant's criteria
    POST /api/proposals/suggest-rewrite {"excerpt": "...", "criterion": "...", "suggestion": "..."}
                                        -> one on-demand rewrite draft for a single flagged excerpt (not part of
                                           the advisory-only audit above -- separate, explicitly user-invoked)
    GET  /api/grants                                                     -> the local grants database
    POST /api/categories/score  {"text": "..."}                          -> rank the grants db's focus_areas
                                                                             categories by correlation with text
    POST /api/grants/{grant_id}/relevant-categories {"text": "..."}       -> rank ONE grant's own focus_areas
                                                                             by correlation with text
    GET  /api/health                                                     -> quick status check

Everything runs on localhost and talks to Ollama on the same machine --
nothing here calls out to the internet on your behalf.
"""
from __future__ import annotations

import io
import json
import uuid
from pathlib import Path

import pdfplumber
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import agent, category_match, grant_fit, grant_ingest, grant_markdown, grant_qa, grants_matcher, ollama_client, proposal_rewrite, proposal_vet
from .tools import DATA_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
UPLOADS_DIR = DATA_DIR / "uploads"

app = FastAPI(title="Local Grant Advisor")

# In-memory conversation store. This is a single-user local tool, so a
# process-lifetime dict is enough -- restart the server to clear history.
_conversations: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    conversation_id: str
    message: str


class AnalyzeTextRequest(BaseModel):
    text: str


class ScoreCategoriesRequest(BaseModel):
    text: str
    top_k: int = 3


class GrantQuestionRequest(BaseModel):
    question: str


class LocateInGrantRequest(BaseModel):
    text: str


class SuggestRewriteRequest(BaseModel):
    excerpt: str
    criterion: str = ""
    suggestion: str = ""


class ArchiveGrantsRequest(BaseModel):
    ids: list[str]
    archived: bool = True


class DeleteGrantsRequest(BaseModel):
    ids: list[str]


class AnalyzeResponse(BaseModel):
    conversation_id: str
    summary: str
    shortlist: list
    source_pdf: str = ""


@app.get("/api/health")
def health():
    return {"status": "ok", "chat_model": ollama_client.CHAT_MODEL}


@app.get("/api/grants")
def list_grants():
    return {"grants": grants_matcher.load_grants()}


@app.post("/api/categories/score")
def score_categories_endpoint(req: ScoreCategoriesRequest):
    return {"categories": category_match.score_categories(req.text, top_k=req.top_k)}


@app.post("/api/grants/{grant_id}/relevant-categories")
def score_grant_categories_endpoint(grant_id: str, req: ScoreCategoriesRequest):
    grant = _get_grant_or_404(grant_id)
    categories = category_match.score_grant_categories(req.text, grant.get("focus_areas") or [], top_k=req.top_k)
    return {"categories": categories}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    dest = UPLOADS_DIR / safe_name
    dest.write_bytes(await file.read())

    try:
        result = agent.analyze_pdf(dest)
    except ollama_client.OllamaUnavailable as e:
        raise HTTPException(503, str(e))
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))

    conversation_id = uuid.uuid4().hex
    _conversations[conversation_id] = result["conversation"]

    return AnalyzeResponse(
        conversation_id=conversation_id,
        summary=result["summary"],
        shortlist=result["shortlist"],
        source_pdf=str(dest.relative_to(PROJECT_ROOT)),
    )


@app.post("/api/analyze-text", response_model=AnalyzeResponse)
def analyze_text(req: AnalyzeTextRequest):
    if not req.text.strip():
        raise HTTPException(400, "Please provide a non-empty text description.")

    try:
        result = agent.analyze_text(req.text)
    except ollama_client.OllamaUnavailable as e:
        raise HTTPException(503, str(e))

    conversation_id = uuid.uuid4().hex
    _conversations[conversation_id] = result["conversation"]

    return AnalyzeResponse(
        conversation_id=conversation_id,
        summary=result["summary"],
        shortlist=result["shortlist"],
    )


@app.post("/api/grants/upload")
async def upload_grant(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    dest = UPLOADS_DIR / safe_name
    dest.write_bytes(await file.read())

    try:
        grant = grant_ingest.ingest_grant_pdf(dest)
    except ollama_client.OllamaUnavailable as e:
        raise HTTPException(503, str(e))
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))

    return {"grant": grant}


# --- grants/ inbox folder: drop PDFs in directly instead of uploading one at a time ---

@app.get("/api/grants/pending")
def pending_grant_pdfs():
    """Grant PDFs sitting in grants/ that haven't been converted yet. The UI
    polls this on load to show a "convert now?" banner.
    """
    return {"pending": grant_ingest.list_pending_pdfs()}


@app.post("/api/grants/convert-pending")
def convert_pending_grants():
    pending = grant_ingest.list_pending_pdfs()
    converted = []
    failed = []
    for filename in pending:
        try:
            grant = grant_ingest.ingest_grant_pdf(grant_ingest.GRANTS_INBOX_DIR / filename)
            converted.append({"filename": filename, "grant": grant})
        except ollama_client.OllamaUnavailable as e:
            raise HTTPException(503, str(e))
        except (FileNotFoundError, ValueError) as e:
            failed.append({"filename": filename, "error": str(e)})

    return {"converted": converted, "failed": failed}

# --- end grants/ inbox folder block ---


# --- bulk archive ("mark outdated") / delete for the View Grants edit/multi-select UI ---

@app.post("/api/grants/archive")
def archive_grants(req: ArchiveGrantsRequest):
    updated = grants_matcher.set_archived(req.ids, req.archived)
    return {"grants": updated}


@app.post("/api/grants/delete")
def delete_grants_endpoint(req: DeleteGrantsRequest):
    removed = grants_matcher.delete_grants(req.ids)
    grant_ingest.delete_markdown_exports(removed)
    return {"deleted": removed}

# --- end archive/delete block ---


# --- grant markdown export + per-grant Q&A ---

@app.get("/api/grants/{grant_id}/markdown")
def get_grant_markdown(grant_id: str):
    path = grant_ingest.GRANTS_MARKDOWN_DIR / f"{grant_id}.md"
    if not path.exists():
        raise HTTPException(404, f"No markdown export found for grant '{grant_id}'.")
    return {"markdown": path.read_text()}


@app.post("/api/grants/{grant_id}/ask")
def ask_grant_question(grant_id: str, req: GrantQuestionRequest):
    if not req.question.strip():
        raise HTTPException(400, "Please provide a non-empty question.")

    try:
        answer = grant_qa.answer_grant_question(grant_id, req.question)
    except ollama_client.OllamaUnavailable as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))

    return {"answer": answer}

# --- end grant markdown export + per-grant Q&A ---


# --- grant source view: original PDF (rendered page images, not a native
# viewer -- see below) + per-page locate, for the web UI's "view in grant
# document" panel (clicking a flagged criterion that has no proposal-side
# evidence to show, so the only useful thing to show is where the
# requirement itself came from in the grant's own document) ---

def _get_grant_or_404(grant_id: str) -> dict:
    grants = grants_matcher.load_grants()
    grant = next((g for g in grants if g["id"] == grant_id), None)
    if not grant:
        raise HTTPException(404, f"Grant '{grant_id}' not found.")
    return grant


def _resolve_source_file(grant: dict) -> Path | None:
    """The grant's original PDF, if `source_file` is set and still exists on
    disk -- resolved and checked to stay inside the project root (source_file
    is always server-generated at ingestion, never user input, but a
    resolved-path check is cheap insurance)."""
    source_file = grant.get("source_file")
    if not source_file:
        return None
    path = (PROJECT_ROOT / source_file).resolve()
    if PROJECT_ROOT.resolve() not in path.parents or not path.is_file():
        return None
    return path


@app.get("/api/grants/{grant_id}/source-file")
def get_grant_source_file(grant_id: str):
    grant = _get_grant_or_404(grant_id)
    path = _resolve_source_file(grant)
    if not path:
        raise HTTPException(404, f"No source file available for grant '{grant_id}'.")
    return FileResponse(path, media_type="application/pdf")


@app.get("/api/grants/{grant_id}/page-image/{page_num}")
def get_grant_page_image(grant_id: str, page_num: int):
    """Renders one page of the grant's original PDF as a PNG -- used instead
    of embedding the PDF in a native <iframe> viewer, since a native viewer
    is a black box that can't have a JS-drawn highlight box overlaid on it.
    Rendered on request, not cached -- this is a low-traffic, single-user
    local tool, and pdfplumber's pypdfium2-backed rendering is fast enough
    (a few hundred ms) that repeat-click caching isn't worth the complexity.
    """
    grant = _get_grant_or_404(grant_id)
    path = _resolve_source_file(grant)
    if not path:
        raise HTTPException(404, f"No source file available for grant '{grant_id}'.")

    with pdfplumber.open(path) as pdf:
        if page_num < 1 or page_num > len(pdf.pages):
            raise HTTPException(404, f"Page {page_num} out of range -- this document has {len(pdf.pages)} pages.")
        image = pdf.pages[page_num - 1].to_image(resolution=150)
        buf = io.BytesIO()
        image.save(buf, format="PNG")

    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/api/grants/{grant_id}/page-text-layer/{page_num}")
def get_grant_page_text_layer(grant_id: str, page_num: int):
    """Every word on one page of the grant's original PDF, with its bounding
    box normalized to fractional (0..1) page coordinates -- this is the raw
    material the frontend renders its own text layer from: an invisible,
    selectable span per word (for copy/select over the page image) plus a
    client-side highlight search (findWordRun in static/index.html, a JS port
    of the same decreasing-prefix/punctuation-tolerant matching grant_markdown
    uses for page citation) that can highlight ANY needle -- including
    several at once -- without a server round trip per highlight. This
    replaces the old approach of the server computing one fixed highlight box
    per /locate call.
    """
    grant = _get_grant_or_404(grant_id)
    path = _resolve_source_file(grant)
    if not path:
        raise HTTPException(404, f"No source file available for grant '{grant_id}'.")

    with pdfplumber.open(path) as pdf:
        if page_num < 1 or page_num > len(pdf.pages):
            raise HTTPException(404, f"Page {page_num} out of range -- this document has {len(pdf.pages)} pages.")
        page = pdf.pages[page_num - 1]
        pw, ph = page.width, page.height
        words = [
            {
                "text": w["text"],
                "x0": w["x0"] / pw,
                "top": w["top"] / ph,
                "x1": w["x1"] / pw,
                "bottom": w["bottom"] / ph,
            }
            for w in page.extract_words()
        ]

    return {"words": words}


@app.post("/api/grants/{grant_id}/locate")
def locate_in_grant(grant_id: str, req: LocateInGrantRequest):
    grant = _get_grant_or_404(grant_id)
    source_path = _resolve_source_file(grant)
    has_source_file = source_path is not None

    pages_file = grant_ingest.pages_path(grant_id)
    if not pages_file.exists():
        return {"located": False, "has_source_file": has_source_file}

    pages = json.loads(pages_file.read_text())
    result = grant_markdown.locate_source(pages, req.text)
    if not result:
        return {"located": False, "has_source_file": has_source_file}

    return {
        "located": True,
        "has_source_file": has_source_file,
        "page": result["page"],
        "index": result["index"],
        "length": result["length"],
        "page_text": result["page_text"],
    }

# --- end grant source view block ---


# --- /api/proposals/vet: grant-agnostic completeness check (advisory-only; see app/proposal_vet.py) ---

@app.post("/api/proposals/vet")
async def vet_proposal(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    dest = UPLOADS_DIR / safe_name
    dest.write_bytes(await file.read())

    try:
        result = proposal_vet.vet_proposal_completeness(dest)
    except ollama_client.OllamaUnavailable as e:
        raise HTTPException(503, str(e))
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))

    return result

# --- end /api/proposals/vet block ---


# --- /api/proposals/vet-grant-fit: fit check against one grant's specific criteria (advisory-only; see app/grant_fit.py) ---

@app.post("/api/proposals/vet-grant-fit")
async def vet_grant_fit_endpoint(file: UploadFile = File(...), grant_id: str | None = Form(None)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    dest = UPLOADS_DIR / safe_name
    dest.write_bytes(await file.read())

    try:
        result = grant_fit.vet_grant_fit(dest, grant_id=grant_id)
    except ollama_client.OllamaUnavailable as e:
        raise HTTPException(503, str(e))
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))

    return result

# --- end /api/proposals/vet-grant-fit block ---


# --- /api/proposals/suggest-rewrite: on-demand rewrite draft for ONE flagged excerpt, invoked explicitly by
# the user from the web UI -- separate from the advisory-only audit above, see app/proposal_rewrite.py ---

@app.post("/api/proposals/suggest-rewrite")
def suggest_rewrite_endpoint(req: SuggestRewriteRequest):
    try:
        rewrite = proposal_rewrite.suggest_rewrite(req.excerpt, req.criterion, req.suggestion)
    except ollama_client.OllamaUnavailable as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))

    return {
        "rewrite": rewrite,
        "note": "Model-generated draft to adapt -- review before using; this tool never finalizes proposal text automatically.",
    }

# --- end /api/proposals/suggest-rewrite block ---


@app.post("/api/chat")
def chat(req: ChatRequest):
    conversation = _conversations.get(req.conversation_id)
    if conversation is None:
        raise HTTPException(404, "Unknown conversation_id. Run /api/analyze first.")

    try:
        result = agent.continue_chat(conversation, req.message)
    except ollama_client.OllamaUnavailable as e:
        raise HTTPException(503, str(e))

    _conversations[req.conversation_id] = result["conversation"]
    return {"conversation_id": req.conversation_id, "summary": result["summary"]}


# Serve the minimal bundled UI at the site root.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return JSONResponse({"message": "Local Grant Advisor API is running."})
