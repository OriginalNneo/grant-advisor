# Grant Advisor

A privacy-first, local-only assistant for grant matching and proposal vetting.

Grant Advisor helps a committee do two things: find grants that fit an event idea, and check a
written proposal against a funder's requirements. Everything runs on your own machine through
[Ollama](https://ollama.com). There is no cloud API, no account, no telemetry, and no network call
that leaves your computer. Your proposals and grant documents never go anywhere.

> Demo video: https://youtu.be/44-u2L9U4KY

---

## What it does

The tool maps onto the two moments where a funding cycle gets stuck:

1. **Discovery phase** — before a proposal exists. Type in a short project idea (or upload a
   workplan) and get a ranked shortlist of grants that fit, with a relevance score for each.
2. **Vetting phase** — once a draft exists. Upload the proposal and check it two ways:
   - **Completeness**: does it contain what any funding proposal should have (purpose, audience,
     budget breakdown, timeline, outcomes, sustainability, accountability)?
   - **Grant fit**: does it satisfy one specific grant's stated requirements, point by point?

Every finding is **advisory**. The tool reports a status, a quoted excerpt from your own text, and a
plain-language suggestion. It never rewrites your proposal on its own. An opt-in "Suggest a rewrite"
button can draft a single flagged excerpt when you ask for it, and always labels the result as a
draft.

---

## Requirements

- **Python 3.10 or newer**
- **[Ollama](https://ollama.com/download)** installed and running locally
- Enough RAM to run a small local model (8 GB works; 16 GB or more is comfortable)

That is the whole list. No database server, no API keys, no internet connection at run time.

---

## Install

```bash
# 1. Clone the repo
git clone https://github.com/OriginalNneo/grant-advisor.git
cd grant-advisor

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Pull the models Ollama will run (see "Choosing a model" below if these tags are unavailable)
ollama pull gemma4:e2b              # chat / reasoning model
ollama pull nomic-embed-text        # embedding model used for grant matching

# 5. Start Ollama (leave this running in its own terminal)
ollama serve
```

### Choosing a model

The default chat model is `gemma4:e2b`. If that tag is not in your Ollama library, use any local
chat model instead and point the app at it with an environment variable:

```bash
ollama pull llama3.2                 # or gemma2:2b, qwen2.5, mistral, etc.
export OLLAMA_CHAT_MODEL=llama3.2    # the app reads this on startup
```

The embedding model (`nomic-embed-text`) is a standard Ollama model, used only for ranking grants.
If Ollama is unreachable, grant matching automatically falls back to a simple keyword overlap so the
app still runs.

---

## Run

### Web app (recommended)

```bash
source .venv/bin/activate
uvicorn app.server:app --port 8000
```

Then open **http://localhost:8000** in a browser. The sidebar has four views: Upload Grant,
View Grants, Find Grant, and Vet Proposal.

**macOS shortcut:** double-click `Launch Grant Advisor.command`. It starts Ollama and the server if
they are not already running, opens the app in Firefox, and streams the server log. This helper is
macOS-only; on other systems use the `uvicorn` command above.

### Command line

Every web feature is also a CLI command, calling the exact same code:

```bash
# Discovery: find grants for an idea
python -m app.cli describe "a 3-day cultural music festival for youth in the community"
python -m app.cli analyze path/to/project.pdf
python -m app.cli chat path/to/project.pdf          # analyze, then ask follow-ups

# Add a grant to the database
python -m app.cli add-grant path/to/grant.pdf

# Vetting
python -m app.cli vet-proposal path/to/proposal.pdf
python -m app.cli vet-grant-fit path/to/proposal.pdf [--grant-id ID]
```

---

## First run: build your grants database

The repository ships **without any grant data**. The database is created empty on first run. Add
grants in any of these ways:

1. **Web UI** — the "Upload Grant" panel. Choose a grant PDF and click Upload & Parse.
2. **Inbox folder** — drop grant PDFs into the `grants/` folder. On the next page load, the web UI
   shows a banner offering to convert them all at once.
3. **CLI** — `python -m app.cli add-grant path/to/grant.pdf`.
4. **CSV import** — `python grants_db/csv_to_json.py my_grants.csv grants_db/grants.json`
   (see `grants_db/grants_template.csv` for the column format).

### Try it with the included samples

The `examples/` folder has two sample files you can use right away:

```bash
# Ingest a real sample grant (the 2024 FutureYOUth Fund information kit)
python -m app.cli add-grant "examples/FutureYOUth_Fund_2024_Information_Kit.pdf"

# Find matching grants for a sample project idea
python -m app.cli analyze examples/sample_project.pdf
```

After ingesting a grant, upload a proposal in the "Vet Proposal" view and check it against that
grant's criteria.

> **Note on documents:** scanned or image-only PDFs are not supported. The tool reads a PDF's text
> layer with `pdfplumber`; it does not do OCR, so a file with no text layer will error out instead
> of being read.

---

## How it works

```
                 Web UI  |  CLI  |  any HTTP client
                              |
                      app/server.py (FastAPI)
        thin shell over the same core functions the CLI calls
                              |
   +----------+-----------+-----------+--------------+-----------+-----------+
   | grant    | grant     | grant     | completeness | grant fit | rewrite   |
   | matching | ingestion | Q&A       | vetting      | vetting   | draft     |
   | agent.py | grant_    | grant_    | proposal_    | grant_    | proposal_ |
   |          | ingest.py | qa.py     | vet.py       | fit.py    | rewrite.py|
   +----------+-----------+-----------+--------------+-----------+-----------+
                              |
   Shared building blocks:
     pdf_extract.py    pdfplumber text + tables (no OCR)
     ollama_client.py  tool-calling loop | schema-constrained JSON (1 retry) | plain chat
     grants_matcher.py embed + cosine-similarity ranking, keyword fallback
     chunking.py       fallback for documents that overflow the context window
     tools.py          agent tools, file access sandboxed to data/
                              |
   Ollama (localhost:11434) + grants_db/grants.json   (all on this machine)
```

- **Structured output.** Ingestion and both vetting checks force the model to answer against a
  fixed JSON schema, one entry per checklist item, so it cannot skip, reorder, or invent items.
- **Grounded evidence.** Each finding quotes your actual text. The web UI locates that quote in the
  document and highlights it, so you can verify the model's judgment in seconds.
- **Long documents.** Full text is passed with a raised context window (64k tokens). If a document
  is larger than the window, `chunking.py` splits it into overlapping segments, runs the same
  checklist on each, and merges the results by strongest status per item.

The code is small and readable; start in `app/` to follow any pipeline end to end.

---

## Configuration

Set these environment variables before starting the server or a CLI command to override defaults:

| Variable | Default | What it does |
|---|---|---|
| `OLLAMA_CHAT_MODEL` | `gemma4:e2b` | Chat/reasoning model Ollama runs |
| `OLLAMA_NUM_CTX` | `32768` | Default context window (tokens) |
| `OLLAMA_MAX_TOOL_ITERATIONS` | `8` | Tool-call loop cap for grant discovery |

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Couldn't reach Ollama` / HTTP 503 | Ollama is not running. Start it with `ollama serve`. |
| `model 'gemma4:e2b' not found` | Pull it (`ollama pull gemma4:e2b`) or set `OLLAMA_CHAT_MODEL` to a model you have. |
| Grant matching works but vetting fails | Vetting needs the chat model. Make sure it is pulled and Ollama has enough RAM. |
| A PDF errors immediately on upload | It is likely a scanned/image-only PDF. There is no OCR; use a PDF with a real text layer. |
| Responses are slow | A small model on CPU can take tens of seconds per vet. This is expected; a GPU is much faster. |

---

## Privacy

- No external API is ever called.
- No document is uploaded to any cloud service.
- Ollama, the models, and the grants database all run and live on your machine.

You can prove it: disconnect from the internet and the app still works.

---

## License

Released under the MIT License. See `LICENSE`.
