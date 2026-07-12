"""
PDF text/table extraction using pdfplumber.

This is the first stage of the pipeline: turn an uploaded PDF (a project
description, organization profile, proposal draft, etc.) into plain text
the rest of the pipeline -- and eventually the local LLM -- can reason about.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber


@dataclass
class ExtractedPDF:
    text: str
    tables: list[list[list[str | None]]] = field(default_factory=list)
    page_count: int = 0
    source_path: str = ""
    # Per-page extracted text, 1:1 with the PDF's physical page order (index 0
    # = page 1, empty string if that page had no extractable text). Lets
    # downstream code (e.g. grant_markdown.py) cite which page a fact came
    # from, which `text` alone can't do once pages are joined together.
    pages: list[str] = field(default_factory=list)


def extract_pdf(path: str | Path) -> ExtractedPDF:
    """Extract all text and tables from a PDF file.

    Raises FileNotFoundError if the path doesn't exist, and ValueError if
    the file has no extractable text at all (e.g. a pure scanned image PDF
    with no OCR layer -- this project doesn't do OCR).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    text_parts: list[str] = []
    pages: list[str] = []
    tables: list[list[list[str | None]]] = []

    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
            if page_text.strip():
                text_parts.append(page_text)
            for table in page.extract_tables():
                tables.append(table)

    full_text = "\n\n".join(text_parts).strip()
    if not full_text:
        raise ValueError(
            f"No extractable text found in {path.name}. If this is a scanned "
            "image PDF, you'll need an OCR step before this pipeline can read it."
        )

    return ExtractedPDF(
        text=full_text,
        tables=tables,
        page_count=page_count,
        source_path=str(path),
        pages=pages,
    )


def summarize_tables(tables: list[list[list[str | None]]], max_rows: int = 5) -> str:
    """Render extracted tables as compact markdown-ish text for inclusion in a prompt."""
    if not tables:
        return ""
    chunks = []
    for i, table in enumerate(tables):
        rows = table[: max_rows + 1]  # header + up to max_rows
        rendered = "\n".join(
            " | ".join(cell or "" for cell in row) for row in rows if row
        )
        chunks.append(f"[Table {i + 1}]\n{rendered}")
    return "\n\n".join(chunks)


if __name__ == "__main__":
    import sys

    result = extract_pdf(sys.argv[1])
    print(f"Pages: {result.page_count}")
    print(f"Tables found: {len(result.tables)}")
    print("--- text preview ---")
    print(result.text[:1000])
