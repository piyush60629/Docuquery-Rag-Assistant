"""Document text extraction helpers for PDF, DOCX, TXT, and Markdown files."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterable

from docx import Document
from pypdf import PdfReader


@dataclass(frozen=True)
class DocumentPart:
    """A text section extracted from one source document."""

    source: str
    text: str
    page: int | None = None


class UnsupportedFileTypeError(ValueError):
    """Raised when a file type is not supported by the application."""


def _read_bytes(file_obj: BinaryIO) -> bytes:
    if hasattr(file_obj, "getvalue"):
        return file_obj.getvalue()
    file_obj.seek(0)
    return file_obj.read()


def extract_pdf(data: bytes, source: str) -> list[DocumentPart]:
    reader = PdfReader(BytesIO(data))
    parts: list[DocumentPart] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            parts.append(DocumentPart(source=source, page=page_number, text=text))
    return parts


def extract_docx(data: bytes, source: str) -> list[DocumentPart]:
    document = Document(BytesIO(data))
    blocks: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name if paragraph.style is not None else "") or ""
        # Keep Word headings as Markdown headings so section-aware chunking can use them.
        if style == "Title":
            text = f"# {text}"
        elif style.startswith("Heading"):
            level = style.removeprefix("Heading").strip()
            depth = min(int(level) + 1, 6) if level.isdigit() else 2
            text = f"{'#' * depth} {text}"
        blocks.append(text)
    text = "\n\n".join(blocks)
    return [DocumentPart(source=source, text=text)] if text else []


def extract_text(data: bytes, source: str) -> list[DocumentPart]:
    text = data.decode("utf-8", errors="replace").strip()
    return [DocumentPart(source=source, text=text)] if text else []


def load_uploaded_file(file_obj: BinaryIO) -> list[DocumentPart]:
    """Extract text from a Streamlit UploadedFile or file-like object."""

    source = getattr(file_obj, "name", "uploaded_document")
    suffix = Path(source).suffix.lower()
    data = _read_bytes(file_obj)

    if suffix == ".pdf":
        return extract_pdf(data, source)
    if suffix == ".docx":
        return extract_docx(data, source)
    if suffix in {".txt", ".md"}:
        return extract_text(data, source)

    raise UnsupportedFileTypeError(
        f"Unsupported file type '{suffix or 'unknown'}'. Use PDF, DOCX, TXT, or MD."
    )


def load_many(files: Iterable[BinaryIO]) -> list[DocumentPart]:
    parts: list[DocumentPart] = []
    for file_obj in files:
        parts.extend(load_uploaded_file(file_obj))
    return parts


SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}


class NamedBytesIO(BytesIO):
    """In-memory file with a `name`, so local files look like Streamlit uploads."""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def open_paths(paths: Iterable[str | Path]) -> list[NamedBytesIO]:
    return [NamedBytesIO(Path(path).read_bytes(), Path(path).name) for path in paths]


def supported_files_in(directory: str | Path) -> list[Path]:
    return sorted(
        path for path in Path(directory).iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def load_paths(paths: Iterable[str | Path]) -> list[DocumentPart]:
    """Extract text from files on disk (used by the demo button and the evaluation script)."""

    return load_many(open_paths(paths))
