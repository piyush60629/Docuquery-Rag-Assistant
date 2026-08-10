"""Streamlit UI for DocuQuery RAG Assistant."""

from __future__ import annotations

import hashlib
import os
from io import BytesIO
from pathlib import Path

import streamlit as st

from document_loader import UnsupportedFileTypeError, load_many
from rag_engine import GeminiRAG, build_chunks


APP_TITLE = "DocuQuery RAG Assistant"
MAX_FILES = 5
MAX_TOTAL_BYTES = 15 * 1024 * 1024


class NamedBytesIO(BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def secret_value(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return os.getenv(name, default)


def require_optional_password() -> None:
    expected = secret_value("APP_PASSWORD")
    if not expected:
        return

    if st.session_state.get("authenticated"):
        return

    st.title(APP_TITLE)
    st.caption("Private portfolio demo")
    supplied = st.text_input("Demo password", type="password")
    if st.button("Open demo", use_container_width=True):
        if supplied == expected:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


def get_api_key() -> str:
    stored = secret_value("GEMINI_API_KEY")
    if stored:
        return stored
    return st.sidebar.text_input(
        "Gemini API key",
        type="password",
        help="The key is used only for the current session and is not written to disk.",
    )


def calculate_file_signature(files: list) -> str:
    digest = hashlib.sha256()
    for file_obj in files:
        digest.update(file_obj.name.encode("utf-8"))
        digest.update(file_obj.getvalue())
    return digest.hexdigest()


def reset_index() -> None:
    for key in ("chunks", "embeddings", "document_signature", "messages", "document_names"):
        st.session_state.pop(key, None)


def app_header() -> None:
    st.title(APP_TITLE)
    st.caption(
        "Upload documents, build a small semantic index, and ask questions answered only from your content."
    )


def sidebar(api_key_ready: bool) -> tuple[int, int, int]:
    with st.sidebar:
        st.header("Project controls")
        st.success("Gemini API connected" if api_key_ready else "Add a Gemini API key")
        chunk_size = st.slider("Chunk size", 500, 1400, 900, 100)
        overlap = st.slider("Chunk overlap", 50, 300, 150, 50)
        top_k = st.slider("Sources per answer", 2, 6, 4)
        st.divider()
        st.markdown(
            "**RAG flow**\n\n"
            "1. Extract text\n"
            "2. Split into chunks\n"
            "3. Create embeddings\n"
            "4. Retrieve similar chunks\n"
            "5. Generate a grounded answer"
        )
        st.warning(
            "Use only non-confidential demo files. Relevant document text is sent to the Gemini API."
        )
        if st.button("Clear documents and chat", use_container_width=True):
            reset_index()
            st.rerun()
    return chunk_size, overlap, top_k


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")
    require_optional_password()
    app_header()

    api_key = get_api_key()
    chunk_size, overlap, top_k = sidebar(bool(api_key))

    uploaded_files = st.file_uploader(
        "Upload up to five files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Maximum combined upload size: 15 MB.",
    )

    col1, col2 = st.columns([1, 1])
    process_clicked = col1.button("Process uploaded documents", type="primary", use_container_width=True)
    demo_clicked = col2.button("Load included demo document", use_container_width=True)

    files_to_process = list(uploaded_files or [])
    if demo_clicked:
        demo_path = Path(__file__).parent / "sample_documents" / "retail_pipeline_runbook.txt"
        files_to_process = [NamedBytesIO(demo_path.read_bytes(), demo_path.name)]
        process_clicked = True

    if process_clicked:
        if not api_key:
            st.error("Add a Gemini API key in the sidebar or configure it in Streamlit secrets.")
        elif not files_to_process:
            st.error("Upload at least one document or load the demo document.")
        elif len(files_to_process) > MAX_FILES:
            st.error(f"Upload no more than {MAX_FILES} files.")
        elif sum(len(file_obj.getvalue()) for file_obj in files_to_process) > MAX_TOTAL_BYTES:
            st.error("The combined file size is above 15 MB.")
        else:
            try:
                with st.status("Building the document index...", expanded=True) as status:
                    st.write("Extracting text")
                    parts = load_many(files_to_process)
                    if not parts:
                        raise ValueError("No readable text was found in the selected files.")

                    st.write("Splitting text into chunks")
                    chunks = build_chunks(parts, chunk_size=chunk_size, overlap=overlap)
                    if not chunks:
                        raise ValueError("No searchable chunks could be created.")

                    st.write("Creating embeddings")
                    rag = GeminiRAG(api_key=api_key)
                    embeddings = rag.embed_documents(chunks)

                    st.session_state.chunks = chunks
                    st.session_state.embeddings = embeddings
                    st.session_state.document_names = sorted({chunk.source for chunk in chunks})
                    st.session_state.document_signature = calculate_file_signature(files_to_process)
                    st.session_state.messages = []
                    status.update(label="Documents are ready", state="complete", expanded=False)
            except UnsupportedFileTypeError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not process the documents: {exc}")

    if "chunks" in st.session_state:
        names = ", ".join(st.session_state.document_names)
        metric1, metric2 = st.columns(2)
        metric1.metric("Documents", len(st.session_state.document_names))
        metric2.metric("Searchable chunks", len(st.session_state.chunks))
        st.info(f"Ready to answer questions from: {names}")

        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("Retrieved source passages"):
                        for source in message["sources"]:
                            st.markdown(
                                f"**{source['label']}** · similarity `{source['score']:.3f}`"
                            )
                            st.write(source["text"])

        question = st.chat_input("Ask a question about the uploaded documents")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Searching the documents..."):
                        rag = GeminiRAG(api_key=api_key)
                        results = rag.retrieve(
                            question,
                            st.session_state.chunks,
                            st.session_state.embeddings,
                            top_k=top_k,
                        )
                        answer = rag.answer(question, results)
                    st.markdown(answer)
                    sources = [
                        {
                            "label": result.chunk.label,
                            "score": result.score,
                            "text": result.chunk.text,
                        }
                        for result in results
                    ]
                    with st.expander("Retrieved source passages"):
                        for source in sources:
                            st.markdown(
                                f"**{source['label']}** · similarity `{source['score']:.3f}`"
                            )
                            st.write(source["text"])
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as exc:
                    st.error(f"Could not answer the question: {exc}")
    else:
        st.markdown(
            "### Good demo questions\n"
            "- What happens when an ETL job fails?\n"
            "- Which data quality checks are performed?\n"
            "- How is an incremental load validated?"
        )


if __name__ == "__main__":
    main()
