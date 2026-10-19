"""Streamlit UI for DocuQuery RAG Assistant."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import streamlit as st

from document_loader import UnsupportedFileTypeError, load_many, open_paths, supported_files_in
from rag_engine import GeminiRAG, build_chunks, default_model, resolve_llm_provider


APP_TITLE = "DocuQuery RAG Assistant"
MAX_FILES = 5
MAX_TOTAL_BYTES = 15 * 1024 * 1024


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


def get_llm_settings() -> tuple[str, str]:
    """Answers use Groq (open-source models) when GROQ_API_KEY is set, otherwise Gemini.
    LLM_PROVIDER = "gemini" or "groq" in secrets forces one of them."""

    groq_key = secret_value("GROQ_API_KEY")
    provider = resolve_llm_provider(secret_value("LLM_PROVIDER") or None, groq_key)
    return provider, groq_key


def get_rag(api_key: str) -> GeminiRAG:
    """Reuse one RAG service (LangChain clients) per session, provider and API keys."""

    provider, groq_key = get_llm_settings()
    key_hash = hashlib.sha256(f"{provider}|{api_key}|{groq_key}".encode("utf-8")).hexdigest()
    if st.session_state.get("rag_key_hash") != key_hash:
        st.session_state.rag = GeminiRAG(api_key=api_key, llm_provider=provider, llm_api_key=groq_key)
        st.session_state.rag_key_hash = key_hash
    return st.session_state.rag


def reset_index() -> None:
    index = st.session_state.get("vector_index")
    if index is not None:
        index.delete()  # free the ChromaDB collection for this session
    for key in ("chunks", "vector_index", "document_signature", "messages", "document_names"):
        st.session_state.pop(key, None)


def app_header() -> None:
    st.title(APP_TITLE)
    st.caption(
        "Upload documents, index them in a ChromaDB vector store, and ask questions answered only from your content."
    )


def sidebar(api_key_ready: bool) -> tuple[str, int, int, int]:
    with st.sidebar:
        st.header("Project controls")
        if api_key_ready:
            provider, _ = get_llm_settings()
            model = default_model(provider)
            st.success(f"Search: Gemini embeddings  \nAnswers: {model} via {provider.title()}")
        else:
            st.warning("Add a Gemini API key")
        chunking_label = st.radio(
            "Chunking strategy",
            ["By section (recommended)", "Fixed size"],
            help="By section splits on document headings first, so each chunk covers one topic.",
        )
        chunking = "sections" if chunking_label.startswith("By section") else "fixed"
        chunk_size = st.slider("Chunk size", 500, 1400, 900, 100)
        overlap = st.slider("Chunk overlap", 50, 300, 150, 50)
        top_k = st.slider("Sources per answer", 2, 6, 4)
        st.divider()
        st.markdown(
            "**RAG flow**\n\n"
            "1. Extract text\n"
            "2. Split by section headings, then into chunks (LangChain text splitter)\n"
            "3. Create Gemini embeddings\n"
            "4. Store vectors in ChromaDB\n"
            "5. Retrieve similar chunks (cosine similarity)\n"
            "6. Generate a grounded answer (LangChain chain; Llama 3.3 via Groq or Gemini)"
        )
        st.warning(
            "Use only non-confidential demo files. Document text is sent to the Gemini API (search) "
            "and to the answer model's provider."
        )
        if st.button("Clear documents and chat", use_container_width=True):
            reset_index()
            st.rerun()
    return chunking, chunk_size, overlap, top_k


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")
    require_optional_password()
    app_header()

    api_key = get_api_key()
    chunking, chunk_size, overlap, top_k = sidebar(bool(api_key))

    uploaded_files = st.file_uploader(
        "Upload up to five files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Maximum combined upload size: 15 MB.",
    )

    col1, col2 = st.columns([1, 1])
    process_clicked = col1.button("Process uploaded documents", type="primary", use_container_width=True)
    demo_clicked = col2.button("Load included demo documents", use_container_width=True)

    files_to_process = list(uploaded_files or [])
    if demo_clicked:
        demo_dir = Path(__file__).parent / "sample_documents"
        files_to_process = open_paths(supported_files_in(demo_dir))
        process_clicked = True

    if process_clicked:
        if not api_key:
            st.error("Add a Gemini API key in the sidebar or configure it in Streamlit secrets.")
        elif not files_to_process:
            st.error("Upload at least one document or load the demo documents.")
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
                    chunks = build_chunks(parts, chunk_size=chunk_size, overlap=overlap, strategy=chunking)
                    if not chunks:
                        raise ValueError("No searchable chunks could be created.")

                    st.write("Creating embeddings and storing them in ChromaDB")
                    rag = get_rag(api_key)
                    vector_index = rag.build_index(chunks)

                    reset_index()  # drop the previous collection before switching
                    st.session_state.chunks = chunks
                    st.session_state.vector_index = vector_index
                    st.session_state.document_names = sorted({chunk.source for chunk in chunks})
                    st.session_state.document_signature = calculate_file_signature(files_to_process)
                    st.session_state.messages = []
                    status.update(label="Documents are ready", state="complete", expanded=False)
            except UnsupportedFileTypeError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not process the documents: {exc}")

    if "vector_index" in st.session_state:
        names = ", ".join(st.session_state.document_names)
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Documents", len(st.session_state.document_names))
        metric2.metric("Searchable chunks", len(st.session_state.chunks))
        metric3.metric("Vector store", "ChromaDB")
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
                        rag = get_rag(api_key)
                        results = rag.retrieve(
                            st.session_state.vector_index,
                            question,
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
                    st.error(
                        f"Could not answer the question: {exc}. "
                        "If the session was idle for a long time, process the documents again."
                    )
    else:
        st.markdown(
            "### Good demo questions\n"
            "The demo set is a synthetic retail pipeline runbook, data quality standards, "
            "incident history and an onboarding FAQ.\n\n"
            "- What happens when an ETL job fails?\n"
            "- What was the root cause of the linked service authentication failure?\n"
            "- How is a sales record with an unknown product identifier handled?\n"
            "- By what time must the Gold tables be ready each day?\n"
            "- Who is the CEO of the company? *(not in the documents, so it should decline)*"
        )


if __name__ == "__main__":
    main()
