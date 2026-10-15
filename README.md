# DocuQuery RAG Assistant

A deployable Retrieval-Augmented Generation (RAG) project built with **LangChain**, **ChromaDB**, and **Gemini**. Users upload PDF, DOCX, TXT, or Markdown documents and ask questions. The app stores document embeddings in a ChromaDB vector store, retrieves the most relevant chunks, and asks Gemini to answer only from those chunks, with source citations and similarity scores.

## What this project demonstrates

- Document text extraction (PDF, DOCX, TXT, Markdown)
- Text cleaning and overlapping chunking with LangChain's `RecursiveCharacterTextSplitter`
- Gemini text embeddings (`gemini-embedding-001`, 768 dimensions) via `langchain-google-genai`
- **ChromaDB vector store** with cosine similarity search and chunk metadata (source file, page, chunk ID)
- Grounded answer generation with a **LangChain (LCEL) chain**: prompt → Gemini → output parser
- Source passage display with similarity scores
- Per-session isolation: every user session gets its own ChromaDB collection
- Streamlit deployment and secrets handling

## Architecture

```text
Uploaded files
    ↓
Text extraction (pypdf, python-docx)
    ↓
LangChain RecursiveCharacterTextSplitter → overlapping chunks + metadata
    ↓
Gemini embeddings (langchain-google-genai)
    ↓
ChromaDB collection (one per session, cosine distance)
    ↓
Question embedding → ChromaDB similarity search → top-k chunks
    ↓
LangChain LCEL chain: prompt | Gemini | StrOutputParser
    ↓
Grounded answer + cited source passages
```

## Free resources

- **App hosting:** Streamlit Community Cloud
- **Generation model:** Gemini Flash free tier
- **Embedding model:** Gemini Embedding free tier
- **Vector store:** ChromaDB, running in memory inside the app (no external database needed)
- **Orchestration:** LangChain
- **Source control:** GitHub free account

Free tiers have usage limits and can change. This project is intended for learning and portfolio demonstrations, not production workloads.

## Run locally

1. Install Python 3.11, 3.12, or 3.13.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your_api_key"
# APP_PASSWORD = "optional_demo_password"
```

5. Start the app:

```bash
streamlit run app.py
```

6. Run the tests (offline; they use fake embeddings and a fake LLM, so no API key is needed):

```bash
pip install pytest
python -m pytest -q
```

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository, for example `docuquery-rag-assistant`.
2. Upload all project files and push them to the `main` branch.
3. Open Streamlit Community Cloud and select **Create app**.
4. Choose the repository, `main` branch, and `app.py` as the entry file.
5. Open **Advanced settings** and add:

```toml
GEMINI_API_KEY = "your_api_key"
# APP_PASSWORD = "optional_demo_password"
```

6. Deploy and choose a simple URL such as `piyush-rag-assistant.streamlit.app` if available.
7. Add the live URL and GitHub repository to your resume and portfolio.

Never commit API keys to GitHub.

## Recommended demo flow

1. Click **Load included demo document**.
2. Ask: `What happens when the pipeline fails?`
3. Open **Retrieved source passages**.
4. Explain that the answer was generated only after semantic retrieval.
5. Ask an unsupported question to demonstrate that the assistant refuses to invent an answer.

## Privacy note

Use only public, synthetic, or non-confidential documents. Uploaded content and retrieved passages are sent to the Gemini API for embedding and answer generation. Files are held only in the active Streamlit session by this application, but the external API provider's terms still apply.

## Resume wording after deployment

**DocuQuery – RAG Knowledge Assistant | Python, LangChain, ChromaDB, Gemini API, Streamlit**

- Built and deployed a RAG application that chunks uploaded PDF, DOCX, TXT, and Markdown files with LangChain, stores Gemini embeddings in a ChromaDB vector store, and retrieves relevant passages using cosine similarity search.
- Generated grounded answers through a LangChain (LCEL) chain that cites source passages and similarity scores to reduce hallucinated responses.

## Limitations and next improvements

- ChromaDB runs in memory, so each session's index resets when the session ends or the app restarts. Collections from sessions idle for 6+ hours are removed automatically.
- Scanned PDFs require OCR, which is not included in this version.
- Planned next: a retrieval evaluation set (hit-rate and faithfulness), hybrid keyword + vector search or a reranker, chat memory, and an optional local Ollama backend.
- A production version should add authentication, persistent vector storage, monitoring, and stronger document-access controls.
