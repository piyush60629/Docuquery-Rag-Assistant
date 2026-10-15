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
- **RAG evaluation**: retrieval Hit@k and MRR, LLM-as-judge faithfulness and correctness, and refusal accuracy on unanswerable questions
- Streamlit deployment, secrets handling, and GitHub Actions for tests and evaluation

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

## Evaluation

DocuQuery is evaluated on **35 questions** (`eval/dataset.json`) about the four synthetic documents in `sample_documents/`: a pipeline runbook, data quality standards, an incident history, and an onboarding FAQ. Five questions are deliberately **not answerable** from the documents.

| Metric | What it measures |
|---|---|
| Retrieval Hit@1 / Hit@k | Share of answerable questions where a chunk containing the evidence is ranked first / in the top k |
| Retrieval MRR | Mean reciprocal rank of the first chunk with the evidence |
| Answer faithfulness | LLM judge: are all claims in the answer supported by the retrieved passages? |
| Answer correctness | LLM judge: does the answer match the reference answer? |
| Refusal accuracy | Share of unanswerable questions where the assistant declines instead of guessing |
| False refusal rate | Share of answerable questions where it wrongly declines (lower is better) |

Evidence is matched by phrase rather than chunk ID, so results stay comparable when the chunk size changes.

<!-- EVAL_RESULTS_START -->
**Latest results**

- Run: 2026-10-15 17:40 UTC  
- Mode: retrieval · chunk size 900 · overlap 150 · top-k 4  
- Models: embeddings `gemini-embedding-001`, answers `gemini-3.5-flash`, judge ``  
- Questions: 35 (30 answerable, 5 unanswerable) · chunks indexed: 13 · errors: 0

| Metric | Score |
|---|---|
| Retrieval Hit@1 | 80.0% |
| Retrieval Hit@4 | 96.7% |
| Retrieval MRR | 0.867 |

Full per-question results: `eval/results/latest.md`.
<!-- EVAL_RESULTS_END -->

**Run it on GitHub (no local setup):** add a repository secret named `GEMINI_API_KEY` (Settings → Secrets and variables → Actions), then open **Actions → Evaluate RAG → Run workflow**. Choose `retrieval` for the search scores (about 1 minute) or `full` to also score answers and refusals (about 8 minutes). Progress is printed live in the run log; the results appear in the run summary and are committed to `eval/results/` and this README. If the API key is wrong or the free-tier quota is used up, the run stops within seconds and explains why.

**Run it locally:**

```bash
export GEMINI_API_KEY=your_key            # Windows PowerShell: $env:GEMINI_API_KEY="your_key"
python evaluate.py --mode retrieval       # retrieval metrics only, fast
python evaluate.py --mode full            # adds answers + LLM-as-judge scores (~8 minutes on the free tier)
python evaluate.py --mode full --chunk-size 600 --label "smaller chunks"   # compare a configuration
```

Every run adds a row to `eval/results/history.csv`, so configurations can be compared before and after a change.

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

1. Click **Load included demo documents**.
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
- The evaluation corpus is small (4 documents, about 13 chunks), so Hit@1 and MRR are the most informative retrieval metrics. The judge uses the same Gemini model family as the generator, which can be slightly lenient.
- Planned next: hybrid keyword + vector search or a reranker (measured against this evaluation set), chat memory, and an optional local Ollama backend.
- A production version should add authentication, persistent vector storage, monitoring, and stronger document-access controls.
