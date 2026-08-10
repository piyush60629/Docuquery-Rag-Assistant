# DocuQuery RAG Assistant

A small, deployable Retrieval-Augmented Generation project for a portfolio. Users upload PDF, DOCX, TXT, or Markdown documents and ask questions. The app retrieves relevant document chunks and asks Gemini to answer only from those chunks.

## What this project demonstrates

- Document text extraction
- Text cleaning and overlapping chunking
- Gemini text embeddings
- NumPy cosine-similarity vector search
- Grounded prompt construction
- Source passage display
- Streamlit deployment and secrets handling

## Architecture

```text
Uploaded files
    ↓
Text extraction
    ↓
Overlapping chunks
    ↓
Gemini embeddings
    ↓
In-memory NumPy vector index
    ↓
Question embedding + cosine similarity
    ↓
Top matching chunks
    ↓
Gemini grounded answer + source passages
```

## Free resources

- **App hosting:** Streamlit Community Cloud
- **Generation model:** Gemini 2.5 Flash free tier
- **Embedding model:** Gemini Embedding free tier
- **Vector search:** NumPy, running inside the app
- **Source control:** GitHub free account

Free tiers have usage limits and can change. This project is intended for learning and portfolio demonstrations, not production workloads.

## Run locally

1. Install Python 3.11 or 3.12.
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

**RAG Knowledge Assistant | Python, Streamlit, Gemini API, Embeddings**

- Built and deployed a document-based RAG application that extracts and chunks uploaded files, creates embeddings, and retrieves relevant passages using cosine similarity.
- Generated answers only from retrieved document content and displayed supporting source passages to improve transparency.

## Limitations and next improvements

- The index is stored in memory and resets when the session ends.
- Scanned PDFs require OCR, which is not included in this version.
- A production version should add authentication, persistent vector storage, monitoring, evaluation, and stronger document-access controls.
