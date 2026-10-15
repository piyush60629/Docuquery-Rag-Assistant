# Deployment Checklist

- [ ] Create a Gemini API key in Google AI Studio.
- [ ] Test locally using `streamlit run app.py`.
- [ ] Create a public GitHub repository named `docuquery-rag-assistant`.
- [ ] Upload the project files; never upload `.streamlit/secrets.toml`.
- [ ] Create a Streamlit Community Cloud app using `app.py`.
- [ ] Add `GEMINI_API_KEY` under Advanced settings → Secrets.
- [ ] Optionally add `APP_PASSWORD` for a controlled recruiter demo.
- [ ] Test the included runbook and at least one PDF.
- [ ] Add the live app URL and GitHub URL to the portfolio.
- [ ] Add the project to the resume only after the live test succeeds.

## Upgrading the live app to LangChain + ChromaDB

- [ ] Push the `upgrade/chromadb-langchain` branch to GitHub (do not merge into `main` yet).
- [ ] In Streamlit Community Cloud, create a **second, temporary app** from the same repo, branch `upgrade/chromadb-langchain`, entry file `app.py`, Python 3.12.
- [ ] Add the same `GEMINI_API_KEY` (and `APP_PASSWORD`, if used) in that app's **Secrets**.
- [ ] Wait for the first build to finish (LangChain + ChromaDB make it slower than before, a few minutes).
- [ ] Test the demo document, one PDF, and one unsupported question on the temporary app.
- [ ] If the logs mention SQLite: check `pysqlite3-binary` installed (it is in requirements.txt) and reboot the app.
- [ ] When it works, merge the branch into `main`. The live app at piyush-docuquery.streamlit.app redeploys automatically.
- [ ] If the live app does not pick up the new packages, use **Manage app → Reboot app**.
- [ ] Delete the temporary app.
- [ ] Update the resume and portfolio project line: Python, LangChain, ChromaDB, Gemini API, Streamlit.
