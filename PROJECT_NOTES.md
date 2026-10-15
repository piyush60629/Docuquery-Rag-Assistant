# Interview Notes

## One-line description

A deployed RAG application that answers questions from user-uploaded documents using LangChain for chunking and orchestration, Gemini embeddings stored in a ChromaDB vector store, cosine-similarity retrieval, and grounded generation with citations.

## Why it is believable for my profile

My professional role remains Data Engineer. This project extends my Python and data-processing knowledge into Generative AI by treating unstructured document text as a searchable data pipeline.

## Components I should be able to explain

1. File extraction for PDF, DOCX, TXT, and Markdown.
2. Why long text is split into overlapping chunks, and how `RecursiveCharacterTextSplitter` tries paragraph → line → sentence → word boundaries.
3. What an embedding represents, and why documents and questions use different task types (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY).
4. Why a vector database (ChromaDB) instead of a NumPy array: metadata stored with each vector, HNSW approximate search that scales, and a standard API that could be swapped for a persistent or hosted store.
5. How cosine similarity finds semantically related chunks (Chroma returns cosine distance; similarity = 1 − distance).
6. Why only top matching chunks are sent to the LLM.
7. How the LangChain LCEL chain (prompt | model | parser) and the prompt keep answers grounded and cited.
8. Why each session gets its own Chroma collection (user isolation on a shared server) and how stale collections are cleaned up.
9. Why API keys must be stored in deployment secrets.
10. How the project is evaluated: 35 questions (30 answerable, 5 unanswerable), retrieval Hit@k and MRR, LLM-as-judge faithfulness and correctness, and refusal accuracy. Why evidence phrases instead of chunk IDs (results stay comparable when chunk size changes). Why unanswerable questions matter (they measure hallucination directly).
11. How evaluation is automated: offline unit tests run on every push (GitHub Actions), and the Gemini-based evaluation runs on demand from the Actions tab and commits its results.
12. Current limitations: no OCR, in-memory vector store (resets with the session), small evaluation corpus, judge model from the same family as the generator.

## Honest answer about Coforge

This was created independently as a self-learning and portfolio project. It was not developed or deployed for Coforge and does not use company data.
