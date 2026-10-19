import hashlib
import math
import re

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import api
from rag_engine import GeminiRAG, get_chroma_client


class WordEmbeddings(Embeddings):
    def _embed(self, text):
        vector = [0.0] * 256
        for word in re.findall(r"[a-z0-9_]+", text.lower()):
            if len(word) > 2:
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector] if any(vector) else [1e-3] * 256

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("DOCUQUERY_API_TOKEN", raising=False)
    monkeypatch.setattr(api, "store", api.IndexStore(ttl_seconds=3600, max_indexes=20))
    rag = GeminiRAG(
        embeddings=WordEmbeddings(),
        llm=FakeListChatModel(responses=["Failed batches are rerun after approval [Source 1]."]),
    )
    api.app.dependency_overrides[api.get_rag] = lambda: rag
    with TestClient(api.app) as test_client:
        yield test_client
    api.app.dependency_overrides.clear()
    for index_id in list(api.store._entries):
        api.store.delete(index_id)


def test_health_reports_configuration(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["gemini_configured"] is True and body["auth_required"] is False


def test_demo_index_and_query_return_cited_sources(client):
    created = client.post("/indexes/demo")
    assert created.status_code == 201
    info = created.json()
    assert info["chunks"] > 10 and len(info["documents"]) == 4 and info["chunking"] == "sections"

    response = client.post(f"/indexes/{info['index_id']}/query", json={"question": "What happens when a pipeline activity fails?", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Failed batches")
    assert [s["rank"] for s in body["sources"]] == [1, 2, 3]
    assert body["sources"][0]["score"] >= body["sources"][1]["score"]
    assert any("Failure Handling" in (s["section"] or "") for s in body["sources"])
    assert body["retrieval_ms"] >= 0 and body["generation_ms"] >= 0


def test_upload_index_with_fixed_chunking(client):
    files = [
        ("files", ("notes.md", b"# Notes\n\n## Access\nRaise a request in the portal.", "text/markdown")),
        ("files", ("runbook.txt", b"Retry the failed batch after approval.", "text/plain")),
    ]
    response = client.post("/indexes", files=files, data={"chunking": "fixed", "chunk_size": "500", "overlap": "50"})
    assert response.status_code == 201, response.text
    info = response.json()
    assert info["documents"] == ["notes.md", "runbook.txt"] and info["chunking"] == "fixed" and info["chunk_size"] == 500
    assert client.get(f"/indexes/{info['index_id']}").json()["chunks"] == info["chunks"]


@pytest.mark.parametrize(
    "files,expected",
    [
        ([("files", ("image.png", b"\x89PNG", "image/png"))], 415),
        ([("files", ("empty.txt", b"   ", "text/plain"))], 422),
        ([("files", (f"f{i}.txt", b"text", "text/plain")) for i in range(6)], 413),
    ],
)
def test_upload_validation(client, files, expected):
    assert client.post("/indexes", files=files).status_code == expected


def test_query_validation_and_unknown_index(client):
    index_id = client.post("/indexes/demo").json()["index_id"]
    assert client.post(f"/indexes/{index_id}/query", json={"question": ""}).status_code == 422
    assert client.post(f"/indexes/{index_id}/query", json={"question": "x", "top_k": 50}).status_code == 422
    assert client.post("/indexes/missing/query", json={"question": "x"}).status_code == 404


def test_delete_frees_the_chroma_collection(client):
    index_id = client.post("/indexes/demo").json()["index_id"]
    collection = api.store._entries[index_id].index.collection_name
    assert client.delete(f"/indexes/{index_id}").status_code == 204
    assert client.get(f"/indexes/{index_id}").status_code == 404
    assert collection not in {getattr(c, "name", c) for c in get_chroma_client().list_collections()}


def test_token_is_required_when_configured(client, monkeypatch):
    monkeypatch.setenv("DOCUQUERY_API_TOKEN", "secret-token")
    assert client.get("/health").status_code == 200
    assert client.post("/indexes/demo").status_code == 401
    assert client.post("/indexes/demo", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/indexes/demo", headers={"X-API-Key": "secret-token"}).status_code == 201


def test_missing_gemini_key_returns_503(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DOCUQUERY_API_TOKEN", raising=False)
    api.app.dependency_overrides.clear()
    with TestClient(api.app) as test_client:
        response = test_client.post("/indexes/demo")
    assert response.status_code == 503 and "GEMINI_API_KEY" in response.json()["detail"]


def test_store_expires_and_caps_indexes(client):
    store = api.IndexStore(ttl_seconds=3600, max_indexes=2)
    api.store = store
    ids = [client.post("/indexes/demo").json()["index_id"] for _ in range(3)]
    assert len(store) == 2 and ids[0] not in store._entries  # oldest dropped first

    store.ttl_seconds = 0
    assert client.get(f"/indexes/{ids[2]}").status_code == 404
    assert len(store) == 1
