# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Exercise ingestion against Qdrant's real in-memory implementation."""

import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient, models

from gepa.adapters.generic_rag_adapter.vector_stores.qdrant_store import QdrantVectorStore

pytestmark = pytest.mark.extra


@pytest.fixture
def store():
    client = QdrantClient(":memory:")
    client.create_collection("documents", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    yield QdrantVectorStore(client, "documents")
    client.close()


def contents(store):
    return {doc["metadata"]["doc_id"]: doc["content"] for doc in store.vector_search([1.0, 0.0], k=20)}


def test_distinct_ids_survive_separate_ingestions(store):
    assert store.add_documents([{"content": "first"}], [[1.0, 0.0]], ["a"]) == ["a"]
    assert store.add_documents([{"content": "second"}], [[0.0, 1.0]], ["b"]) == ["b"]
    assert store.get_collection_info()["document_count"] == 2
    assert contents(store) == {"a": "first", "b": "second"}
    assert store.delete_documents(["a"])
    assert contents(store) == {"b": "second"}


def test_explicit_ids_upsert_independently_of_batch_order_and_instance(store):
    ids = ["a", "b", "0", "", "文档", "00000000-0000-0000-0000-000000000000"]
    store.add_documents([{"content": name} for name in ids], [[1.0, 0.0]] * len(ids), ids)
    reopened = QdrantVectorStore(store.client, store.collection_name)
    reopened.add_documents([{"content": "updated b"}], [[0.0, 1.0]], ["b"])
    expected = {name: name for name in ids}
    expected.update(b="updated b")
    assert contents(store) == expected
    reopened.add_documents([{"content": "updated b again"}, {"content": "updated a"}], [[0.0, 1.0]] * 2, ["b", "a"])
    expected.update(a="updated a", b="updated b again")
    assert contents(store) == expected
    assert store.get_collection_info()["document_count"] == len(ids)


def test_generated_ids_are_unique_and_reusable(store):
    first = store.add_documents([{"content": "first"}], [[1.0, 0.0]])
    second = store.add_documents([{"content": "second"}], [[0.0, 1.0]])
    assert len(set(first + second)) == 2
    assert contents(store) == {first[0]: "first", second[0]: "second"}
    store.add_documents([{"content": "updated"}], [[1.0, 0.0]], first)
    assert contents(store) == {first[0]: "updated", second[0]: "second"}
    assert store.delete_documents(first + second)
    assert store.get_collection_info()["document_count"] == 0


def test_new_documents_preserve_legacy_integer_points(store):
    store.client.upsert(
        store.collection_name,
        [models.PointStruct(id=0, vector=[1.0, 0.0], payload={"original_id": "legacy", "content": "old"})],
    )
    store.add_documents([{"content": "new"}], [[0.0, 1.0]], ["new"])
    assert contents(store) == {"legacy": "old", "new": "new"}
    assert store.delete_documents(["legacy"])
    assert contents(store) == {"new": "new"}


@pytest.mark.parametrize("documents,embeddings,ids", [([{}], [], None), ([{}], [[1.0, 0.0]], [])])
def test_length_validation_does_not_write(store, documents, embeddings, ids):
    with pytest.raises(ValueError, match="must match"):
        store.add_documents(documents, embeddings, ids)
    assert store.get_collection_info()["document_count"] == 0


def test_duplicate_ids_in_one_batch_upsert_one_point(store):
    store.add_documents([{"content": "first"}, {"content": "last"}], [[1.0, 0.0]] * 2, ["same", "same"])
    assert store.get_collection_info()["document_count"] == 1
    assert contents(store) == {"same": "last"}


@pytest.mark.parametrize("ids", [None, []])
def test_empty_batch_leaves_existing_documents_unchanged(store, ids):
    store.add_documents([{"content": "keep"}], [[1.0, 0.0]], ["keep"])
    assert store.add_documents([], [], ids) == []
    assert contents(store) == {"keep": "keep"}
