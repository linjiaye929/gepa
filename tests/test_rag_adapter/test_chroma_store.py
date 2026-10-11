# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from uuid import uuid4

import pytest

from gepa.adapters.generic_rag_adapter.vector_stores.chroma_store import ChromaVectorStore


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({}, {}),
        ({"tenant": "a"}, {"tenant": {"$eq": "a"}}),
        ({"rank": {"$gt": 1}}, {"rank": {"$gt": 1}}),
        ({"tenant": {"$in": ["a", "b"]}}, {"tenant": {"$in": ["a", "b"]}}),
        ({"tenant": {"$nin": ["b"]}}, {"tenant": {"$nin": ["b"]}}),
        (
            {"tenant": "a", "status": "active"},
            {"$and": [{"tenant": {"$eq": "a"}}, {"status": {"$eq": "active"}}]},
        ),
        (
            {"tenant": {"$in": ["a", "b"]}, "rank": {"$gt": 1}},
            {"$and": [{"tenant": {"$in": ["a", "b"]}}, {"rank": {"$gt": 1}}]},
        ),
    ],
)
def test_convert_filters(filters, expected):
    store = ChromaVectorStore.__new__(ChromaVectorStore)

    assert store._convert_filters(filters) == expected


@pytest.fixture
def chroma_store():
    chromadb = pytest.importorskip("chromadb")

    class ConstantEmbeddingFunction(chromadb.EmbeddingFunction):
        def __init__(self):
            pass

        def __call__(self, input):
            return [[1.0, 0.0] for _ in input]

        def get_config(self):
            return {}

        @staticmethod
        def build_from_config(config):
            return ConstantEmbeddingFunction()

        @staticmethod
        def name():
            return "test-constant-embedding"

    store = ChromaVectorStore.create_memory(
        collection_name=f"test-filters-{uuid4().hex}",
        embedding_function=ConstantEmbeddingFunction(),
    )
    store.collection.add(
        ids=["a-active", "a-inactive", "b-active"],
        documents=["a-active", "a-inactive", "b-active"],
        embeddings=[[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
        metadatas=[
            {"tenant": "a", "status": "active", "rank": 2},
            {"tenant": "a", "status": "inactive", "rank": 1},
            {"tenant": "b", "status": "active", "rank": 3},
        ],
    )
    try:
        yield store
    finally:
        store.client.delete_collection(store.collection_name)


@pytest.mark.extra
@pytest.mark.parametrize("search_method", ["vector_search", "similarity_search"])
@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        (None, {"a-active", "a-inactive", "b-active"}),
        ({}, {"a-active", "a-inactive", "b-active"}),
        ({"tenant": "a"}, {"a-active", "a-inactive"}),
        ({"rank": {"$gt": 1}}, {"a-active", "b-active"}),
        ({"tenant": {"$in": ["a"]}}, {"a-active", "a-inactive"}),
        ({"tenant": {"$nin": ["b"]}}, {"a-active", "a-inactive"}),
        ({"tenant": "a", "status": "active"}, {"a-active"}),
        ({"tenant": "b", "status": "inactive"}, set()),
        ({"tenant": {"$in": ["a"]}, "rank": {"$gt": 1}}, {"a-active"}),
    ],
)
def test_search_metadata_filters(chroma_store, search_method, filters, expected):
    query = [1.0, 0.0] if search_method == "vector_search" else "query"

    results = getattr(chroma_store, search_method)(query, k=3, filters=filters)

    assert {result["content"] for result in results} == expected
