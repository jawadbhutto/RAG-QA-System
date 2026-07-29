from langchain_core.documents import Document

import src.retriever as retriever_module


def _fake_results():
    """Mimic Chroma's similarity_search_with_score output: (Document, distance)."""
    return [
        (
            Document(
                page_content="AI is the simulation of human intelligence.",
                metadata={"source": "a.pdf", "page": 0, "chunk_number": 1},
            ),
            0.30,
        ),
        (
            Document(
                page_content="Machine learning is a subset of AI.",
                metadata={"source": "a.pdf", "page": 1, "chunk_number": 2},
            ),
            0.45,
        ),
        (
            Document(
                page_content="The FIFA World Cup is a football tournament.",
                metadata={"source": "b.pdf", "page": 0, "chunk_number": 1},
            ),
            0.75,
        ),
    ]


def test_retrieve_filters_out_low_relevance_results(monkeypatch):
    monkeypatch.setattr(retriever_module, "similarity_search", lambda query, k=None: _fake_results())

    results = retriever_module.retrieve("What is AI?", k=3, threshold=0.5)

    assert len(results) == 2
    assert all(chunk.similarity >= 0.5 for chunk in results)
    assert results[0].source == "a.pdf"
    assert results[0].chunk_number == 1


def test_retrieve_returns_empty_list_when_nothing_relevant(monkeypatch):
    monkeypatch.setattr(
        retriever_module,
        "similarity_search",
        lambda query, k=None: [(Document(page_content="irrelevant", metadata={}), 0.9)],
    )

    results = retriever_module.retrieve("Who won the FIFA World Cup?", threshold=0.5)

    assert results == []


def test_retrieve_empty_question_returns_empty_list():
    assert retriever_module.retrieve("   ") == []
