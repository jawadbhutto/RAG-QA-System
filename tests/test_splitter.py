from langchain_core.documents import Document

from src.splitter import split_documents


def test_split_documents_empty_list_returns_empty():
    assert split_documents([]) == []


def test_split_documents_produces_chunks_with_clean_metadata():
    long_text = "Artificial intelligence is transforming many industries. " * 40
    docs = [
        Document(
            page_content=long_text,
            metadata={"source": "/full/path/to/AI_Ethics.pdf", "page": 0},
        )
    ]

    chunks = split_documents(docs)

    assert len(chunks) > 1
    for i, chunk in enumerate(chunks, start=1):
        assert chunk.metadata["source"] == "AI_Ethics.pdf"
        assert chunk.metadata["chunk_number"] == i
        assert chunk.metadata["document_type"] == "pdf"
        assert len(chunk.page_content) > 0
