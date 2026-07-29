import src.qa_chain as qa_chain_module
from src.memory import ConversationMemory
from src.retriever import RetrievedChunk


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Stand-in for a real chat model: returns pre-programmed responses in order."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return FakeMessage(self._responses.pop(0))


def _sample_chunks():
    return [
        RetrievedChunk(
            text="AI is the simulation of human intelligence by machines.",
            source="AI_Ethics.pdf",
            page=1,
            chunk_number=13,
            similarity=0.87,
        ),
        RetrievedChunk(
            text="Machine learning is a core subfield of AI.",
            source="AI_Ethics.pdf",
            page=2,
            chunk_number=14,
            similarity=0.61,
        ),
    ]


def test_no_chunks_retrieved_returns_refusal_without_calling_llm(monkeypatch):
    monkeypatch.setattr(qa_chain_module, "retrieve", lambda q, k=None, threshold=None: [])
    fake_llm = FakeLLM([])  # should never be called
    monkeypatch.setattr(qa_chain_module, "get_llm", lambda: fake_llm)

    result = qa_chain_module.answer_question("Who won the FIFA World Cup?")

    assert result.found_answer is False
    assert result.answer == qa_chain_module.settings.no_answer_message
    assert result.sources == []
    assert fake_llm.calls == []  # never invoked


def test_relevant_chunks_produce_grounded_answer_with_sources(monkeypatch):
    monkeypatch.setattr(
        qa_chain_module, "retrieve", lambda q, k=None, threshold=None: _sample_chunks()
    )
    fake_llm = FakeLLM(["AI simulates human intelligence in machines."])
    monkeypatch.setattr(qa_chain_module, "get_llm", lambda: fake_llm)

    memory = ConversationMemory()
    result = qa_chain_module.answer_question("What is AI?", memory=memory)

    assert result.found_answer is True
    assert result.answer == "AI simulates human intelligence in machines."
    assert len(result.sources) == 2
    assert result.sources[0].document == "AI_Ethics.pdf"
    assert result.sources[0].chunk_number == 13
    assert result.sources[0].page == 1
    # memory should now contain this turn
    assert memory.turns[-1].question == "What is AI?"
    assert memory.turns[-1].answer == result.answer


def test_llm_refusal_on_retrieved_but_insufficient_context(monkeypatch):
    monkeypatch.setattr(
        qa_chain_module, "retrieve", lambda q, k=None, threshold=None: _sample_chunks()
    )
    fake_llm = FakeLLM([qa_chain_module.settings.no_answer_message])
    monkeypatch.setattr(qa_chain_module, "get_llm", lambda: fake_llm)

    result = qa_chain_module.answer_question("What is the airspeed of a swallow?")

    assert result.found_answer is False
    assert result.answer == qa_chain_module.settings.no_answer_message
    assert result.sources == []


def test_contextualize_not_called_when_memory_empty(monkeypatch):
    monkeypatch.setattr(
        qa_chain_module, "retrieve", lambda q, k=None, threshold=None: _sample_chunks()
    )
    fake_llm = FakeLLM(["Direct answer."])
    monkeypatch.setattr(qa_chain_module, "get_llm", lambda: fake_llm)

    memory = ConversationMemory()  # empty
    result = qa_chain_module.answer_question("What is AI?", memory=memory)

    # Only one LLM call should happen (the answer call), not a contextualize call.
    assert len(fake_llm.calls) == 1
    assert result.standalone_question == "What is AI?"


def test_empty_question_returns_prompt_without_retrieval(monkeypatch):
    called = {"retrieve": False}

    def fake_retrieve(q, k=None, threshold=None):
        called["retrieve"] = True
        return []

    monkeypatch.setattr(qa_chain_module, "retrieve", fake_retrieve)

    result = qa_chain_module.answer_question("   ")

    assert called["retrieve"] is False
    assert result.found_answer is False
