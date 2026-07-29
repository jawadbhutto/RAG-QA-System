from langchain_core.messages import AIMessage, HumanMessage

from src.memory import ConversationMemory


def test_new_memory_is_empty():
    memory = ConversationMemory()
    assert memory.is_empty()
    assert memory.as_messages() == []
    assert memory.as_text() == ""


def test_add_turn_stores_question_and_answer():
    memory = ConversationMemory()
    memory.add_turn("What is AI?", "AI is the simulation of human intelligence.")

    assert not memory.is_empty()
    assert len(memory.turns) == 1
    assert memory.turns[0].question == "What is AI?"
    assert memory.turns[0].answer == "AI is the simulation of human intelligence."


def test_as_messages_alternates_human_and_ai():
    memory = ConversationMemory()
    memory.add_turn("Q1", "A1")
    memory.add_turn("Q2", "A2")

    messages = memory.as_messages()

    assert len(messages) == 4
    assert isinstance(messages[0], HumanMessage) and messages[0].content == "Q1"
    assert isinstance(messages[1], AIMessage) and messages[1].content == "A1"
    assert isinstance(messages[2], HumanMessage) and messages[2].content == "Q2"
    assert isinstance(messages[3], AIMessage) and messages[3].content == "A2"


def test_memory_trims_to_max_turns():
    memory = ConversationMemory(max_turns=2)
    memory.add_turn("Q1", "A1")
    memory.add_turn("Q2", "A2")
    memory.add_turn("Q3", "A3")

    assert len(memory.turns) == 2
    assert memory.turns[0].question == "Q2"
    assert memory.turns[1].question == "Q3"


def test_clear_empties_history():
    memory = ConversationMemory()
    memory.add_turn("Q1", "A1")
    memory.clear()

    assert memory.is_empty()
