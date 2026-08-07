import json

from src.evaluate import EvalRow, _load_dataset, _retrieved_sources_str, _source_matches, _write_csv, _write_markdown
from src.qa_chain import QAResult, Source


def test_dataset_file_loads_and_has_30_questions():
    rows = _load_dataset("data/eval/eval_dataset.json")
    assert len(rows) == 30


def test_dataset_has_required_fields_and_valid_types():
    rows = _load_dataset("data/eval/eval_dataset.json")
    valid_types = {"answerable", "unanswerable", "partially_answerable"}
    for row in rows:
        assert set(row.keys()) == {"question", "expected_answer", "expected_source", "question_type"}
        assert row["question_type"] in valid_types
        assert row["question"].strip()
        assert row["expected_answer"].strip()


def test_dataset_type_distribution_matches_spec():
    rows = _load_dataset("data/eval/eval_dataset.json")
    counts = {}
    for row in rows:
        counts[row["question_type"]] = counts.get(row["question_type"], 0) + 1
    assert counts["answerable"] == 15
    assert counts["unanswerable"] == 10
    assert counts["partially_answerable"] == 5


def _fake_result_with_sources():
    return QAResult(
        answer="The base model uses 8 attention heads.",
        sources=[
            Source(document="Attention_Is_All_You_Need.pdf", page=4, chunk_number=9, similarity=0.82, text="..."),
        ],
        found_answer=True,
        standalone_question="How many attention heads?",
    )


def _fake_result_no_sources():
    return QAResult(
        answer="I could not find this information in the documents.",
        sources=[],
        found_answer=False,
        standalone_question="What is the price of Nvidia stock?",
    )


def test_source_matches_true_when_filename_referenced():
    result = _fake_result_with_sources()
    assert _source_matches("Attention Is All You Need, Section 3.2.2", result) is True


def test_source_matches_false_when_wrong_document():
    result = _fake_result_with_sources()
    assert _source_matches("An Overview of Artificial Intelligence Ethics", result) is False


def test_source_matches_none_expected_and_none_retrieved_is_correct():
    result = _fake_result_no_sources()
    assert _source_matches("None", result) is True


def test_source_matches_none_expected_but_sources_retrieved_is_incorrect():
    result = _fake_result_with_sources()
    assert _source_matches("None", result) is False


def test_retrieved_sources_str_formats_and_dedupes():
    result = _fake_result_with_sources()
    text = _retrieved_sources_str(result)
    assert "Attention_Is_All_You_Need.pdf" in text
    assert "p.4" in text
    assert "chunk 9" in text


def test_retrieved_sources_str_empty_is_none():
    result = _fake_result_no_sources()
    assert _retrieved_sources_str(result) == "None"


def test_write_markdown_and_csv_produce_files(tmp_path):
    rows = [
        EvalRow(
            question="What is AI?",
            question_type="answerable",
            expected_source="AIMA Ch.1",
            retrieved_source="AIMA.pdf (p.1, chunk 1)",
            answer_correct="Yes",
            source_correct="Yes",
            hallucination="No",
            used_context="Yes",
            refused_correctly="Yes",
            notes="Matches expected answer.",
        ),
        EvalRow(
            question="Who won the World Cup?",
            question_type="unanswerable",
            expected_source="None",
            retrieved_source="None",
            answer_correct="Yes",
            source_correct="Yes",
            hallucination="No",
            used_context="No",
            refused_correctly="Yes",
            notes="Correctly refused.",
        ),
    ]

    md_path = tmp_path / "results.md"
    csv_path = tmp_path / "results.csv"
    _write_markdown(rows, md_path)
    _write_csv(rows, csv_path)

    md_text = md_path.read_text(encoding="utf-8")
    assert "| Question | Expected Source" in md_text
    assert "What is AI?" in md_text
    assert "## Summary" in md_text
    assert "Answer accuracy: 100.0%" in md_text

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "What is AI?" in csv_text
    assert "Who won the World Cup?" in csv_text
