"""
evaluate.py
-----------
Runs data/eval/eval_dataset.json through the real, live QA system
(retrieval + LLM) and produces the results table requested for manual
review:

    Question | Expected Source | Retrieved Source | Answer Correct? |
    Source Correct? | Hallucination? | Notes

This is NOT a unit test — it makes real calls to your embedding model
and your configured LLM (Groq), so it requires:
    1. Your documents already indexed: `python -m src.pipeline`
    2. A valid .env with GROQ_API_KEY set

Correctness and hallucination are judged with an LLM-as-judge pass
(reusing the same chat model configured in src/llm.py) rather than
exact string matching, since answers are natural language. LLM-judged
columns are inherently approximate — always spot-check a sample of the
Notes column yourself before trusting the aggregate numbers.

Usage:
    python -m src.evaluate
    python -m src.evaluate --dataset data/eval/eval_dataset.json --out results/evaluation_results
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import settings
from src.llm import get_llm
from src.memory import ConversationMemory
from src.qa_chain import QAResult, answer_question
from src.utils import get_logger

logger = get_logger(__name__)

_JUDGE_PROMPT = """You are grading a RAG system's answer against an expected answer.

Question: {question}
Question type: {question_type}
Expected answer: {expected_answer}
System's actual answer: {actual_answer}
Retrieved context the system was given: {context}

Judge two things:
1. "correct": Does the system's actual answer convey the same substantive information as \
the expected answer? For "unanswerable" questions, correct=true means the system correctly \
refused (its answer should be the refusal message or equivalent). For "partially_answerable" \
questions, correct=true means the system gave an honest partial answer or appropriately \
expressed uncertainty, WITHOUT confidently inventing the missing half.
2. "hallucinated": Does the actual answer state any specific fact (a number, name, date, \
quote, or claim) that is NOT supported by the retrieved context shown above? An honest \
refusal or an honest "I'm not certain" is never a hallucination, even if it doesn't fully \
answer the question.

Respond with ONLY a JSON object, no other text, in this exact form:
{{"correct": true or false, "hallucinated": true or false, "reasoning": "one short sentence"}}
"""


@dataclass
class EvalRow:
    question: str
    question_type: str
    expected_source: str
    retrieved_source: str
    answer_correct: str
    source_correct: str
    hallucination: str
    used_context: str
    refused_correctly: str
    notes: str


def _load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _retrieved_sources_str(result: QAResult) -> str:
    if not result.sources:
        return "None"
    # de-duplicate while preserving order
    seen = []
    for s in result.sources:
        label = f"{s.document} (p.{s.page}, chunk {s.chunk_number})"
        if label not in seen:
            seen.append(label)
    return "; ".join(seen)

def _normalize(text: str) -> str:
    """Lowercase and collapse underscores/hyphens/extra spaces to single spaces,
    so 'Attention_Is_All_You_Need.pdf' lines up with 'Attention Is All You Need'."""
    text = re.sub(r"\.pdf$", "", text.strip().lower())
    text = re.sub(r"[_\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _source_matches(expected_source: str, result: QAResult) -> bool:
    """Loose match: does any retrieved chunk's filename appear referenced by
    the expected_source string? Falls back to 'no sources expected/retrieved'
    agreement for unanswerable questions."""
    if expected_source.strip().lower() == "none":
        return not result.sources

    if not result.sources:
        return False

    expected_norm = _normalize(expected_source)
    return any(_normalize(s.document) in expected_norm for s in result.sources)


def _judge(item: dict, result: QAResult) -> dict:
    """LLM-as-judge: returns {"correct": bool, "hallucinated": bool, "reasoning": str}."""
    context = "\n\n".join(f"[{s.document} p.{s.page}] {s.text}" for s in result.sources) or "(none retrieved)"
    prompt = _JUDGE_PROMPT.format(
        question=item["question"],
        question_type=item["question_type"],
        expected_answer=item["expected_answer"],
        actual_answer=result.answer,
        context=context,
    )
    llm = get_llm()
    raw = llm.invoke([SystemMessage(content=prompt)]).content.strip()
    raw = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        verdict = json.loads(raw)
        return {
            "correct": bool(verdict.get("correct", False)),
            "hallucinated": bool(verdict.get("hallucinated", False)),
            "reasoning": str(verdict.get("reasoning", "")),
        }
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Judge response was not valid JSON, marking for manual review: %r", raw)
        return {"correct": False, "hallucinated": False, "reasoning": "JUDGE PARSE FAILED — review manually"}


def run_evaluation(dataset_path: Path) -> list[EvalRow]:
    dataset = _load_dataset(dataset_path)
    rows: list[EvalRow] = []

    for i, item in enumerate(dataset, start=1):
        logger.info("[%d/%d] %s", i, len(dataset), item["question"])

        # Each question is independent — fresh memory per question, not a
        # multi-turn conversation, so results aren't affected by ordering.
        result = answer_question(item["question"], memory=ConversationMemory())

        source_correct = _source_matches(item["expected_source"], result)
        used_context = bool(result.sources)
        should_refuse = item["question_type"] == "unanswerable"
        refused_correctly = (
            (result.found_answer is False) if should_refuse else (result.found_answer is True)
        )

        verdict = _judge(item, result)

        rows.append(
            EvalRow(
                question=item["question"],
                question_type=item["question_type"],
                expected_source=item["expected_source"],
                retrieved_source=_retrieved_sources_str(result),
                answer_correct="Yes" if verdict["correct"] else "No",
                source_correct="Yes" if source_correct else "No",
                hallucination="Yes" if verdict["hallucinated"] else "No",
                used_context="Yes" if used_context else "No",
                refused_correctly="Yes" if refused_correctly else "No",
                notes=verdict["reasoning"],
            )
        )

    return rows


def _write_markdown(rows: list[EvalRow], out_path: Path) -> None:
    lines = [
        "| Question | Expected Source | Retrieved Source | Answer Correct? | Source Correct? | Hallucination? | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        q = r.question.replace("|", "\\|")
        exp = r.expected_source.replace("|", "\\|")
        ret = r.retrieved_source.replace("|", "\\|")
        notes = r.notes.replace("|", "\\|")
        lines.append(f"| {q} | {exp} | {ret} | {r.answer_correct} | {r.source_correct} | {r.hallucination} | {notes} |")

    total = len(rows)
    if total:
        answer_acc = sum(r.answer_correct == "Yes" for r in rows) / total * 100
        source_acc = sum(r.source_correct == "Yes" for r in rows) / total * 100
        halluc_rate = sum(r.hallucination == "Yes" for r in rows) / total * 100
        refusal_acc = sum(r.refused_correctly == "Yes" for r in rows) / total * 100

        summary = [
            "",
            "## Summary",
            "",
            f"- Questions evaluated: {total}",
            f"- Answer accuracy: {answer_acc:.1f}%",
            f"- Source accuracy: {source_acc:.1f}%",
            f"- Hallucination rate: {halluc_rate:.1f}%",
            f"- Correct refusal/answer behavior: {refusal_acc:.1f}%",
        ]
        lines = lines + summary

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(rows: list[EvalRow], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG evaluation dataset against the live system.")
    parser.add_argument("--dataset", default="data/eval/eval_dataset.json")
    parser.add_argument("--out", default="results/evaluation_results")
    args = parser.parse_args()

    settings.ensure_directories()
    dataset_path = Path(args.dataset)
    rows = run_evaluation(dataset_path)

    md_path = Path(f"{args.out}.md")
    csv_path = Path(f"{args.out}.csv")
    _write_markdown(rows, md_path)
    _write_csv(rows, csv_path)

    print(f"Wrote {md_path} and {csv_path}")
