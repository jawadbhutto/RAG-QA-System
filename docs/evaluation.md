# Evaluation

## Dataset

`data/eval/eval_dataset.json` — 30 hand-written questions across the
three source documents, fact-checked against real published content
(not generated from guesses):

| Source | Answerable | Unanswerable | Partially answerable | Total |
|---|---|---|---|---|
| *Artificial Intelligence: A Modern Approach* (2nd Ed., Russell & Norvig) | 6 | 3 | 2 | 11 |
| *Attention Is All You Need* (Vaswani et al.) | 5 | 3 | 2 | 10 |
| *An Overview of Artificial Intelligence Ethics* (Huang et al.) | 4 | 4 | 1 | 9 |
| **Total** | **15** | **10** | **5** | **30** |

Each entry follows the schema:
```json
{
  "question": "",
  "expected_answer": "",
  "expected_source": "",
  "question_type": "answerable | unanswerable | partially_answerable"
}
```

- **Answerable** — the fact is directly stated in one of the three documents.
- **Unanswerable** — genuinely absent from all three (either off-topic
  entirely, e.g. "who won the World Cup", or about a real thing these
  specific documents don't cover, e.g. GPT-4/BERT, which postdate or
  fall outside these documents' scope).
- **Partially answerable** — requires combining two documents that only
  each cover half the question, or asks something the documents can't
  fully confirm either way. A good system should give an honest partial
  answer or express uncertainty here, not confidently fill the gap.

## Running the evaluation

Requires your documents already indexed and a working `.env`
(`GROQ_API_KEY` set):

```bash
python -m src.pipeline          # index the 3 PDFs first, if not already done
python -m src.evaluate          # runs all 30 questions, writes results
```

Outputs:
- `results/evaluation_results.md` — the requested table, plus a summary
  (answer accuracy, source accuracy, hallucination rate, correct
  refusal/answer rate)
- `results/evaluation_results.csv` — same data, for spreadsheet analysis

## How each column is determined

| Column | How it's computed |
|---|---|
| Retrieved Source | Read directly from `QAResult.sources` — the real chunks used |
| Source Correct? | `src/evaluate.py::_source_matches()` — normalizes filenames vs. the expected-source label and checks for a match; for unanswerable questions, correct means *no* source was retrieved |
| Answer Correct? | **LLM-as-judge** — the same configured LLM compares the actual answer to `expected_answer` and the retrieved context, and returns a verdict |
| Hallucination? | Same LLM-judge call — flags any specific fact in the answer not backed by the retrieved context; an honest refusal is never counted as a hallucination |
| Used Context / Refused Correctly | Read directly from `QAResult.found_answer` against what the question type expects |

## A note on the LLM-as-judge columns

"Answer Correct?" and "Hallucination?" are judged by prompting an LLM
to compare the system's answer against the expected answer and the
retrieved context — this is standard practice for RAG evaluation
(exact string matching doesn't work for natural language), but it is
**not infallible**. Treat the automated verdicts as a strong first
pass, not ground truth:

- Spot-check the `Notes` column (the judge's one-sentence reasoning)
  for a sample of rows, especially any marked incorrect or hallucinated.
- The 5 partially-answerable questions are the hardest for the judge to
  score consistently — read those rows manually regardless of the
  automated verdict.
- If a row's Notes say "JUDGE PARSE FAILED — review manually", the
  judge's raw response wasn't valid JSON; that row needs a manual look.

## What this evaluation does and doesn't cover

It measures end-to-end behavior (retrieval + generation + refusal
logic) against a fixed, known-answer dataset. It does **not** measure:
- retrieval quality independent of generation (e.g. precision/recall
  of the raw top-K chunks before the LLM sees them)
- performance under conversational follow-ups (every question here is
  asked independently, with fresh memory)
- consistency across repeated runs (LLM outputs vary run to run,
  especially if `LLM_TEMPERATURE` isn't `0.0`)

Re-running `python -m src.evaluate` periodically as you tune
`CHUNK_SIZE`, `TOP_K`, `RELEVANCE_THRESHOLD`, or swap models is the
intended workflow — compare the summary stats across runs to see
whether a change actually helped.
