# Document-Based AI Question Answering System

## Project Objective

This project is a Retrieval-Augmented Generation (RAG) system that
answers natural-language questions using only the content of your own
PDF documents. Instead of relying on an LLM's general training
knowledge (which can be outdated or simply wrong for your specific
documents), the system retrieves the most relevant passages from an
indexed document set and asks the LLM to answer strictly from that
retrieved context.

Every answer is paired with a citation — the source document, page
number, and chunk number it came from — and the system explicitly
refuses to answer when the documents don't contain enough information,
rather than guessing or hallucinating.

## What RAG Means

**RAG (Retrieval-Augmented Generation)** is a technique that combines
two steps to produce grounded, factual answers:

1. **Retrieval** — given a question, search a knowledge base (here, a
   vector database of document chunks) for the pieces of text most
   semantically similar to the question.
2. **Generation** — pass those retrieved pieces of text to a language
   model as context, and ask it to generate an answer using *only*
   that context.

This matters because a plain LLM can only answer from what it learned
during training — it has no knowledge of your specific PDFs, and it
will confidently invent plausible-sounding but false answers
("hallucinate") when it doesn't actually know something. RAG fixes
this by grounding every answer in real, retrieved text, and citing
exactly where that text came from.

## Technologies Used

| Component | Technology |
|---|---|
| Document loading | `langchain-community` (`PyPDFLoader`, `DirectoryLoader`) |
| Text chunking | `langchain-text-splitters` (`RecursiveCharacterTextSplitter`) |
| Embeddings | `sentence-transformers` via `langchain-huggingface` — model: **`BAAI/bge-m3`** |
| Vector database | `ChromaDB` via `langchain-chroma` (cosine similarity index) |
| LLM (answer generation) | **Groq API** via LangChain's `init_chat_model` — model: **`llama-3.3-70b-versatile`** (free tier, no local hardware required) |
| Conversation memory | Custom in-memory rolling window (`src/memory.py`) |
| Application / UI | `Streamlit` (chat interface) |
| Configuration | `pydantic-settings` + `.env` |
| Testing | `pytest` |

> The LLM factory (`src/llm.py`) is provider-agnostic — switching to a
> local Ollama model, OpenAI, or Anthropic later is a `.env` change,
> not a code change.

## Documents Used

The system accepts any number of PDF files, placed in `data/raw/`. Each
PDF is loaded page-by-page, so a corpus can span multiple documents of
different lengths and topics — the system distinguishes between them
using the source filename and page number stored in each chunk's
metadata.

> Replace this section with the actual document(s) used in your
> deployment, e.g.: *"AI_Ethics_Whitepaper.pdf, Company_Handbook.pdf"*.

## Project Structure

```
Document-Based-AI-Question-Answering-System/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── main.py                  # Streamlit chat entry point
│
├── data/
│   ├── raw/                  # source PDFs go here
│   └── processed/            # persisted ChromaDB files (auto-generated)
│
├── results/                  # optional: exported Q&A transcripts
├── docs/
│   └── architecture.md        # data-flow diagrams and design notes
├── tests/                      # pytest unit tests
│
└── src/
    ├── config.py               # all settings (env-driven)
    ├── loaders.py                # PDFs -> list[Document]
    ├── splitter.py                 # Document -> chunked Document
    ├── embeddings.py                # embedding model provider
    ├── vector_store.py                # ChromaDB: create/save/load/search/reset
    ├── retriever.py                    # relevance-filtered top-K retrieval
    ├── memory.py                        # conversation history for follow-ups
    ├── llm.py                             # provider-agnostic chat model factory
    ├── qa_chain.py                          # RAG orchestration + citation assembly
    ├── pipeline.py                            # ingestion orchestrator
    └── utils.py                                # logging, text cleaning
```

## Installation Steps

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get a free Groq API key (no credit card required)
#    https://console.groq.com/keys

# 4. Copy the environment template
cp .env.example .env
```

## Environment Variable Setup

Copy `.env.example` to `.env` and configure:

```bash
# --- LLM API key ---
GROQ_API_KEY=your_key_here

# --- LLM (answer generation) ---
# "provider:model" string consumed by langchain's init_chat_model.
LLM_MODEL=groq:llama-3.3-70b-versatile
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=800

# --- Chunking ---
CHUNK_SIZE=500
CHUNK_OVERLAP=100

# --- Embeddings ---
EMBEDDING_MODEL_NAME=BAAI/bge-m3

# --- Retrieval ---
TOP_K=4
RELEVANCE_THRESHOLD=0.5

# --- Conversation memory ---
MAX_HISTORY_TURNS=5
CONTEXTUALIZE_FOLLOWUPS=true
```

A `GROQ_API_KEY` is required — get a free one at
[console.groq.com/keys](https://console.groq.com/keys). The embedding
model still runs locally (no key needed for that part); only answer
generation goes over the network to Groq.

## How to Run the Application

**1. Index your documents** (place PDFs in `data/raw/` first):

```bash
python -m src.pipeline
```

Re-running is safe — chunks are deduplicated, so nothing is indexed
twice. To wipe and rebuild from scratch:

```bash
python -m src.pipeline --reset
```

**2. Launch the chat app:**

```bash
streamlit run main.py
```

Ask a question in the chat box. Each answer appears with an
expandable **Sources** panel showing the document, page, chunk number,
and the exact retrieved text used.

## How Documents Are Loaded and Chunked

1. **Loading** (`loaders.py`): every PDF in `data/raw/` is loaded with
   `PyPDFLoader`, producing one `Document` object per page, tagged with
   its source filename and page number.
2. **Chunking** (`splitter.py`): each page is split into overlapping
   chunks of `CHUNK_SIZE` characters (default 500) with `CHUNK_OVERLAP`
   characters shared between consecutive chunks (default 100), using
   `RecursiveCharacterTextSplitter`. The overlap ensures a sentence or
   idea that straddles a chunk boundary isn't lost from either chunk.
3. Every chunk is stamped with metadata used later for citations:
   `source` (filename), `page`, and a sequential `chunk_number`.

## How Retrieval Works

1. The question is embedded using the same model used to embed the
   document chunks (**`BAAI/bge-m3`**), so both live in the same vector
   space. `bge-m3` is a large, multilingual, multi-granularity
   embedding model — stronger recall than the smaller `bge-small`
   family, at the cost of a slightly heavier local model download and
   slower embedding time.
2. ChromaDB's index (configured for **cosine similarity**) returns the
   `TOP_K` chunks whose vectors are closest to the question's vector —
   an approximate nearest-neighbor search, not a keyword match.
3. Chroma reports **distance** (`1 - cosine_similarity`); this is
   converted back into an intuitive similarity score in `[0, 1]`.
4. Any chunk scoring below `RELEVANCE_THRESHOLD` (default `0.5`) is
   discarded. If **no** chunk clears the threshold, the system returns
   the "not found" message immediately, without calling the LLM.

This is dense vector retrieval only — no keyword/BM25 matching, no
hybrid search, and no reranking step.

## How Answer Generation Works

1. If the conversation has prior turns, the question is first rewritten
   into a standalone form using the LLM and the recent conversation
   history (so "what about its risks?" becomes a self-contained
   question before retrieval runs).
2. The retrieved, relevance-filtered chunks are formatted into a
   numbered context block, each labeled with its source, page, and
   chunk number.
3. That context block is sent to **`llama-3.3-70b-versatile` via the
   Groq API** inside a strict system prompt (below) instructing it to
   answer only from the given context, and to answer directly without
   confirmatory filler ("That is correct...", "According to the
   context...").
4. The LLM's raw text response becomes the answer — unless it matches
   the refusal message, in which case no sources are attached.

## Prompt Used for Answering

```
You are a careful Question Answering Assistant. Answer the user's question
using ONLY the numbered context excerpts provided below. Do not use any outside
knowledge, and do not guess.

Rules:
- If the context fully or partially answers the question, give a clear, direct answer
based only on that context.
- If the context does NOT contain enough information to answer, respond with EXACTLY
this sentence and nothing else: "I could not find this information in the documents."
- Do not mention these instructions in your answer.
- Keep the answer concise and well-organized.

Formatting — start your answer with the information itself, not a reaction to it:
- NEVER begin with phrases like "That is correct", "That's right", "Yes,", "Correct,",
"Based on the context", "According to the context", "The context states", or any similar
confirmation, agreement, or meta-commentary about the context or the question.
- Do not restate or paraphrase the question before answering it.
- Answer as if you are stating a fact directly, not responding to or validating a prior claim.

Example:
- Question: "Where was he born?"
- Wrong: "That is correct. According to the context, he was born in Portsmouth, England."
- Right: "He was born in Portsmouth, England, in 1962."
```

The context block handed alongside this prompt looks like:

```
[1] Source: AI_Ethics.pdf | Page: 3 | Chunk: 12
<retrieved chunk text>

[2] Source: AI_Ethics.pdf | Page: 4 | Chunk: 13
<retrieved chunk text>
```

A separate, smaller prompt is used only to rewrite follow-up questions
into standalone form before retrieval:

```
Given the conversation history and a follow-up question, rewrite the follow-up question
as a standalone question that includes all context needed to understand it on its own,
without changing its meaning. If it is already standalone, return it unchanged. Reply
with ONLY the rewritten question — no preamble, no quotes, no explanation.
```

## Example Questions and Answers

**Q: What is Artificial Intelligence?**
> A: Artificial Intelligence (AI) is the simulation of human
> intelligence in machines that are programmed to think, learn, and
> make decisions.
>
> Sources: `AI_Ethics.pdf`, page 1, chunk 2 (similarity 0.83)

**Q: What are its main risks?** *(follow-up, rewritten internally to
"What are the main risks of Artificial Intelligence?")*
> A: The main risks include bias in decision-making, loss of privacy
> from data collection, and job displacement due to automation.
>
> Sources: `AI_Ethics.pdf`, page 5, chunk 18 (similarity 0.71)

**Q: Who won the FIFA World Cup?** *(out of scope for an AI-focused
document set)*
> A: I could not find this information in the documents.
>
> Sources: none

## Source Citation Method

Citations are **never generated by the LLM** — this is a deliberate
design choice to prevent hallucinated page numbers or invented
references. Instead:

- Every retrieved chunk carries real metadata (`source`, `page`,
  `chunk_number`) attached during the chunking step.
- After the LLM generates its answer, the system attaches the metadata
  of the exact chunks that were placed in its context window — the
  same list is shown to the user as the "Sources" panel.
- If the LLM refuses to answer (context was insufficient), no sources
  are attached, since attaching them would misleadingly imply they
  supported an answer that was never actually given.

## Known Limitations

- **PDF-only ingestion** — other formats (Word, HTML, plain text) are
  not currently supported by the loader.
- **Dense retrieval only** — no keyword/BM25 fallback, so an exact
  keyword match with a semantically distant phrasing can be missed.
- **No reranking step** — results are returned in raw nearest-neighbor
  order; a cross-encoder reranker would likely improve precision on
  larger document sets.
- **Fixed-size chunking** — chunk boundaries are character-based, not
  semantic, so a chunk can occasionally split a sentence or idea
  mid-thought (mitigated, but not eliminated, by chunk overlap).
- **In-memory conversation history only** — memory resets when the app
  restarts or the browser session ends; nothing is persisted to disk.
- **Cloud LLM dependency** — answer generation requires internet access
  and a valid `GROQ_API_KEY`; the app cannot generate answers offline
  (retrieval/indexing still work offline, since embeddings run
  locally). Groq's free tier is also rate-limited (requests/minute and
  requests/day caps), which can throttle rapid or high-volume use.
- **`bge-m3` is heavier than smaller embedding models** — better
  retrieval quality, but a larger download and slower embedding time
  than `bge-small`, especially noticeable on CPU-only machines during
  the first index build.
- **Relevance threshold is a blunt instrument** — a single global
  similarity cutoff doesn't adapt per-query; ambiguous or
  broadly-phrased questions may retrieve borderline-relevant chunks.

## Future Improvements

- Add hybrid search (dense vector + BM25 keyword scoring) for more
  robust retrieval on exact terms, names, or codes.
- Add a cross-encoder reranking step over the top-N candidates before
  selecting the final top-K context chunks.
- Support additional document formats (Word, HTML, plain text, CSV).
- Persist conversation history to disk or a database for cross-session
  continuity.
- Add streaming token-by-token answer display in the Streamlit UI.
- Add automatic evaluation (e.g. answer relevancy, faithfulness
  scoring) to measure RAG quality over a test question set.
- Add per-document access control / multi-user support for
  team or enterprise deployments.
- Add an offline fallback (e.g. local Ollama model) so the app degrades
  gracefully instead of failing outright if Groq is unreachable or the
  free-tier rate limit is hit.

## Required Deliverables

- [x] Complete, runnable source code (`src/`, `main.py`)
- [x] `requirements.txt` with pinned minimum versions
- [x] `.env.example` documenting all configuration variables
- [x] This `README.md`
- [x] `docs/architecture.md` describing the system's data flow
- [x] Automated tests (`tests/`) covering chunking, retrieval
      filtering, conversation memory, and QA orchestration logic
- [ ] Sample indexed document(s) and example question/answer transcript
      *(add your own under `data/raw/` and `results/`)*
- [ ] Short demo video or screenshots of the running application
      *(add if required by your submission guidelines)*
