# Architecture

This document describes how data flows through the Document-Based AI
Question Answering System, from raw PDFs on disk to a cited, grounded
answer in the chat UI. Diagrams below use [Mermaid](https://mermaid.js.org/)
— they render automatically on GitHub, GitLab, VS Code (with the Mermaid
extension), Obsidian, and most modern Markdown viewers.

---

## 1. Ingestion Pipeline

Turns raw PDFs into searchable vectors in ChromaDB.

```mermaid
flowchart LR
    A["📄 data/raw/*.pdf\n(one or more PDFs)"]:::input
    B["loaders.py\nload_pdfs()\n→ 1 Document per page"]:::process
    C["splitter.py\nsplit_documents()\n→ ~500-char chunks\n+ source/page/chunk_number tags"]:::process
    D["embeddings.py\nget_embedding_model()\nBAAI/bge-m3"]:::model
    E[("🗄️ vector_store.py\nChromaDB\ncosine-indexed, deduplicated")]:::storage

    A --> B --> C --> D --> E

    classDef input fill:#FFE0B2,stroke:#FB8C00,stroke-width:2px,color:#4A2C00
    classDef process fill:#BBDEFB,stroke:#1E88E5,stroke-width:2px,color:#0D2B4E
    classDef model fill:#E1BEE7,stroke:#8E24AA,stroke-width:2px,color:#3A0A47
    classDef storage fill:#C8E6C9,stroke:#43A047,stroke-width:2px,color:#173C1D
```

`pipeline.py` runs all four steps above, in order, in a single call —
it's the only module that touches every stage of ingestion. Re-running
it is safe: chunks are deduplicated by a deterministic
`source_page_chunknumber` ID, so nothing gets indexed twice.

`bge-m3` runs locally via `sentence-transformers` — no API key or
network call needed for embedding, only for the LLM step later.

---

## 2. Question-Answering Flow

Turns a user's question into a grounded, cited answer — or an honest
refusal.

```mermaid
flowchart TD
    Q["💬 User question"]:::input
    M{"Conversation\nhistory exists?"}:::decision
    CTX["qa_chain.py\n_contextualize_question()\nrewrite follow-up →\nstandalone question"]:::model
    R["retriever.py\nretrieve()\ntop-K nearest chunks\nby cosine similarity"]:::process
    T{"Any chunk clears\nRELEVANCE_THRESHOLD?"}:::decision
    NF["🚫 Return refusal:\n'I could not find this\ninformation in the documents.'\n(no LLM call)"]:::refuse
    CB["qa_chain.py\n_build_context_block()\nnumbered, cited excerpts"]:::process
    LLM["llm.py → get_llm()\nGroq API\nllama-3.3-70b-versatile\nanswers strictly from context"]:::model
    IS{"LLM says it\ncan't answer?"}:::decision
    NF2["🚫 Return refusal\n(sources dropped)"]:::refuse
    OK["✅ QAResult\nanswer + sources\n(document, page, chunk, text)"]:::output
    MEM[("🧠 memory.py\nConversationMemory\nadd_turn(question, answer)")]:::storage

    Q --> M
    M -- yes --> CTX --> R
    M -- no --> R
    R --> T
    T -- no --> NF --> MEM
    T -- yes --> CB --> LLM --> IS
    IS -- yes --> NF2 --> MEM
    IS -- no --> OK --> MEM

    classDef input fill:#FFE0B2,stroke:#FB8C00,stroke-width:2px,color:#4A2C00
    classDef process fill:#BBDEFB,stroke:#1E88E5,stroke-width:2px,color:#0D2B4E
    classDef model fill:#E1BEE7,stroke:#8E24AA,stroke-width:2px,color:#3A0A47
    classDef storage fill:#C8E6C9,stroke:#43A047,stroke-width:2px,color:#173C1D
    classDef decision fill:#FFF9C4,stroke:#F9A825,stroke-width:2px,color:#4A3B00
    classDef refuse fill:#FFCDD2,stroke:#E53935,stroke-width:2px,color:#5A0F0C
    classDef output fill:#B2DFDB,stroke:#00897B,stroke-width:2px,color:#003D33
```

The LLM step (`CB → LLM`) is the only network call to Groq in the
entire flow — retrieval, thresholding, and memory are all local.

---

## 3. Technology Stack

```mermaid
flowchart TB
    subgraph UI["🖥️ Application Layer"]
        direction LR
        S["Streamlit\nmain.py"]:::app
    end

    subgraph RAG["🔎 RAG Core"]
        direction LR
        L["loaders.py\nPyPDFLoader"]:::core
        SP["splitter.py\nRecursiveCharacterTextSplitter"]:::core
        RT["retriever.py"]:::core
        QC["qa_chain.py"]:::core
        MEM["memory.py"]:::core
    end

    subgraph DATA["💾 Data Layer (local, no API key)"]
        direction LR
        EMB["HuggingFace\nBAAI/bge-m3"]:::data
        VDB[("ChromaDB\ncosine index")]:::data
    end

    subgraph LLMS["🤖 LLM Layer (cloud, requires GROQ_API_KEY)"]
        direction LR
        GR["Groq API\nllama-3.3-70b-versatile"]:::llm
        OL["Ollama (local)\noptional fallback"]:::llmalt
        OA["OpenAI / Anthropic\noptional"]:::llmalt
    end

    S --> RAG
    RAG --> DATA
    QC --> LLMS

    classDef app fill:#FFE0B2,stroke:#FB8C00,stroke-width:2px,color:#4A2C00
    classDef core fill:#BBDEFB,stroke:#1E88E5,stroke-width:2px,color:#0D2B4E
    classDef data fill:#C8E6C9,stroke:#43A047,stroke-width:2px,color:#173C1D
    classDef llm fill:#E1BEE7,stroke:#8E24AA,stroke-width:2px,color:#3A0A47
    classDef llmalt fill:#F3E5F5,stroke:#CE93D8,stroke-width:2px,color:#4A1458
```

`llm.py` is provider-agnostic (LangChain's `init_chat_model`) — Groq
is the active default, but Ollama/OpenAI/Anthropic remain one `.env`
change away with no code changes.

---

## Why Citations Are Never LLM-Generated

The LLM only ever produces the **answer text**. Every `Source` in the
final result — document name, page, chunk number, retrieved text —
comes directly from `RetrievedChunk` objects assembled in
`retriever.py`, **before** the LLM is even called.

```mermaid
flowchart LR
    A["Retrieved chunks\n(real metadata)"]:::real --> B["Context block\nsent to LLM"]:::real
    B --> C["Groq LLM generates\nANSWER TEXT ONLY"]:::model
    A -.->|"citations built\ndirectly from here,\nnot from the LLM"| D["Sources shown\nto user"]:::output

    classDef real fill:#C8E6C9,stroke:#43A047,stroke-width:2px,color:#173C1D
    classDef model fill:#E1BEE7,stroke:#8E24AA,stroke-width:2px,color:#3A0A47
    classDef output fill:#B2DFDB,stroke:#00897B,stroke-width:2px,color:#003D33
```

This guarantees citations are always real: the LLM has no opportunity
to invent a page number or misquote a source, because it never decides
what the citations are — it only decides what to say using the context
it was given.

---

## Two Independent Refusal Paths

| # | Path | Where it happens | LLM called? |
|---|---|---|---|
| 1 | 🚫 **No relevant chunks retrieved** | `retriever.py` — nothing clears `RELEVANCE_THRESHOLD` | ❌ No — saves an API call, guarantees no hallucinated answer when the corpus doesn't cover the topic |
| 2 | 🚫 **LLM decides context is insufficient** | `qa_chain.py` — `_is_refusal()` detects the LLM's own refusal | ✅ Yes — chunks were retrieved, but didn't actually answer *this specific* question |

In both cases, the user sees the same message:
> *"I could not find this information in the documents."*

---

## Answer Style Enforcement

Beyond grounding and citations, the system prompt also enforces *how*
the LLM opens each answer. Smaller/faster conversational models like
`llama-3.3-70b-versatile` tend to default to confirmatory filler
("That is correct. According to the context, ..."), which reads
awkwardly for a fresh question. The prompt explicitly bans this class
of phrasing and shows a wrong/right example, so the model states facts
directly instead of reacting to them.

```mermaid
flowchart LR
    P["System prompt:\nban confirmation phrases\n+ wrong/right example"]:::model
    Q["Question: 'Where was he born?'"]:::input
    W["❌ 'That is correct.\nAccording to the context,\nhe was born in...'"]:::refuse
    G["✅ 'He was born in\nPortsmouth, England, in 1962.'"]:::output

    P --> Q
    Q -.->|without rule| W
    Q -->|with rule| G

    classDef input fill:#FFE0B2,stroke:#FB8C00,stroke-width:2px,color:#4A2C00
    classDef model fill:#E1BEE7,stroke:#8E24AA,stroke-width:2px,color:#3A0A47
    classDef refuse fill:#FFCDD2,stroke:#E53935,stroke-width:2px,color:#5A0F0C
    classDef output fill:#B2DFDB,stroke:#00897B,stroke-width:2px,color:#003D33
```

---

## Conversation Memory

`ConversationMemory` (`memory.py`) is a simple in-memory, per-session
rolling window of the last `MAX_HISTORY_TURNS` (question, answer)
pairs — no database, no persistence across restarts. In the Streamlit
app, one instance lives in `st.session_state` per browser session.

```mermaid
flowchart LR
    T1["Turn 1\nQ: What is AI?\nA: ..."]:::turn
    T2["Turn 2\nQ: What about its risks?\nA: ..."]:::turn
    T3["Turn 3 (new)\nQ: How can they be reduced?"]:::new

    T1 --> T2 --> T3
    T3 -.->|"history used to\nrewrite into a\nstandalone question"| CTX["'How can the risks\nof AI be reduced?'"]:::result

    classDef turn fill:#BBDEFB,stroke:#1E88E5,stroke-width:2px,color:#0D2B4E
    classDef new fill:#FFE0B2,stroke:#FB8C00,stroke-width:2px,color:#4A2C00
    classDef result fill:#C8E6C9,stroke:#43A047,stroke-width:2px,color:#173C1D
```

It's used in exactly one place: `_contextualize_question()` sends the
history plus the new question to the LLM (Groq) and asks it to rewrite
ambiguous follow-ups into standalone questions **before** retrieval
runs. Retrieval and the final answer are otherwise stateless — memory
affects what gets searched for, not what "counts" as relevant.

---

## Local vs. Cloud Boundary

Worth calling out explicitly, since it affects offline behavior and
data privacy: only the LLM step leaves your machine.

```mermaid
flowchart TB
    subgraph LOCAL["💻 Runs entirely on your machine"]
        direction LR
        L1["PDF loading"]:::localbox
        L2["Chunking"]:::localbox
        L3["bge-m3 embedding"]:::localbox
        L4["ChromaDB storage & search"]:::localbox
        L5["Relevance thresholding"]:::localbox
        L6["Conversation memory"]:::localbox
    end

    subgraph CLOUD["☁️ Sent to Groq's servers"]
        direction LR
        C1["Retrieved context\n+ your question"]:::cloudbox
    end

    LOCAL -->|"only the final\ncontext + question"| CLOUD

    classDef localbox fill:#C8E6C9,stroke:#43A047,stroke-width:2px,color:#173C1D
    classDef cloudbox fill:#E1BEE7,stroke:#8E24AA,stroke-width:2px,color:#3A0A47
```

Indexing and retrieval work fully offline; only asking a question
requires internet access and a valid `GROQ_API_KEY`.