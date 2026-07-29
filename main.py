"""
main.py
-------
Entry point: a chat UI over your indexed PDF documents.

    build/update index -> ask a question -> retrieve -> LLM answers
    from context only -> show answer + citations -> remember for
    follow-ups

Run with:
    streamlit run main.py
"""

import streamlit as st

from src.config import settings
from src.memory import ConversationMemory
from src.pipeline import run_ingestion_pipeline
from src.qa_chain import answer_question
from src.vector_store import collection_count, reset_vector_store

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📚", layout="centered")
st.title("📚 Document Question Answering")
st.caption("Ask questions about your indexed PDFs. Answers are grounded only in retrieved context, with sources cited.")

# --------------------------------------------------------------------- #
# Session state: conversation memory + chat log (log includes sources,
# which ConversationMemory itself doesn't store).
# --------------------------------------------------------------------- #
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []  # list of {"role", "content", "sources"}

# --------------------------------------------------------------------- #
# Sidebar: index management
# --------------------------------------------------------------------- #
with st.sidebar:
    st.header("Index")
    st.write(f"Source folder: `{settings.raw_data_dir}`")
    try:
        st.write(f"Vectors indexed: **{collection_count()}**")
    except Exception:
        st.write("Vectors indexed: (none yet)")

    if st.button("Rebuild / update index", use_container_width=True):
        with st.spinner("Loading, chunking, and embedding documents..."):
            summary = run_ingestion_pipeline()
        st.success("Index updated.")
        st.json(summary)

    if st.button("Reset index (delete all vectors)", use_container_width=True):
        reset_vector_store()
        st.warning("Index deleted. Rebuild it before asking questions.")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.memory.clear()
        st.session_state.chat_log = []
        st.rerun()

    st.divider()
    st.subheader("Retrieval settings")
    top_k = st.slider("Top K chunks", min_value=1, max_value=10, value=settings.top_k)
    threshold = st.slider(
        "Relevance threshold",
        min_value=0.0,
        max_value=1.0,
        value=settings.relevance_threshold,
        step=0.05,
        help="Minimum similarity a chunk must reach to be used as context.",
    )

# --------------------------------------------------------------------- #
# Replay chat history
# --------------------------------------------------------------------- #
for entry in st.session_state.chat_log:
    with st.chat_message(entry["role"]):
        st.write(entry["content"])
        if entry.get("sources"):
            with st.expander(f"Sources ({len(entry['sources'])})"):
                for i, source in enumerate(entry["sources"], start=1):
                    st.markdown(
                        f"**[{i}] {source.document}** — page {source.page}, "
                        f"chunk {source.chunk_number} · relevance `{source.similarity:.3f}`"
                    )
                    st.caption(source.text)

# --------------------------------------------------------------------- #
# New question
# --------------------------------------------------------------------- #
question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.chat_log.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer_question(
                question,
                memory=st.session_state.memory,
                k=top_k,
                threshold=threshold,
            )
        st.write(result.answer)

        if result.sources:
            with st.expander(f"Sources ({len(result.sources)})"):
                for i, source in enumerate(result.sources, start=1):
                    st.markdown(
                        f"**[{i}] {source.document}** — page {source.page}, "
                        f"chunk {source.chunk_number} · relevance `{source.similarity:.3f}`"
                    )
                    st.caption(source.text)

    st.session_state.chat_log.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )
