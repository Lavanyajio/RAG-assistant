import streamlit as st
from pathlib import Path
from rag_assistant.pipeline import RAGPipeline

st.title("🤖 Hybrid RAG Assistant")

@st.cache_resource
def load_pipeline():
    return RAGPipeline()

pipeline = load_pipeline()

# Load sample documents
if st.button("Load Sample Docs"):

    docs = list(Path("data/sample_docs").glob("*"))

    report = pipeline.ingest(docs)

    st.success("Documents Indexed!")
    st.text(report)

question = st.chat_input("Ask a question")

if question:

    result = pipeline.answer(question)

    st.write(result.text)