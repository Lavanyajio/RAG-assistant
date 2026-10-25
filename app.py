"""Hybrid RAG Assistant — Gradio chat UI. Run: python app.py → http://localhost:7860"""
import logging
from pathlib import Path

# import gradio as gr

import streamlit as gr

from rag_assistant.config import settings
from rag_assistant.llm import LLMError
from rag_assistant.pipeline import RAGPipeline, format_sources

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

pipeline = RAGPipeline()
print("✅ Pipeline initialized successfully")


def index_status() -> str:
    if len(pipeline.index) == 0:
        return "**Index is empty.** Upload documents or load the samples."
    files = "\n".join(f"- {name}" for name in pipeline.index.sources)
    return f"**{len(pipeline.index)} chunks** from {len(pipeline.index.sources)} file(s):\n{files}"


def ingest_uploads(files):
    if not files:
        return "No files selected.", index_status()
    paths = [Path(f if isinstance(f, str) else f.name) for f in files]
    return pipeline.ingest(paths), index_status()


def ingest_samples():
    paths = sorted(p for p in settings.sample_docs_dir.glob("*") if p.is_file())
    return pipeline.ingest(paths), index_status()


def clear_index():
    pipeline.reset()
    return "Index cleared.", index_status()


def respond(message: str, history: list, top_k: int):
    message = (message or "").strip()
    if not message:
        return history, ""
    prior = list(history)
    history = prior + [{"role": "user", "content": message}]
    try:
        result = pipeline.answer(message, prior, int(top_k))
        history.append({"role": "assistant", "content": result.text})
        sources = format_sources(result)
        if sources:
            history.append({"role": "assistant", "content": sources,
                            "metadata": {"title": f"📚 Sources & retrieval details ({len(result.hits)})"}})
    except LLMError as exc:
        history.append({"role": "assistant", "content": f"⚠️ {exc}"})
    except Exception as exc:  # keep the UI alive on unexpected errors
        logger.exception("Answer failed")
        history.append({"role": "assistant", "content": f"⚠️ Something went wrong: {exc}"})
    return history, ""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Hybrid RAG Assistant",theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔎 Hybrid RAG Assistant\n"
                    "Semantic search (Hugging Face embeddings + FAISS) **and** keyword search (BM25), "
                    "fused with Reciprocal Rank Fusion. Every answer cites its sources.")
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                uploads = gr.File(label="Upload PDF / TXT / MD", file_count="multiple",
                                  file_types=[".pdf", ".txt", ".md"])
                with gr.Row():
                    ingest_btn = gr.Button("📥 Index files", variant="primary")
                    sample_btn = gr.Button("📘 Load sample docs")
                clear_btn = gr.Button("🗑️ Clear index", variant="stop")
                top_k = gr.Slider(2, 10, value=settings.top_k, step=1, label="Passages per answer (top-k)")
                ingest_log = gr.Textbox(label="Ingestion log", lines=4, interactive=False)
                status = gr.Markdown(index_status())
            with gr.Column(scale=2):
                # Simplified chatbot (removed unsupported args)
                chatbot = gr.Chatbot(type="messages", height=560, label="Chat",allow_tags=False)
                with gr.Row():
                    question = gr.Textbox(placeholder="Ask about your documents…", scale=5,
                                          show_label=False, autofocus=True)
                    send = gr.Button("Send", variant="primary", scale=1)
                gr.Examples(["What does error code E07 mean and how do I fix it?",
                             "How long does a full charge take?",
                             "Is battery damage from water covered by warranty?"], inputs=question)
                gr.ClearButton([chatbot, question], value="🧹 New chat")

        ingest_btn.click(ingest_uploads, uploads, [ingest_log, status])
        sample_btn.click(ingest_samples, None, [ingest_log, status])
        clear_btn.click(clear_index, None, [ingest_log, status])
        send.click(respond, [question, chatbot, top_k], [chatbot, question])
        question.submit(respond, [question, chatbot, top_k], [chatbot, question])
    return demo


# Build once, reuse demo
demo = build_ui()
print("✅ UI built successfully")

if __name__ == "__main__":
    # demo.queue().launch(server_name="127.0.0.1")
    build_ui().queue().launch(
        server_name=settings.host,
        server_port=settings.port,
        share=True
    )
