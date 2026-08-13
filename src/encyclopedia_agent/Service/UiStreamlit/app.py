"""Streamlit chat UI for the encyclopedia RAG agent.

Run with: streamlit run src/encyclopedia_agent/Service/UiStreamlit/app.py

Uses absolute imports (not relative) because Streamlit executes this file
as __main__, not as a module inside the encyclopedia_agent package.
"""

import uuid

import streamlit as st
from deepagents.backends.utils import file_data_to_string
from langgraph.checkpoint.memory import InMemorySaver

from encyclopedia_agent.Service.EncyclopediaAgent.graph import build_agent

st.set_page_config(page_title="Encyclopedia Agent", page_icon="📚")


@st.cache_resource(show_spinner="Loading embedding model and connecting to Qdrant...")
def get_agent():
    return build_agent(checkpointer=InMemorySaver())


try:
    agent = get_agent()
except RuntimeError as exc:
    st.error(str(exc))
    st.info(
        "Set QDRANT_URL and QDRANT_API_KEY (and the chat model's API key) "
        "as environment variables, then restart the app."
    )
    st.stop()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("📚 Encyclopedia Agent")
    if st.button("Percakapan baru"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Tanya sesuatu tentang ensiklopedia...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        activity_placeholder = st.expander("Aktivitas", expanded=True).empty()
        answer_placeholder = st.empty()
        activity_lines: list[str] = []
        answer_text = ""

        def log_activity(line: str) -> None:
            activity_lines.append(line)
            activity_placeholder.markdown("\n".join(f"- {entry}" for entry in activity_lines))

        try:
            for namespace, mode, chunk in agent.stream(
                {"messages": [{"role": "user", "content": prompt}]},
                config,
                stream_mode=["updates", "messages"],
                subgraphs=True,
            ):
                if mode == "messages":
                    message_chunk, metadata = chunk
                    if namespace == () and metadata.get("langgraph_node") == "model":
                        piece = message_chunk.content
                        if piece:
                            answer_text += piece
                            answer_placeholder.markdown(answer_text + "▌")
                elif mode == "updates" and namespace == ():
                    for update in chunk.values():
                        if not update:
                            continue
                        for msg in update.get("messages", []):
                            for call in getattr(msg, "tool_calls", None) or []:
                                if call["name"] == "search_documentation":
                                    query = call["args"].get("query", "")
                                    log_activity(f"🔍 search: {query}")
                                elif call["name"] == "task":
                                    description = call["args"].get("description", "")
                                    first_line = description.splitlines()[0][:80] if description else ""
                                    log_activity(f"🧠 chunk-analyst: {first_line}")
                            if getattr(msg, "name", None) == "search_documentation":
                                first_line = (msg.content or "").splitlines()[0]
                                log_activity(f"📄 {first_line}")

            answer_placeholder.markdown(answer_text or "_(tidak ada jawaban)_")
        except Exception as exc:  # noqa: BLE001 - surface any agent failure in the UI
            st.error(f"Agent error: {exc}")
            answer_text = ""

        if answer_text:
            state = agent.get_state(config)
            files = (state.values or {}).get("files") or {}
            sources: list[str] = []
            for path, file_data in files.items():
                if not path.startswith("/retrieved/"):
                    continue
                content = file_data_to_string(file_data)
                if content.startswith("# Source:"):
                    sources.append(content.splitlines()[0].removeprefix("# Source:").strip())
            if sources:
                with st.expander("Sumber"):
                    for source in dict.fromkeys(sources):
                        st.markdown(f"- {source}")

    if answer_text:
        st.session_state.messages.append({"role": "assistant", "content": answer_text})
