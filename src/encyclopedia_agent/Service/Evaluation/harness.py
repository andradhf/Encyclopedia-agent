"""Invoke the encyclopedia agent for a single question and harvest RAGAS inputs.

Mirrors the state-extraction pattern already used by the Streamlit UI
(Service/UiStreamlit/app.py) to recover retrieved chunks: the
search_documentation tool doesn't return context text directly, it writes
chunks to the deep agent's virtual filesystem under /retrieved/.
"""

import uuid

from deepagents.backends.utils import file_data_to_string
from loguru import logger
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed

RECURSION_LIMIT = 100


def _is_transient_error(exc: BaseException) -> bool:
    message = str(exc)
    return (
        "RESOURCE_EXHAUSTED" in message
        or "429" in message
        or "Server disconnected" in message
    )


@retry(
    retry=retry_if_exception(_is_transient_error),
    stop=stop_after_attempt(4),
    wait=wait_fixed(20),
    reraise=True,
)
def _invoke_agent(agent, question: str, config: dict):
    return agent.invoke({"messages": [{"role": "user", "content": question}]}, config)


def run_question(agent, question: str) -> dict:
    """Run one question through a fresh thread and return RAGAS-ready fields.

    Returns a dict with keys: question, answer, contexts, error (error is
    None on success). A fresh thread_id per call is required so that
    /retrieved/ files from earlier questions in the same eval run don't
    leak into this question's contexts.

    The agent makes several LLM calls per question (main model + one
    subagent call per retrieved chunk), which burns through Gemini free-tier
    rate limits fast when running a whole dataset back to back and
    occasionally trips a dropped-connection error. Retry with a fixed
    backoff on both rather than dropping the question.
    """
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }

    try:
        result = _invoke_agent(agent, question, config)
        answer = result["messages"][-1].text

        state = agent.get_state(config)
        files = (state.values or {}).get("files") or {}
        contexts = []
        for path, file_data in files.items():
            if not path.startswith("/retrieved/"):
                continue
            content = file_data_to_string(file_data)
            if content.startswith("# Source:"):
                content = content.split("\n", 2)[-1].lstrip("\n")
            contexts.append(content)

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - one bad question must not abort the run
        logger.exception("run_question failed for {!r}", question)
        return {
            "question": question,
            "answer": "",
            "contexts": [],
            "error": str(exc),
        }
