from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

from .Prompt.template_analysis import CHUNK_ANALYST_INSTRUCTIONS
from .Prompt.template_delegation import SUBAGENT_DELEGATION_INSTRUCTIONS
from .Prompt.template_workflow import RAG_WORKFLOW_INSTRUCTIONS

from .tools import backend, search_documentation

max_concurrent_analysts = 3

INSTRUCTIONS = (
    RAG_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_analysts=max_concurrent_analysts,
    )
)

chunk_analyst_subagent = {
    "name": "chunk-analyst",
    "description": (
        "Analyze one retrieved documentation chunk file. "
        "Pass the user question and a single file path under /retrieved/."
    ),
    "system_prompt": CHUNK_ANALYST_INSTRUCTIONS,
}

def build_agent(checkpointer=None):
    """Build a fresh compiled deep agent graph.

    A factory rather than a module-level singleton so callers (e.g. the
    Streamlit UI) can inject a checkpointer for multi-turn memory.
    """
    model = init_chat_model(model="google_genai:gemini-3.1-flash-lite")
    return create_deep_agent(
        model=model,
        tools=[search_documentation],
        backend=backend,
        system_prompt=INSTRUCTIONS,
        subagents=[chunk_analyst_subagent],
        checkpointer=checkpointer,
    )