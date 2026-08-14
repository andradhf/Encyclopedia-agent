"""Workaround for a ragas 0.4.3 import bug.

ragas.llms.base unconditionally imports ChatVertexAI from
langchain_community.chat_models.vertexai, a path that no longer exists in
langchain-community>=0.4 (ChatVertexAI moved to langchain-google-vertexai).
The class is only used for an isinstance() check, so a stub is enough.
Remove this once upstream ragas fixes the import (tracked as
explodinggradients/ragas#2745).

Must be called before the first `import ragas` anywhere in the process.
"""

import sys
import types


def patch_ragas_imports() -> None:
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    stub = types.ModuleType(module_name)

    class ChatVertexAI:  # noqa: N801 - matches the real class name
        pass

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = stub
