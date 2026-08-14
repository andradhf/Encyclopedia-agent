import uuid

from deepagents.backends import StateBackend
from langchain.tools import tool
from loguru import logger

from ...Infrstructure.vector_config import vector_store

backend = StateBackend()

@tool(parse_docstring=True)
def search_documentation(query: str) -> str:
    """Search the encyclopedia corpus and save matching chunks to the agent filesystem.

    Args:
        query: Natural language search query.

    Returns:
        File paths where retrieved chunks were saved under /retrieved/.
    """
    logger.info("search_documentation(query={!r})", query)
    retrieved_docs = vector_store.similarity_search(query, k=4)
    batch_id = uuid.uuid4().hex[:8]
    uploads: list[tuple[str, bytes]] = []
    saved_paths: list[str] = []

    for index, doc in enumerate(retrieved_docs, start=1):
        path = f"/retrieved/{batch_id}/chunk_{index}.md"
        headers = doc.metadata.get("headers") or {}
        heading = headers.get("h2") or headers.get("h1")
        source_bits = [str(doc.metadata.get("doc") or "unknown")]
        if heading:
            source_bits.append(heading)
        page_index = doc.metadata.get("page_index")
        if page_index is not None:
            source_bits.append(f"hal. {page_index + 1}")
        content = (
            f"# Source: {' | '.join(source_bits)}\n\n"
            f"{doc.page_content}"
        )
        uploads.append((path, content.encode("utf-8")))
        saved_paths.append(path)

    backend.upload_files(uploads)
    logger.info("search_documentation -> {} chunks saved to {}", len(saved_paths), saved_paths)
    return (
        f"Saved {len(saved_paths)} documentation chunks:\n"
        + "\n".join(saved_paths)
    )