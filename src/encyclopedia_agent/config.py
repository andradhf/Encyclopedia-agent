import os

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "encyclopedia")

# Must match the model used by the ingestion pipeline (see
# Service/IngestionPipeline) so query and passage vectors live in the same
# space.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def require_config() -> None:
    """Raise one readable error naming every missing required variable.

    Call this before touching Qdrant or loading the embedding model, so a
    missing env var fails fast instead of surfacing as a bare KeyError deep
    inside client construction.
    """
    missing = [
        name
        for name, value in (
            ("QDRANT_URL", QDRANT_URL),
            ("QDRANT_API_KEY", QDRANT_API_KEY),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): " + ", ".join(missing)
        )
