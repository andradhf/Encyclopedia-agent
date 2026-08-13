from langchain_core.documents import Document
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from ..config import (
    EMBEDDING_MODEL,
    EMBEDDING_QUERY_PREFIX,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
    require_config,
)


class QdrantEncyclopediaStore:
    """Query-time interface for the collection built by Service/IngestionPipeline.

    That pipeline wrote a flat payload (`text`, `doc`, `headers`, `pages`,
    `has_image`, `image_urls`) on the default unnamed vector, not
    langchain_qdrant's `page_content`/`metadata` schema, so we talk to
    qdrant_client directly instead of using QdrantVectorStore.
    """

    def __init__(self, client: QdrantClient, model: SentenceTransformer, collection: str) -> None:
        self._client = client
        self._model = model
        self._collection = collection

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        vector = self._model.encode(
            EMBEDDING_QUERY_PREFIX + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()
        results = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=k,
        ).points
        return [
            Document(
                page_content=point.payload.get("text", ""),
                metadata={
                    "doc": point.payload.get("doc"),
                    "headers": point.payload.get("headers"),
                    "pages": point.payload.get("pages"),
                    "has_image": point.payload.get("has_image"),
                    "image_urls": point.payload.get("image_urls"),
                },
            )
            for point in results
        ]


require_config()

_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
_model = SentenceTransformer(EMBEDDING_MODEL)

vector_store = QdrantEncyclopediaStore(_client, _model, QDRANT_COLLECTION)
